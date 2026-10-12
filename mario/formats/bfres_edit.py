"""
Grow a BFRES file: add bones to a skeleton, bone tracks to animation clips, and whole new clips.

`mario.formats.bfres` edits a model where it lies, and that cannot add anything. This module adds,
in the one way that leaves the rest of the file as it was.

The layout. A BFRES file is cut into sections. The first, the main section, holds the header and
every structure (models, skeletons, materials, clips) and ends with the pool of names. After it
come the vertex and index buffers, the shader archive and the textures, and last of all a
relocation table (`_RLT`). The header says where the table is and how long the file is.

The new block. Everything added (bone arrays, name dictionaries, names, clip tracks) goes into one
block placed straight after the main section, which is made longer to take it in. What used to
follow is moved back by a whole number of pages to make room, byte for byte unchanged. Offsets into
the buffers count from the buffer pool and not from the start of the file, so they stay true; only
real pointers into the moved part change, and the relocation table lists every one of them.

An array cannot be made longer where it is. So a structure that has to grow (a skeleton's bone
array, a clip's array of bone tracks, the file's list of clips) is copied into the new block with
the additions on its end, and the one pointer to it is redirected. The old copy is left where it
was, unused.

The relocation table. A pointer is stored as an offset from the start of the file. The game turns
it into an address when it loads the file, but only if the table lists it: a pointer the table
leaves out is never converted, and the game crashes when it follows it. The table gives each
section's position and size, and for each section a run of entries. An entry stands for many
pointers at once: (where the first is, how many structures, how many pointers in each, how many
8-byte words to skip after each). A pointer is listed under the section it points into. Here the
old entries are kept as they are and entries are added for every pointer in the new block:
`Grower.pointer` notes each one as it writes it. `check` tests that every listed pointer lands in
its section, and `unlisted` walks a finished file's skeletons and clips for pointers the table
lacks.

Name dictionaries. Every list of named things (a skeleton's bones, the file's clips) has a
dictionary beside it, by which the game finds a thing from its name, so a longer list needs a new
one. A dictionary is a Patricia trie over the bits of the names, counted from the end of the
string: bit 0 is the low bit of the last character. (The game's own lookup is inline in its
clip-binding code, FUN_60fba940, and reads them that way.) It is stored as the number of names and
then 16-byte entries: a bit number, a left and a right link, and a pointer to a name. Entry 0 is
the root and entry i + 1 belongs to item i. A lookup starts at the root and at each entry takes
the left or right link by the name's bit of that number, until a link leads to an entry whose bit
number is no higher; that entry's name is the only one it can be. `build_dict` makes the trie for
a list of names by adding them in order. For the lists of the game's own file it gives exactly the
entries stored there, so the one it makes for a longer list is the one the game expects.

Like `mario.formats.bfres`, this follows BFRES version 0.8.1.1, the one Pikmin 3 Deluxe uses.
"""

from __future__ import annotations

import struct

from typing import ClassVar

from mario.formats.bfres import Bfres

PAGE = 0x1000


def _bit(key: bytes, n: int) -> int:
    """Bit n of a name, counted from the low bit of its last character; past the start it is 0."""
    i = n >> 3
    return (key[len(key) - 1 - i] >> (n & 7)) & 1 if i < len(key) else 0


def _first_difference(a: bytes, b: bytes) -> int:
    """The lowest bit in which two names differ."""
    n = 0
    while _bit(a, n) == _bit(b, n):
        n += 1
        if n > 8 * max(len(a), len(b)):
            raise ValueError(f"duplicate name {a!r} in a dictionary")
    return n


def build_dict(names: list[str]) -> list[tuple[int, int, int]]:
    """The trie for `names` in index order: [(reference bit, left, right)], entry 0 the root."""
    keys = [b""] + [n.encode("utf-8") for n in names]
    ref, left, right = [-1], [0], [0]
    for t in range(1, len(keys)):
        key = keys[t]
        # walk down to the name already in the trie that this one would be taken for
        p, x = 0, left[0]
        while ref[x] > ref[p]:
            p, x = x, (right[x] if _bit(key, ref[x]) else left[x])
        b = _first_difference(key, keys[x])  # the bit that tells the two apart
        # walk down again and stop where the new entry belongs: above the first that tests a higher bit than that
        p, x = 0, left[0]
        while ref[x] > ref[p] and ref[x] < b:
            p, x = x, (right[x] if _bit(key, ref[x]) else left[x])
        ref.append(b)
        left.append(x if _bit(key, b) else t)
        right.append(t if _bit(key, b) else x)
        if p == 0 or not _bit(key, ref[p]):
            left[p] = t
        else:
            right[p] = t
    return [(ref[i] & 0xFFFFFFFF, left[i], right[i]) for i in range(len(keys))]


def dict_find(f: Bfres, off: int, name: str) -> int:
    """Look a name up in the dictionary at `off` the way the trie is meant to be walked; its index, or -1."""
    key = name.encode("utf-8")

    def entry(i: int) -> tuple[int, int, int, int]:
        return struct.unpack_from("<iHHQ", f.d, off + 8 + i * 16)

    # Follow links for as long as they lead to a higher bit number; the entry where that stops is
    # the only one the name can be.
    above = -1
    x = entry(0)[1]
    ref, left, right, name_off = entry(x)
    while ref > above:
        above = ref
        x = right if _bit(key, ref) else left
        ref, left, right, name_off = entry(x)
    return x - 1 if x and f.s(name_off) == name else -1


class Grower:
    """
    A BFRES file being added to. Call the `add_` methods, then `finish` for the new file.

    `base` is where the new block starts and `new` the block so far, so the offsets `alloc` hands
    out are true in the finished file. `tail` is the first byte of the old file that will move, and
    `shift` how far it went, known once `finish` has run. `slots` maps the place of every pointer
    the relocation table lists to the number of the section it points into. `touched` lists the
    spans (start, end) of the old file that were overwritten.
    """

    # curve target (an offset into a bone's scale, translation, rotation) -> the track flag that says it has a curve
    TARGETS: ClassVar[dict[int, int]] = {
        0x4: 0x40, 0x8: 0x80, 0xC: 0x100, 0x20: 0x200, 0x24: 0x400, 0x28: 0x800,
        0x10: 0x2000, 0x14: 0x4000, 0x18: 0x8000,
    }  # fmt: skip

    def __init__(self, data: bytes | bytearray) -> None:
        self.d = bytearray(data)
        self.f = f = Bfres(self.d)
        self.rlt = rlt = f.u32(0x18)
        if self.d[rlt : rlt + 4] != b"_RLT":
            raise ValueError("no relocation table where the header says")
        nsec = f.u32(rlt + 8)
        # a section: [unused, position, size, number of its first entry, how many entries]
        self.sections = [list(struct.unpack_from("<QIIii", self.d, rlt + 0x10 + i * 0x18)) for i in range(nsec)]
        base = rlt + 0x10 + nsec * 0x18
        total = sum(s[4] for s in self.sections)
        # an entry: (position, structures, pointers in each, 8-byte words skipped after each)
        self.entries = [struct.unpack_from("<IHBB", self.d, base + i * 8) for i in range(total)]
        if base + total * 8 != len(self.d) or f.u32(0x1C) != len(self.d):
            raise ValueError("the relocation table is not the end of the file")
        # Where every listed pointer is, and which section it points into.
        self.slots = {}
        for s, (_, _, _, first, count) in enumerate(self.sections):
            for pos, structs, pointers, skip in self.entries[first : first + count]:
                for _ in range(structs):
                    for _ in range(pointers):
                        self.slots[pos] = s
                        pos += 8
                    pos += 8 * skip
        main_end = self.sections[0][1] + self.sections[0][2]
        self.tail = self.sections[1][1]  # the first byte that will move
        if not main_end <= self.tail or any(s[1] < self.tail for s in self.sections[1:]):
            raise ValueError("sections are not laid out main first")
        self.base = (main_end + 7) & ~7  # where the new block starts
        self.new = bytearray()
        self.added = []  # pointer slots to add to the table (all point into the main section)
        self.strings = {}
        # spans of the old file this has overwritten: first the header's table offset and file size
        self.touched = [(0x18, 0x20)]
        self.shift = 0  # how far the moved part went; set by finish()
        if any(slot >= self.tail for slot in self.slots):
            raise ValueError("a listed pointer lies in the part of the file that would move")

    # -- writing the new block -------------------------------------------------------------------

    def alloc(self, data: bytes | bytearray, align: int = 8) -> int:
        """Append bytes to the new block; returns their file offset."""
        self.new += bytes(-len(self.new) % align)
        off = self.base + len(self.new)
        self.new += data
        return off

    def poke(self, off: int, data: bytes | bytearray) -> None:
        """Overwrite bytes in the new block, or anywhere before the part of the file that moves."""
        if off >= self.base:
            self.new[off - self.base : off - self.base + len(data)] = data
        elif off + len(data) <= self.tail:
            self.d[off : off + len(data)] = data
            self.touched.append((off, off + len(data)))
        else:
            raise ValueError(f"{off:#x} is in the part of the file that moves")

    def peek(self, off: int, n: int) -> bytes:
        """`n` bytes at `off`, in the old file or the new block, as they now stand."""
        if off >= self.base:
            return bytes(self.new[off - self.base : off - self.base + n])
        return bytes(self.d[off : off + n])

    def pointer(self, slot: int, target: int) -> None:
        """Store a pointer and make sure the relocation table lists its slot."""
        self.poke(slot, struct.pack("<Q", target))
        if slot not in self.slots:
            self.slots[slot] = 0
            self.added.append(slot)

    def string(self, text: str) -> int:
        """The offset of a name in the file's format (length, bytes, terminator); a new one is added once."""
        if text not in self.strings:
            raw = text.encode("utf-8")
            self.strings[text] = self.alloc(struct.pack("<H", len(raw)) + raw + b"\0", 2)
        return self.strings[text]

    def name_at(self, off: int) -> str:
        """The name stored at `off`."""
        n = self._u16(off)
        return self.peek(off + 2, n).decode("utf-8")

    def _u16(self, off: int) -> int:
        return struct.unpack("<H", self.peek(off, 2))[0]

    def _i32(self, off: int) -> int:
        return struct.unpack("<i", self.peek(off, 4))[0]

    def _u64(self, off: int) -> int:
        return struct.unpack("<Q", self.peek(off, 8))[0]

    def _dictionary(self, names: list[str], name_offs: list[int], old: int) -> int:
        """Write the dictionary for a list of names that replaces the one at `old`; returns its offset."""
        empty = self._u64(old + 8 + 8)  # the root entry's (empty) name
        trie = build_dict(names)
        dic = self.alloc(struct.pack("<II", 0, len(names)) + bytes(16 * len(trie)))
        for i, (ref, left, right) in enumerate(trie):
            self.poke(dic + 8 + i * 16, struct.pack("<IHH", ref, left, right))
            self.pointer(dic + 8 + i * 16 + 8, name_offs[i - 1] if i else empty)
        return dic

    # -- skeletons -------------------------------------------------------------------------------

    def add_bones(self, skel_off: int, bones: list[tuple]) -> tuple[int, int, int, list[int]]:
        """
        Append bones to a skeleton. Each is (name, parent index, rotation xyz, translation, skinned).

        A skinned bone is given the next matrix number. Its inverse matrix is left zero for the
        caller to fill.

        Returns:
            (index of the first new bone, bone array offset, inverse array offset, matrix -> bone list).

        Raises:
            ValueError: If the skeleton has rigid-skinned bones, or already has a bone of that name.
        """
        u16, u64 = self._u16, self._u64
        nbone, nsmooth, nrigid = u16(skel_off + 0x4C), u16(skel_off + 0x4E), u16(skel_off + 0x50)
        if nrigid:
            raise ValueError("skeleton has rigid-skinned bones; not handled")
        old_bones, old_list, old_inv = u64(skel_off + 0x18), u64(skel_off + 0x20), u64(skel_off + 0x28)
        names = [self.name_at(u64(old_bones + i * 0x60)) for i in range(nbone)]
        matrix_bones = [u16(old_list + 2 * i) for i in range(nsmooth)]
        blob = bytearray(self.peek(old_bones, nbone * 0x60))
        name_offs = [u64(old_bones + i * 0x60) for i in range(nbone)]
        for k, (name, parent, rot, trans, skinned) in enumerate(bones):
            if name in names:
                raise ValueError(f"bone {name} already exists")
            index = nbone + k
            smooth = 0xFFFF
            if skinned:
                smooth = len(matrix_bones)
                matrix_bones.append(index)
            b = bytearray(0x60)
            # visible, Euler angles, scale of one; no "is zero" hints, so nothing is skipped
            struct.pack_into("<HHHHHHI", b, 0x28, index, parent, smooth, 0xFFFF, 0xFFFF, 0, 0x03001001)
            struct.pack_into("<3f4f3f", b, 0x38, 1.0, 1.0, 1.0, rot[0], rot[1], rot[2], 1.0, *trans)
            blob += b
            names.append(name)
            name_offs.append(self.string(name))
        new_bones = self.alloc(blob)
        for i, off in enumerate(name_offs):
            self.pointer(new_bones + i * 0x60, off)
            for extra in (8, 0x10):  # user data: listed in the table for every stock bone too
                self.pointer(new_bones + i * 0x60 + extra, u64(new_bones + i * 0x60 + extra))
        dic = self._dictionary(names, name_offs, u64(skel_off + 0x10))
        new_list = self.alloc(struct.pack(f"<{len(matrix_bones)}H", *matrix_bones))
        new_inv = self.alloc(self.peek(old_inv, nsmooth * 48) + bytes(48 * (len(matrix_bones) - nsmooth)))
        self.pointer(skel_off + 0x10, dic)
        self.pointer(skel_off + 0x18, new_bones)
        self.pointer(skel_off + 0x20, new_list)
        self.pointer(skel_off + 0x28, new_inv)
        self.poke(skel_off + 0x4C, struct.pack("<HH", len(names), len(matrix_bones)))
        return nbone, new_bones, new_inv, matrix_bones

    def set_skin_bones(self, shape_off: int, bones: list[int]) -> None:
        """Replace a shape's list of the bones its vertices follow."""
        lst = self.alloc(struct.pack(f"<{len(bones)}H", *bones))
        self.pointer(shape_off + 0x28, lst)
        self.poke(shape_off + 0x64, struct.pack("<H", len(bones)))

    # -- animation clips ---------------------------------------------------------------------------

    def add_clips(self, clips: list[tuple[str, int, int]]) -> list[int]:
        """
        Append clips to the file's list. Each is (name, last frame, offset of a clip to model it on).

        A new clip starts with no tracks; `add_tracks` gives it some. The list is an array, so it
        is copied into the new block with the new entries on its end. Every change to one of the
        file's own clips must be made before this is called: afterwards the copy is the list, and
        the old array, where `Bfres.skel_anims()` of the original file points, is dead. Use the
        offsets this returns from then on.

        Returns:
            The new clips' offsets, in order.

        Raises:
            ValueError: If the file already has a clip of that name, or the model is not one of its clips.
        """
        u64 = self._u64
        n = self._u16(0xBE)
        old, old_dic = u64(0x38), u64(0x40)
        names = [self.name_at(u64(old + i * 0x60 + 0x10)) for i in range(n)]
        blob = bytearray(self.peek(old, n * 0x60))
        listed = [[k for k in range(0, 0x60, 8) if old + i * 0x60 + k in self.slots] for i in range(n)]
        name_offs = [u64(old + i * 0x60 + 0x10) for i in range(n)]
        for name, frames, model in clips:
            if name in names:
                raise ValueError(f"clip {name} already exists")
            if not old <= model < old + n * 0x60 or (model - old) % 0x60:
                raise ValueError(f"{model:#x} is not a clip of this file")
            entry = bytearray(self.peek(model, 0x60))
            # its own name; the model's source path; nothing bound, no tracks, no user data
            for k in (0x20, 0x28, 0x30, 0x38, 0x40):
                struct.pack_into("<Q", entry, k, 0)
            struct.pack_into("<iiiHH", entry, 0x4C, frames, 0, 0, 0, 0)
            listed.append(listed[(model - old) // 0x60])
            names.append(name)
            name_offs.append(self.string(name))
            struct.pack_into("<Q", entry, 0x10, name_offs[-1])
            blob += entry
        arr = self.alloc(blob)
        for i, slots in enumerate(listed):
            for k in slots:
                self.pointer(arr + i * 0x60 + k, u64(arr + i * 0x60 + k))
        dic = self._dictionary(names, name_offs, old_dic)
        self.pointer(0x38, arr)
        self.pointer(0x40, dic)
        self.poke(0xBE, struct.pack("<H", len(names)))
        return [arr + (n + j) * 0x60 for j in range(len(clips))]

    def add_tracks(self, clip_off: int, tracks: list[tuple]) -> None:
        """
        Append bone tracks to a clip. Each is (bone name, scale, rotation xyz, translation, curves).

        The three values are the track's constants; a translation of None leaves the bone where its
        model puts it, as the game's clips do for every bone but the root. `curves` maps a target
        (4/8/0xc scale x/y/z, 0x20/0x24/0x28 rotation, 0x10/0x14/0x18 translation) to one value for
        each whole frame from 0 to the clip's last. Curves are stored the way a fifth of the game's
        own are: byte frame numbers and float cubic segments, here straight lines from one frame to
        the next.

        Raises:
            ValueError: If the clip has a track for that bone already, or a curve is not one this can write.
        """
        frames, ncurve, baked = self._i32(clip_off + 0x4C), self._i32(clip_off + 0x50), self._i32(clip_off + 0x54)
        nbone = self._u16(clip_off + 0x58)
        old = self._u64(clip_off + 0x30)
        blob = bytearray(self.peek(old, nbone * 0x28))
        # (offset inside the array, target) of each pointer in it
        pointers = [(i * 0x28 + k, self._u64(old + i * 0x28 + k)) for i in range(nbone) for k in (0, 8, 0x10)]
        present = {self.name_at(self._u64(old + i * 0x28)) for i in range(nbone)}
        for name, scale, rot, trans, curves in tracks:
            if name in present:
                raise ValueError(f"clip already has a track for {name}")
            if set(curves) - set(self.TARGETS):
                raise ValueError(f"{name}: no such curve target {sorted(set(curves) - set(self.TARGETS))}")
            if trans is None and any(t in curves for t in (0x10, 0x14, 0x18)):
                raise ValueError(f"{name}: a translation curve needs a translation")
            order = [t for t in self.TARGETS if t in curves]  # scale, rotation, translation: the game's order
            flags = 0x18 if trans is None else 0x38  # which constants follow: scale and rotation, translation
            if tuple(scale) == (1.0, 1.0, 1.0) and not any(t < 0x10 for t in order):
                flags |= 0x03000000  # the hint "its scale is one", as the game's own tracks carry it
            for t in order:
                flags |= self.TARGETS[t]
            constants = struct.pack("<3f4f", scale[0], scale[1], scale[2], rot[0], rot[1], rot[2], 1.0)
            base = self.alloc(constants + (struct.pack("<3f", *trans) if trans is not None else b""))
            curve_arr = 0
            if order:
                if frames > 255:
                    raise ValueError("clip is longer than a byte frame number allows")
                curve_arr = self.alloc(bytes(0x30 * len(order)))
                for c, t in enumerate(order):
                    if len(curves[t]) != frames + 1:
                        raise ValueError(f"{name}: a curve needs {frames + 1} values, got {len(curves[t])}")
                    self._curve(curve_arr + c * 0x30, t, curves[t], frames)
                    # what the clip's header counts for a curve, should the game ever bake them
                    baked += ((4 * (frames + 2) + 7) & ~7) + 24
            entry = bytearray(0x28)
            struct.pack_into("<IBBBBi", entry, 0x18, flags, 3, 6, len(order), 7, ncurve)
            pointers += [(len(blob), self.string(name)), (len(blob) + 8, curve_arr), (len(blob) + 0x10, base)]
            blob += entry
            ncurve += len(order)
            nbone += 1
            present.add(name)
        arr = self.alloc(blob)
        for inner, target in pointers:
            self.pointer(arr + inner, target)
        bind = self.alloc(b"\xff\xff" * nbone)  # the bone each track is bound to: none yet
        self.pointer(clip_off + 0x28, bind)
        self.pointer(clip_off + 0x30, arr)
        self.poke(clip_off + 0x50, struct.pack("<ii", ncurve, baked))
        self.poke(clip_off + 0x58, struct.pack("<H", nbone))

    def _curve(self, off: int, target: int, vals: list[float], frames: int) -> None:
        """Fill in the curve at `off` with one key for each frame: its value there, and the step to the next."""
        fr = self.alloc(bytes(range(frames + 1)), 4)
        keys = bytearray()
        for i, v in enumerate(vals):
            step = vals[i + 1] - v if i < frames else 0.0
            keys += struct.pack("<4f", v, step, 0.0, 0.0)
        ky = self.alloc(keys, 4)
        self.pointer(off, fr)
        self.pointer(off + 8, ky)
        # byte frames and float cubic keys; a key for each frame; then the first and last frame,
        # the keys' scale and offset, and how far the value goes from one end to the other
        header = struct.pack("<HHIfffff", 0x2, frames + 1, target, 0.0, float(frames), 1.0, 0.0, vals[-1] - vals[0])
        self.poke(off + 0x10, header)

    # -- putting the file back together ----------------------------------------------------------

    def finish(self) -> bytes:
        """The grown file. With nothing added it is the input, byte for byte."""
        gap = self.tail - self.base
        shift = self.shift = max(0, (len(self.new) - gap + PAGE - 1) // PAGE * PAGE)
        d = self.d
        # pointers into the part that moves: every one is in the table, under the section it points into
        for slot, sec in self.slots.items():
            if sec:
                v = self._u64(slot)
                if v:
                    if v < self.tail:
                        raise ValueError(f"pointer at {slot:#x} is listed under section {sec} but points to {v:#x}")
                    self.poke(slot, struct.pack("<Q", v + shift))
        body = bytes(d[: self.base]) + bytes(self.new)
        body += bytes(self.tail + shift - len(body)) + bytes(d[self.tail : self.rlt])
        out = bytearray(body)
        # the relocation table: old entries as they were, then one entry per run of new pointers
        runs = []
        for slot in sorted(self.added):
            if runs and runs[-1][0] + 8 * runs[-1][1] == slot and runs[-1][1] < 255:
                runs[-1][1] += 1
            else:
                runs.append([slot, 1])
        sections = [list(s) for s in self.sections]
        first0, count0 = sections[0][3], sections[0][4]
        new_entries = [(pos, 1, n, 0) for pos, n in runs]
        entries = self.entries[first0 : first0 + count0] + new_entries + self.entries[first0 + count0 :]
        if self.new:
            sections[0][2] = self.base + len(self.new) - sections[0][1]
        sections[0][4] += len(runs)
        for s in sections[1:]:
            s[1] += shift
            s[3] += len(runs)
        rlt = len(out)
        out += struct.pack("<4sIII", b"_RLT", rlt, len(sections), 0)
        for s in sections:
            out += struct.pack("<QIIii", *s)
        for e in entries:
            out += struct.pack("<IHBB", *e)
        struct.pack_into("<II", out, 0x18, rlt, len(out))  # the header: where the table is, and the file's size
        return bytes(out)


def check(data: bytes | bytearray) -> int:
    """
    Test a file's relocation table.

    Every pointer it lists must be null or land inside the section it is listed under, and no two
    entries may list the same slot.

    Returns:
        The number of pointers listed.

    Raises:
        ValueError: Saying what is wrong with the table, or with the first pointer that fails.
    """
    f = Bfres(data)
    rlt = f.u32(0x18)
    nsec = f.u32(rlt + 8)
    if data[rlt : rlt + 4] != b"_RLT" or f.u32(rlt + 4) != rlt or f.u32(0x1C) != len(data):
        raise ValueError("relocation table header is wrong")
    sections = [struct.unpack_from("<QIIii", data, rlt + 0x10 + i * 0x18) for i in range(nsec)]
    base = rlt + 0x10 + nsec * 0x18
    if base + 8 * sum(s[4] for s in sections) != len(data):
        raise ValueError("relocation entries do not end the file")
    seen = set()
    for _, position, size, first, count in sections:
        for i in range(first, first + count):
            pos, structs, pointers, skip = struct.unpack_from("<IHBB", data, base + i * 8)
            for _ in range(structs):
                for _ in range(pointers):
                    if pos in seen:
                        raise ValueError(f"pointer at {pos:#x} is listed twice")
                    seen.add(pos)
                    v = f.u64(pos)
                    if v and not position <= v < position + max(size, 1) and v != position + size:
                        end = position + size
                        raise ValueError(f"pointer at {pos:#x} -> {v:#x} leaves its section ({position:#x}..{end:#x})")
                    pos += 8
                pos += 8 * skip
    return len(seen)


def followed(f: Bfres) -> list[int]:
    """
    Where every pointer is that reading the skeletons and clips follows, null or not.

    These are the structures this module rebuilds. `unlisted` compares them with the relocation
    table; on the game's own file it finds nothing to report, which is what says this walk knows
    where the pointers are.

    Returns:
        The offsets of the pointers.
    """
    out = []
    for m in f.models():
        sk = m.skel.off
        out += [sk + 0x10, sk + 0x18, sk + 0x20, sk + 0x28]
        out += [b["off"] for b in m.skel.bones]
        dic = f.u64(sk + 0x10)
        out += [dic + 8 + i * 16 + 8 for i in range(f.u32(dic + 4) + 1)]
        out += [sh.off + 0x28 for sh in m.shapes]
    for a in f.skel_anims():
        out += [a.o + 0x10, a.o + 0x28, a.o + 0x30]
        arr = f.u64(a.o + 0x30)
        for b in range(f.u16(a.o + 0x58)):
            bo = arr + b * 0x28
            out += [bo, bo + 8, bo + 0x10]
            for c in range(f.u8(bo + 0x1E)):
                co = f.u64(bo + 8) + c * 0x30
                out += [co, co + 8]
    return out


def unlisted(data: bytes | bytearray) -> list[int]:
    """
    Pointers that are followed and not null, yet missing from the relocation table.

    The game turns a pointer into an address only if the table lists it, so any of these is a crash.

    Returns:
        Where each such pointer is.
    """
    g = Grower(data)
    return [slot for slot in followed(g.f) if g.f.u64(slot) and slot not in g.slots]
