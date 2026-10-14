"""
The Switch sound archive (.bfsar) and the files it indexes, as Pikmin 3 Deluxe uses them.

These are NintendoWare's sound formats, little-endian on the Switch. Each file opens the same way:
four letters, a byte-order mark, the file's size, and a list of its blocks as (id, offset, size).

* The archive is the index. Its INFO block holds eight tables: sounds, sound groups, banks, wave
  archives, groups, players, files, and one record of limits. Things in it are named by an item id
  whose top byte is a type (1 sound, 2 sound group, 3 bank, 4 player, 5 wave archive, 6 group) and
  whose other three bytes are a number in that type's table.
* A group file (.bfgrp) holds many of the archive's files in one piece. The game reads it into
  memory whole, and most sounds are loaded from one: GROUP_STATIC.bfgrp.
* A wave sound, the only kind written here, is followed to its samples in four steps. The sound's
  entry in the archive names a file and an index. That file is a wave sound file (.bfwsd), whose
  sound number `index` names a wave archive, by item id, and a wave number. The wave archive's
  entry in the archive names the file it is kept in (.bfwar). Wave number N of that file is a wave
  file (.bfwav): a short header and the samples.

The game does not play a sound by its id (0x01000000 | its number) directly. It plays commands from
the text tables of command/command_table.szs, and a command names the sound id.

What is here:

* `Archive` reads the tables, and `Archive.grow` gives a new archive with entries added and every
  old one in place. `user_field` and `with_fields` read and set what Pikmin 3 keeps in a sound's two
  spare numbers (`USER_FIELDS`): how many variants to pick among, their weights, and which of
  several runs of sounds is played according to who is making the sound (`TYPE_SLOTS`).
* `labels` reads the names the archive gives its sounds, which is how one of the game's own is found.
* `Group` reads a group file, and `Group.append` gives a new one with files added at the end.
* `make_wave`, `make_wave_archive` and `make_wave_sounds` write the three files a new wave sound
  needs; `read_wave`, `read_wave_archive` and `read_wave_sounds` read them, the game's own included.
  A 16-bit wave or a plain wave sound file of the game's, read and written again, comes out byte
  for byte.

Sample data must sit on a 64-byte boundary once loaded, and every layer keeps to that: a file in a
group starts on a multiple of 64 bytes of the group file, a wave in a wave archive on a multiple of
64 bytes of the archive, and a wave's samples a multiple of 64 bytes into the wave (0xC0 in the
ones written here).
"""

from __future__ import annotations

import struct

# The INFO block's eight tables, by the id each is listed under.
SOUNDS = 0x2100
BANKS = 0x2101
PLAYERS = 0x2102
WAVE_ARCHIVES = 0x2103
SOUND_GROUPS = 0x2104
GROUPS = 0x2105
FILES = 0x2106
LIMITS = 0x220B  # one fixed record, not a list of entries like the other seven

# What a sound is: the `kind` of `Archive.sound`.
STREAM_SOUND = 0x2201
WAVE_SOUND = 0x2202
SEQUENCE_SOUND = 0x2203

# A sound carries two 32-bit numbers the sound library leaves to the game (NintendoWare calls them
# user parameters). Pikmin 3 packs these fields into them, as (parameter, shift, mask); they are
# the table at 0x61497748 of its program.
USER_FIELDS = {
    "volume_spread": (0, 29, 7),
    "pitch_spread": (0, 26, 7),
    "variants": (0, 18, 7),  # how many sounds, counting this one, to pick among at random
    "weight": (0, 14, 15),  # this one's share in that pick (0 counts as 16)
    "spaced": (1, 31, 1),  # refuse to play again within `gap` to `gap + gap_spread` frames
    "gap": (1, 28, 7),
    "gap_spread": (1, 24, 15),
    "kind": (1, 14, 15),  # which list of Pikmin and leader types picks the variant: see TYPE_SLOTS
}

# A sound whose `kind` is not 0 is the first of a family: one run of sounds for each slot, the runs
# one after another and each as long as that first sound's `variants` (at least 1). Asked for that
# sound, the game plays sound + slot * variants, where the slot comes from who is making the sound
# (FUN_608366e0). The first sound of the run it lands on then says, with its own `variants`, how
# many of the run to pick among, each by its `weight`.
# Kind 2: every Pikmin but Rock and Winged shares slot 0, and the leaders A to E (Alph, Brittany,
# Charlie, Louie, Olimar) have one each. Three more kinds give other Pikmin a run of their own.
# ("pikmin" is every kind of Pikmin a list does not name.) A family need not have a run for every
# slot of its kind: one made for the Pikmin alone has two or three, and a leader's place in it
# falls past its end, on whatever sounds follow it in the archive.
TYPE_SLOTS = {
    2: {"pikmin": 0, "rock": 1, "winged": 2, "A": 3, "B": 4, "C": 5, "D": 6, "E": 7},
    8: {
        "red": 0, "blue": 1, "yellow": 2, "white": 3, "purple": 4, "rock": 5, "winged": 6,
        "A": 7, "B": 8, "C": 9, "D": 10, "E": 11,
    },
    9: {"pikmin": 0, "yellow": 1, "rock": 2, "winged": 3, "A": 4, "B": 5, "C": 6, "D": 7, "E": 8},
    10: {"pikmin": 0, "red": 1, "rock": 2, "winged": 3, "A": 4, "B": 5, "C": 6, "D": 7, "E": 8},
}  # fmt: skip


def u16(d: bytes | bytearray, o: int) -> int:
    """The unsigned 16-bit number at offset `o`."""
    return struct.unpack_from("<H", d, o)[0]


def u32(d: bytes | bytearray, o: int) -> int:
    """The unsigned 32-bit number at offset `o`."""
    return struct.unpack_from("<I", d, o)[0]


def s32(d: bytes | bytearray, o: int) -> int:
    """The signed 32-bit number at offset `o`."""
    return struct.unpack_from("<i", d, o)[0]


def user_field(params: tuple[int, int], name: str) -> int:
    """Read a field of a sound's (parameter 0, parameter 1)."""
    which, shift, mask = USER_FIELDS[name]
    return params[which] >> shift & mask


def with_fields(params: tuple[int, int], **fields: int) -> tuple[int, int]:
    """(parameter 0, parameter 1) with some fields set."""
    params = list(params)
    for name, value in fields.items():
        which, shift, mask = USER_FIELDS[name]
        if not 0 <= value <= mask:
            raise ValueError(f"{name} = {value} does not fit (0 to {mask})")
        params[which] = params[which] & ~(mask << shift) | value << shift
    return tuple(params)


def _blocks(d: bytes | bytearray, base: int = 0) -> dict[int, tuple[int, int]]:
    """{block id: (offset, size)} from the header of a NintendoWare sound file at `base`."""
    if u16(d, base + 4) != 0xFEFF:
        raise ValueError(f"not a little-endian NintendoWare file: {bytes(d[base : base + 8])!r}")
    out = {}
    for i in range(u16(d, base + 0x10)):
        e = base + 0x14 + i * 12
        out[u16(d, e)] = (base + s32(d, e + 4), u32(d, e + 8))
    return out


def _pad(b: bytes, to: int) -> bytes:
    """`b` with zeros added until its length is a multiple of `to`."""
    return b + bytes(-len(b) % to)


def labels(d: bytes | bytearray) -> dict[str, int]:
    """
    The names an archive gives its things, from its STRG block: a table of strings and a search tree
    whose leaves pair a string with an item id.

    Returns:
        The item id of each name.
    """
    body = _blocks(d)[0x2000][0] + 8
    table = body + s32(d, body + 4)
    tree = body + s32(d, body + 12)
    strings = []
    for i in range(u32(d, table)):
        at = table + s32(d, table + 4 + i * 12 + 4)
        strings.append(bytes(d[at : d.index(b"\0", at)]).decode("ascii", "replace"))
    names = {}
    for i in range(u32(d, tree + 4)):
        node = tree + 8 + i * 20
        if u16(d, node) & 1:  # a leaf
            names[strings[u32(d, node + 12)]] = u32(d, node + 16)
    return names


# -- the archive's tables --------------------------------------------------------------------------
#
# The archive's blocks are 0x2000 STRG (the names of things), 0x2001 INFO (the tables) and 0x2002
# FILE (the few files kept inside the archive itself).
#
# A reference is eight bytes: a 16-bit type, two bytes of nothing, and a signed 32-bit offset. The
# INFO block opens with eight references, each to a table: a count, that many references (a type
# and an offset from the table's start), then the entries themselves. Nothing inside an entry points
# outside it, so a table grows by adding references and entries and moving what follows.


class Archive:
    """
    A sound archive: its sounds, files and wave archives, and a way to add more of each.

    `d` is the archive's bytes and `blocks` its blocks as {id: (offset, size)}. `order` is the eight
    table ids in the order the INFO block lists them, and `tables` is {table id: where it starts}.
    """

    def __init__(self, data: bytes | bytearray) -> None:
        d = self.d = bytes(data)
        if d[:4] != b"FSAR":
            raise ValueError(f"not a BFSAR file: {d[:8]!r}")
        self.blocks = _blocks(d)
        if sorted(self.blocks) != [0x2000, 0x2001, 0x2002] or u32(d, 12) != len(d):
            raise ValueError(f"the archive is not laid out as expected (blocks {sorted(self.blocks)})")
        at, size = self.blocks[0x2001]
        if d[at : at + 4] != b"INFO" or u32(d, at + 4) != size or at + size != self.blocks[0x2002][0]:
            raise ValueError("the INFO block is not where the header says")
        self.info = at + 8
        self.order = [u16(d, self.info + 8 * i) for i in range(8)]
        self.tables = {t: self.info + s32(d, self.info + 8 * i + 4) for i, t in enumerate(self.order)}
        starts = [*sorted(self.tables.values()), at + size]
        expected = sorted((SOUNDS, SOUND_GROUPS, BANKS, WAVE_ARCHIVES, GROUPS, PLAYERS, FILES, LIMITS))
        if sorted(self.order) != expected or starts[0] != self.info + 0x40:
            raise ValueError("the INFO block does not hold the eight tables expected")
        self.ends = {t: starts[starts.index(a) + 1] for t, a in self.tables.items()}

    def count(self, table: int) -> int:
        """How many entries a table has."""
        return u32(self.d, self.tables[table])

    def entry(self, table: int, i: int) -> tuple[int, int]:
        """(offset, size) of entry i of a table."""
        t = self.tables[table]
        n = u32(self.d, t)
        if not 0 <= i < n:
            raise IndexError(f"table {table:#x} has {n} entries, not {i + 1}")
        at = t + s32(self.d, t + 8 + 8 * i)
        # An entry runs to the next one, so padding after a table's last entry counts as part of it.
        end = t + s32(self.d, t + 16 + 8 * i) if i + 1 < n else self.ends[table]
        return at, end - at

    def sound(self, i: int) -> dict:
        """
        Read a sound's entry: which file and player, how loud, and the options that are set.

        An entry has a word of flags and one 32-bit value for each bit set in it, lowest bit first.
        Bit 0 is the sound's name, bit 2 holds its priority, and bits 31 and 30 are the two numbers
        left to the game, parameter 0 and parameter 1.

        Returns:
            A dict: `file` (a number in the files table), `player` (an item id), `volume`, `kind`
            (`WAVE_SOUND`, `SEQUENCE_SOUND` or `STREAM_SOUND`), `flags`, `string` (the number of its
            name, or None), `priority`, `user` (the two parameters `USER_FIELDS` describes) and
            `size` (of the entry, in bytes). A wave sound also has `index`: its number in its file.
        """
        d = self.d
        o, size = self.entry(SOUNDS, i)
        flags = u32(d, o + 0x14)
        values, p = {}, o + 0x18
        for bit in range(32):
            if flags >> bit & 1:
                values[bit] = u32(d, p)
                p += 4
        kind = u16(d, o + 0xC)
        detail = o + s32(d, o + 0x10)
        out = {
            "file": u32(d, o),
            "player": u32(d, o + 4),
            "volume": d[o + 8],
            "kind": kind,
            "flags": flags,
            "string": values.get(0),
            "priority": values.get(2, 0) & 0xFF,
            "user": (values.get(31, 0), values.get(30, 0)),
            "size": size,
        }
        if kind == WAVE_SOUND:
            out["index"] = u32(d, detail)
        return out

    def file(self, i: int) -> dict:
        """
        Read a file's entry.

        Returns:
            {`path`} for a file of its own beside the archive (the streamed music). Otherwise
            {`offset`, `size`, `groups`}: where it is in the archive's own FILE block (-1 and
            0xFFFFFFFF if it is not there), and the item id of each group whose file holds it.
        """
        d = self.d
        o, _ = self.entry(FILES, i)
        body = o + s32(d, o + 4)
        if u16(d, o) == 0x220D:
            return {"path": d[body : d.index(b"\0", body)].decode("ascii")}
        groups = []
        if s32(d, body + 16) not in (0, -1):
            g = body + s32(d, body + 16)
            groups = [u32(d, g + 4 + 4 * k) for k in range(u32(d, g))]
        return {"offset": s32(d, body + 4), "size": u32(d, body + 8), "groups": groups}

    def wave_archive(self, i: int) -> int:
        """The file a wave archive is kept in."""
        return u32(self.d, self.entry(WAVE_ARCHIVES, i)[0])

    def raw(self, table: int, i: int) -> bytes:
        """The bytes of entry i of a table."""
        o, size = self.entry(table, i)
        return self.d[o : o + size]

    def grow(self, additions: dict[int, list[bytes]]) -> bytes:
        """A new archive with entries added: {table: [entry bytes, ...]}. Old entries keep their numbers."""
        d = self.d
        at, size = self.blocks[0x2001]
        regions, offsets = [], {}
        position = 0x40
        for t in sorted(self.tables, key=self.tables.get):
            region = d[self.tables[t] : self.ends[t]]
            new = additions.get(t) or []
            if new:
                if t == LIMITS:
                    raise ValueError(f"table {t:#x} is not a list")
                n = u32(region, 0)
                kind = u16(region, 4)
                if any(len(b) % 4 for b in new):
                    raise ValueError("entries must be a multiple of four bytes long")
                region = _pad(region, 4)
                # Every old reference moves past the new ones; the new entries go after the old.
                refs = bytearray(region[4 : 4 + 8 * n])
                for i in range(n):
                    struct.pack_into("<i", refs, 8 * i + 4, s32(refs, 8 * i + 4) + 8 * len(new))
                where = len(region) + 8 * len(new)
                for b in new:
                    refs += struct.pack("<HHi", kind, 0, where)
                    where += len(b)
                region = struct.pack("<I", n + len(new)) + bytes(refs) + region[4 + 8 * n :] + b"".join(new)
            offsets[t] = position
            regions.append(region)
            position += len(region)
        head = b"".join(struct.pack("<HHi", t, 0, offsets[t]) for t in self.order)
        body = _pad(head + b"".join(regions) + bytes(8), 0x40)[:-8]  # the block is a multiple of 0x40
        block = b"INFO" + struct.pack("<I", len(body) + 8) + body
        moved = len(block) - size
        out = bytearray(d[:at] + block + d[at + size :])
        struct.pack_into("<I", out, 12, len(out))
        for i in range(u16(d, 0x10)):
            e = 0x14 + i * 12
            if u16(d, e) == 0x2001:
                struct.pack_into("<I", out, e + 8, len(block))
            elif u16(d, e) == 0x2002:
                struct.pack_into("<i", out, e + 4, s32(d, e + 4) + moved)
        return bytes(out)


def wave_sound_entry(template: bytes | bytearray, **fields: int | tuple[int, int]) -> bytes:
    """
    Make a wave sound's entry for the archive from another's, with some fields changed.

    The fields are `file`, `player`, `volume`, `string`, `priority`, `index` and `user` (the two
    parameters). They are found by position, so `template` must set exactly the usual options.

    Returns:
        The new entry, as `Archive.grow` takes it.

    Raises:
        ValueError: If `template` is not an ordinary wave sound's entry.
    """
    b = bytearray(template)
    if (
        len(b) != 0x58
        or u16(b, 0xC) != WAVE_SOUND
        or s32(b, 0x10) != 0x48
        or u32(b, 0x14) != 0xC0020107
        or u32(b, 0x50) != 1
    ):
        raise ValueError("the sound taken as a pattern is not an ordinary wave sound")
    where = {"file": 0x00, "player": 0x04, "string": 0x18, "index": 0x48}
    for name, value in fields.items():
        if name == "volume":
            b[8] = value
        elif name == "priority":
            b[0x20] = value
        elif name == "user":  # stored in the order of their flag bits: parameter 1, then parameter 0
            struct.pack_into("<II", b, 0x2C, value[1], value[0])
        else:
            struct.pack_into("<I", b, where[name], value)
    return bytes(b)


# -- group files -----------------------------------------------------------------------------------


class Group:
    """
    A group file (.bfgrp): the files a group of sounds needs, loaded in one piece.

    The game reads the whole file into memory and hands the sound library each file in it by number.
    Every file starts on a multiple of 64 bytes, which the audio hardware needs of sample data.

    Its blocks are 0x7800 INFO (the list of files), 0x7801 FILE (the files) and 0x7802 INFX, which
    is carried over as it is. `d` is the file's bytes and `blocks` its blocks as {id: (offset,
    size)}. `items` is [(file number, offset, size)] in the order the files are stored, and `body`
    is the position in `d` those offsets count from.
    """

    def __init__(self, data: bytes | bytearray) -> None:
        d = self.d = data
        if bytes(d[:4]) != b"FGRP":
            raise ValueError(f"not a BFGRP file: {bytes(d[:8])!r}")
        self.blocks = b = _blocks(d)
        if sorted(b) != [0x7800, 0x7801, 0x7802] or u32(d, 12) != len(d):
            raise ValueError("the group file is not laid out as expected")
        info, files, extra = b[0x7800], b[0x7801], b[0x7802]
        if (
            info[0] != 0x40
            or files[0] != info[0] + info[1]
            or extra[0] != files[0] + files[1]
            or extra[0] + extra[1] != len(d)
            or files[0] % 64
        ):
            raise ValueError("the group file's blocks are not where expected")
        table = info[0] + 8
        n = u32(d, table)
        self.items = []
        for i in range(n):
            o = table + 4 + 8 * n + 16 * i
            if u16(d, table + 4 + 8 * i) != 0x7900 or table + s32(d, table + 8 + 8 * i) != o or u16(d, o + 4) != 0x1F00:
                raise ValueError(f"entry {i} of the group file is not laid out as expected")
            self.items.append((u32(d, o), s32(d, o + 8), u32(d, o + 12)))
        self.body = files[0] + 8
        end = 0
        for fid, off, size in self.items:
            if off < end or (self.body + off) % 64:
                raise ValueError(f"file {fid} of the group is out of order or not on a 64-byte boundary")
            end = off + size
        if end > files[1] - 8:
            raise ValueError("the group's last file runs past its FILE block")

    def file(self, number: int) -> bytes | bytearray:
        """The bytes of a file in the group, by the archive's file number."""
        for fid, off, size in self.items:
            if fid == number:
                return self.d[self.body + off : self.body + off + size]
        raise KeyError(f"file {number} is not in this group")

    def append(self, files: list[tuple[int, bytes | bytearray]]) -> bytes:
        """A new group file with files added at the end: [(file number, bytes), ...]."""
        d = self.d
        old, extra = self.blocks[0x7801], self.blocks[0x7802]
        if any(fid == have for fid, _ in files for have, _, _ in self.items):
            raise ValueError("a file being added is already in the group")
        items = list(self.items)
        # The FILE block keeps its bytes; new files go after them, each on a 64-byte boundary.
        body = bytes(d[self.body : old[0] + old[1]])
        added = []
        for fid, data in files:
            pad = -(len(body) + 8) % 64
            added.append(bytes(pad) + bytes(data))
            items.append((fid, len(body) + pad, len(data)))
            body += added[-1]
        body = _pad(bytes(8) + body, 0x20)[8:]
        n = len(items)
        table = struct.pack("<I", n)
        table += b"".join(struct.pack("<HHi", 0x7900, 0, 4 + 8 * n + 16 * i) for i in range(n))
        table += b"".join(struct.pack("<IHHiI", fid, 0x1F00, 0, off, size) for fid, off, size in items)
        table = _pad(bytes(8) + table, 0x40)[8:]  # keeps the FILE block on a 64-byte boundary
        head = bytearray(d[:0x40])
        blocks = [
            (0x7800, b"INFO" + struct.pack("<I", len(table) + 8) + table),
            (0x7801, b"FILE" + struct.pack("<I", len(body) + 8) + body),
            (0x7802, bytes(d[extra[0] : extra[0] + extra[1]])),
        ]
        at = 0x40
        for i, (kind, block) in enumerate(blocks):
            struct.pack_into("<HHiI", head, 0x14 + 12 * i, kind, 0, at, len(block))
            at += len(block)
        struct.pack_into("<I", head, 12, at)
        return bytes(head) + b"".join(block for _, block in blocks)


# -- waves, wave archives, wave sounds -------------------------------------------------------------


def make_wave(samples: list[int], rate: int) -> bytes:
    """A wave file (.bfwav): one channel of signed 16-bit samples, played once."""
    data = struct.pack(f"<{len(samples)}h", *samples)
    info = struct.pack("<BBHIIII", 1, 0, 0, rate, 0, len(samples), 0)  # PCM16, no loop
    info += struct.pack("<I", 1) + struct.pack("<HHi", 0x7100, 0, 0xC)  # one channel
    info += struct.pack("<HHi", 0x1F00, 0, 0x38) + struct.pack("<HHi", 0, 0, -1) + bytes(4)
    info = b"INFO" + struct.pack("<I", 0x40) + _pad(info, 0x38)
    body = b"DATA" + struct.pack("<I", 0x40 + len(data)) + bytes(0x38) + data  # samples 0xC0 into the file
    head = b"FWAV" + struct.pack("<HHII", 0xFEFF, 0x40, 0x00010200, 0x80 + len(body))
    head += struct.pack("<HH", 2, 0) + struct.pack("<HHiI", 0x7000, 0, 0x40, 0x40)
    head += struct.pack("<HHiI", 0x7001, 0, 0x80, len(body))
    return _pad(head, 0x40) + info + body


def read_wave(d: bytes | bytearray, base: int = 0) -> dict:
    """
    Read the first channel of the wave file (.bfwav) at `base`, 16-bit PCM or DSP ADPCM.

    Returns:
        A dict: `rate` (samples a second), `loops`, `encoding` (1 PCM16, 2 DSP ADPCM), `samples`
        (signed 16-bit numbers), `channels` (how many the file has) and `size` (of the file).

    Raises:
        ValueError: If this is not a wave file, or it is encoded some other way.
    """
    if bytes(d[base : base + 4]) != b"FWAV":
        raise ValueError(f"not a BFWAV file: {bytes(d[base : base + 8])!r}")
    b = _blocks(d, base)
    info, data = b[0x7000][0] + 8, b[0x7001][0] + 8
    encoding, loops, rate, count = d[info], d[info + 1], u32(d, info + 4), u32(d, info + 12)
    table = info + 0x14
    channel = table + s32(d, table + 8)
    at = data + s32(d, channel + 4)
    if encoding == 1:
        samples = list(struct.unpack_from(f"<{count}h", d, at))
    elif encoding == 2:
        # Eight-byte frames of fourteen samples: a byte choosing a pair of the channel's sixteen
        # coefficients and a scale, then a nibble for each sample, which is that nibble scaled plus
        # a prediction from the two samples before it. The two before the first are in the header.
        adpcm = channel + s32(d, channel + 12)
        coefs = struct.unpack_from("<16h", d, adpcm)
        h1, h2 = struct.unpack_from("<hh", d, adpcm + 34)
        samples = []
        for frame in range((count + 13) // 14):
            p = at + 8 * frame
            c1, c2 = coefs[2 * (d[p] >> 4)], coefs[2 * (d[p] >> 4) + 1]
            scale = 1 << (d[p] & 0xF)
            for k in range(14):
                nib = d[p + 1 + k // 2] >> (0 if k & 1 else 4) & 0xF
                v = ((nib - 16 if nib >= 8 else nib) * scale * 2048 + 1024 + c1 * h1 + c2 * h2) >> 11
                h2, h1 = h1, max(-32768, min(32767, v))
                samples.append(h1)
        del samples[count:]
    else:
        raise ValueError(f"wave encoding {encoding} is not handled")
    return {
        "rate": rate,
        "loops": bool(loops),
        "encoding": encoding,
        "samples": samples,
        "channels": u32(d, table),
        "size": u32(d, base + 12),
    }


def make_wave_archive(waves: list[bytes]) -> bytes:
    """A wave archive (.bfwar) holding these wave files, each on a 64-byte boundary."""
    body, table = bytearray(), struct.pack("<I", len(waves))
    for w in waves:
        body += bytes(-(len(body) + 8) % 64)
        table += struct.pack("<HHiI", 0x1F00, 0, len(body), len(w))
        body += w
    info = _pad(bytes(8) + table, 0x40)  # so the FILE block starts on a boundary too
    info = b"INFO" + struct.pack("<I", len(info)) + info[8:]
    files = b"FILE" + struct.pack("<I", len(body) + 8) + bytes(body)
    head = b"FWAR" + struct.pack("<HHII", 0xFEFF, 0x40, 0x00010000, 0x40 + len(info) + len(files))
    head += struct.pack("<HH", 2, 0) + struct.pack("<HHiI", 0x6800, 0, 0x40, len(info))
    head += struct.pack("<HHiI", 0x6801, 0, 0x40 + len(info), len(files))
    return _pad(head, 0x40) + info + files


def read_wave_archive(d: bytes | bytearray, base: int = 0) -> list[tuple[int, int]]:
    """[(offset, size)] of the wave files in the wave archive at `base`; offsets are from the start of `d`."""
    if bytes(d[base : base + 4]) != b"FWAR":
        raise ValueError(f"not a BFWAR file: {bytes(d[base : base + 8])!r}")
    b = _blocks(d, base)
    info, files = b[0x6800][0] + 8, b[0x6801][0] + 8
    out = []
    for i in range(u32(d, info)):
        e = info + 4 + 12 * i
        if u16(d, e) != 0x1F00:
            raise ValueError(f"wave {i} of the archive is not laid out as expected")
        out.append((files + s32(d, e + 4), u32(d, e + 8)))
    return out


def _wave_sound(index: int) -> bytes:
    """One wave sound of a wave sound file, as every one in the game's small sound files is written."""
    # Centred, at its own pitch, full send, an envelope that is all the way open, and one track with
    # one note that plays wave `index` of the file's own list of waves.
    info = struct.pack("<I", 0x307) + struct.pack("<IfIII", 0x40, 1.0, 0x40, 0x18, 0x20)
    info += struct.pack("<BB6x", 0x7F, 3) + struct.pack("<HHi", 0, 0, 8) + bytes([0x7F] * 5) + bytes(3)
    track = struct.pack("<I", 1) + struct.pack("<HHi", 0x4903, 0, 0xC) + struct.pack("<HHi", 0, 0, 8)
    track += struct.pack("<I", 1) + struct.pack("<HHi", 0x4904, 0, 0xC) + bytes(16)
    note = struct.pack("<I", 1) + struct.pack("<HHi", 0x4902, 0, 0xC) + struct.pack("<II", index, 0)
    head = struct.pack("<HHi", 0x4901, 0, 0x18) + struct.pack("<HHi", 0x0101, 0, 0x18 + len(info))
    head += struct.pack("<HHi", 0x0101, 0, 0x18 + len(info) + len(track))
    return head + info + track + note


def make_wave_sounds(waves: list[tuple[int, int]]) -> bytes:
    """A wave sound file (.bfwsd) with one plain sound for each (wave archive item id, wave number)."""
    n = len(waves)
    ids = struct.pack("<I", n) + b"".join(struct.pack("<II", archive, index) for archive, index in waves)
    sounds = [_wave_sound(i) for i in range(n)]
    table, at = struct.pack("<I", n), 4 + 8 * n
    for s in sounds:
        table += struct.pack("<HHi", 0x4900, 0, at)
        at += len(s)
    body = struct.pack("<HHi", 0x0100, 0, 0x10) + struct.pack("<HHi", 0x0101, 0, 0x10 + len(ids))
    body += ids + table + b"".join(sounds)
    info = b"INFO" + struct.pack("<I", len(body) + 8) + body
    head = b"FWSD" + struct.pack("<HHII", 0xFEFF, 0x20, 0x00010100, 0x20 + len(info))
    head += struct.pack("<HH", 1, 0) + struct.pack("<HHiI", 0x6800, 0, 0x20, len(info))
    return _pad(head, 0x20) + info


def read_wave_sounds(d: bytes | bytearray, base: int = 0) -> list[tuple[int, int, bytes]]:
    """
    Read the sounds of the wave sound file (.bfwsd) at `base`.

    Only the first note of the first track is followed, which is all the game's voice files use.

    Returns:
        (wave archive item id, wave number, entry bytes) for each sound of the file.

    Raises:
        ValueError: If this is not a wave sound file, or a sound names a wave the file does not list.
    """
    if bytes(d[base : base + 4]) != b"FWSD":
        raise ValueError(f"not a BFWSD file: {bytes(d[base : base + 8])!r}")
    info = _blocks(d, base)[0x6800][0] + 8
    ids = info + s32(d, info + 4)
    table = info + s32(d, info + 12)
    n = u32(d, table)
    out = []
    for i in range(n):
        at = table + s32(d, table + 8 + 8 * i)
        end = table + s32(d, table + 16 + 8 * i) if i + 1 < n else base + u32(d, base + 12)
        notes = at + s32(d, at + 20)
        note = notes + s32(d, notes + 8)
        which = u32(d, note)
        if which >= u32(d, ids):
            raise ValueError(f"sound {i} names wave {which} of {u32(d, ids)}")
        out.append((u32(d, ids + 4 + 8 * which), u32(d, ids + 8 + 8 * which), bytes(d[at:end])))
    return out
