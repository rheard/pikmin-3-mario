"""
Read Mario out of a Super Mario 64 ROM: his model, its textures and his animations.

Nothing from the ROM is kept anywhere: your own copy is read each time. Any byte order is accepted
(.z64, .v64, .n64), and nothing is looked up by fixed address: the model is found through the level
script that loads it and the animation table by its header, so other releases should work, though
only the US one has been read (SHA-1 9bef1128717f958171a4afac3ed78ee2bb4e86ce as a .z64).

`Mario(path)` gives:

* `parts`: the 20 animated parts in the order animations address them, each with its parent, its
  offset from the parent and the triangles drawn with it. The model's own space is used throughout:
  Y up, facing +Z, his left at +X, which is also how the Pikmin 3 leaders are laid out.
* `pose(rotations, root)`: every part's (rotation, position) in model space for a set of angles.
  `REST` is a T-pose (standing, arms straight out) built from the same joints, for binding a
  skeleton; the game itself has no rest pose, only animations.
* `animation(n)`: an `Animation` whose `rotations(frame)` and `root(frame)` feed `pose`.
  `sequence(names)` is several of them end to end, as one `Sequence` that comes round.
* `image(material)`: a material's texture as rows of RGBA, the decal already laid over the part's
  colour (the game blends the two when it draws), or the flat colour for an untextured part.

`Mario(path, eyes="closed")` gives him another of his eight sets of eyes (`EYES`): the head is
stored once per eye texture and the game switches between them to blink, glance and die.
`Rom.hud_glyph("mario")` is the 16x16 head from the game's own HUD, the one beside the count of
lives; the coin, the star and the digits are there too.

What the game leaves to code is not in the model: the torso's lean when running (`pose(lean=)`
applies it), the head turning, and the fist or foot swelling for a punch or kick (`swell` says
which part each of the three scale nodes holds, `swell_table()` how much, read out of the game
function that drives them). The first choice is taken at every other switch in the model: cap on,
fists, the nearest level of detail, no wing or metal cap.

The formats (MIO0, geo layout, Fast3D display lists, the animation table) follow the public
documentation of the game.
"""

from __future__ import annotations

import math
import struct

from itertools import starmap
from pathlib import Path

from mario.errors import BuildError
from mario.geometry import Matrix, Vector, euler_matrix, matmul, rotate

Pixel = tuple[int, int, int, int]
Rows = list[list[Pixel]]
Corner = tuple[Vector, Vector, tuple[float, float]]  # of a triangle: position, normal, texture coordinate
Pose = list[tuple[Matrix, Vector]]  # the rotation and position of every part, in the order of `PARTS`

# Animation numbers worth a name here (the game has 209).
ANIMATIONS = {
    "walking": 72, "running": 114, "idle_left": 195, "idle_right": 196, "idle_centre": 197,
    "single_jump": 77, "land_single_jump": 78, "double_jump_rise": 80, "double_jump_fall": 76, "land_double_jump": 75,
    "triple_jump": 193, "triple_jump_land": 192, "general_fall": 86, "general_land": 87,
    "backflip": 4, "ground_kick": 102,
    "first_punch": 103, "second_punch": 104, "first_punch_fast": 105, "second_punch_fast": 106,
    # Left alone he falls asleep, in this order: he droops, scratches, yawns, sits down, sleeps sitting
    # (a loop), lies down and sleeps lying (a loop); and he wakes from either with a start.
    "start_sleep_idle": 129, "start_sleep_scratch": 130, "start_sleep_yawn": 131, "start_sleep_sitting": 132,
    "sleep_idle": 133, "sleep_start_lying": 134, "sleep_lying": 135, "wake_from_sleep": 200, "wake_from_lying": 201,
}  # fmt: skip

# The parts, in the order the model's layout visits them, which is the order animations use.
PARTS = (
    "root", "pelvis", "torso", "head",
    "left_shoulder", "left_upperarm", "left_forearm", "left_hand",
    "right_shoulder", "right_upperarm", "right_forearm", "right_hand",
    "left_hip", "left_thigh", "left_shin", "left_foot",
    "right_hip", "right_thigh", "right_shin", "right_foot",
)  # fmt: skip

# A rest pose, in degrees about each part's own X, Y, Z (applied in that order). Every part is
# modelled lying along its own +X, so with no rotation at all the model is a tangle; these angles
# stand him up with his arms out and his fists and feet level.
REST = {
    "root": (90.0, 0.0, 90.0),
    "left_shoulder": (180.0, -90.0, 0.0),
    "right_shoulder": (180.0, 90.0, 0.0),
    "left_hand": (0.0, 0.0, -23.0),
    "right_hand": (0.0, 0.0, -23.0),
    "left_hip": (0.0, 0.0, 180.0),
    "right_hip": (0.0, 0.0, 180.0),
    "left_foot": (0.0, 0.0, -70.0),
    "right_foot": (0.0, 0.0, -70.0),
}
REST_ROOT = (0.0, 189.0, 0.0)  # where the root sits when he stands: the height his animations' headers name

# His eyes: the head is stored eight times over, once per eye texture, and the game picks one.
EYES = ("open", "half", "closed", "right", "left", "up", "down", "dead")

# The HUD's own pictures (digits, letters, the coin, his head, the star): 16x16 each.
HUD_GLYPHS = {"times": 50, "coin": 51, "mario": 52, "star": 53}


def mio0(d: bytes | bytearray, off: int) -> bytes:
    """Decompress the MIO0 block at `off`."""
    if d[off : off + 4] != b"MIO0":
        raise BuildError(f"no MIO0 block at {off:#x}")
    size, comp, raw = struct.unpack_from(">III", d, off + 4)
    comp += off
    raw += off
    out = bytearray()
    bits = off + 0x10
    mask, cur = 0, 0
    while len(out) < size:
        if not mask:
            cur, mask = d[bits], 0x80
            bits += 1
        if cur & mask:
            out.append(d[raw])
            raw += 1
        else:
            v = d[comp] << 8 | d[comp + 1]
            comp += 2
            start = len(out) - (v & 0xFFF) - 1
            for i in range((v >> 12) + 3):  # a byte at a time: the run may reach into what it is writing
                out.append(out[start + i])
        mask >>= 1
    return bytes(out)


class Rom:
    """
    A Super Mario 64 ROM, with what is needed to find Mario in it.

    `d` is the whole ROM in the console's byte order, whichever order the file is in; `title` and
    `code` are from its header. `segments` holds the blocks that segmented addresses point into, by
    segment number. `mario_geo` is the segmented address of the model's layout, and `anim_table`
    where the table of his animations starts in `d`.
    """

    def __init__(self, path: str | Path) -> None:
        data = bytearray(Path(path).read_bytes())
        if data[:4] == b"\x37\x80\x40\x12":  # byte-swapped (.v64, and many files named .n64)
            data[0::2], data[1::2] = data[1::2], data[0::2]
        elif data[:4] == b"\x40\x12\x37\x80":  # little-endian words
            for i in range(0, len(data), 4):
                data[i : i + 4] = data[i : i + 4][::-1]
        elif data[:4] != b"\x80\x37\x12\x40":
            raise BuildError(f"{path} is not an N64 ROM")
        self.d = bytes(data)
        self.title = self.d[0x20:0x34].decode("ascii", "replace").strip()
        self.code = self.d[0x3B:0x3F].decode("ascii", "replace")
        if not self.title.startswith("SUPER MARIO 64"):
            raise BuildError(f'{path} is "{self.title}", not Super Mario 64')
        self.segments: dict[int, bytes] = {}
        self.hud_table = 0  # where the HUD's list of pictures is in segment 2, once `hud_glyph` has found it
        self._find_mario()

    def _find_mario(self) -> None:
        """Find the level script that loads Mario's model, and with it his two segments."""
        d = self.d
        at = d.find(b"\x22\x08\x00\x01\x17")  # LOAD_MODEL_FROM_GEO(model 1, a segment 0x17 address)
        if at < 0:
            raise BuildError("the script that loads Mario was not found in this ROM")
        self.mario_geo = struct.unpack_from(">I", d, at + 4)[0]
        p = at - 0x80
        while p < at:
            if d[p] in {0x17, 0x18} and d[p + 1] == 0x0C and d[p + 2] == 0:  # load a segment, raw or compressed
                segment = d[p + 3]
                start, end = struct.unpack_from(">II", d, p + 4)
                self.segments[segment] = mio0(d, start) if d[p] == 0x18 else d[start:end]
                p += 12
            else:
                p += 4
        if 0x04 not in self.segments or 0x17 not in self.segments:
            raise BuildError("Mario's segments were not found beside the script that loads him")
        # The animation table: a count, a zero, then (offset, size) pairs, the first just past the table.
        at = d.find(struct.pack(">III", 209, 0, 8 + 209 * 8))
        if at < 0:
            raise BuildError("Mario's animation table was not found in this ROM")
        self.anim_table = at

    def seg(self, addr: int) -> tuple[bytes, int]:
        """(bytes, offset) for a segmented address."""
        return self.segments[addr >> 24], addr & 0xFFFFFF

    def hud_glyph(self, which: str | int) -> Rows:
        """
        One of the HUD's 16x16 pictures as rows of RGBA: a name from `HUD_GLYPHS` or a number.

        They sit at the start of the game's common segment, which no level script loads (the game
        does at start-up), so it is found by what it holds: the first compressed block containing a
        table of pointers to consecutive 16x16 pictures from its own start, the digits 0 to 9.

        Returns:
            Sixteen rows of sixteen (r, g, b, a).

        Raises:
            BuildError: If the ROM has no such table, or the table no such picture.
        """
        if 0x02 not in self.segments:
            self._find_hud()
        segment = self.segments[0x02]
        number = HUD_GLYPHS.get(which, which)
        entry = self.hud_table + 4 * number
        addr = struct.unpack_from(">I", segment, entry)[0] if entry + 4 <= len(segment) else 0
        if addr >> 24 != 0x02 or (addr & 0xFFFFFF) + 0x200 > self.hud_table:
            raise BuildError(f"the HUD has no picture {which!r}")
        return _rgba16(segment, addr & 0xFFFFFF, 16, 16)

    def _find_hud(self) -> None:
        """Find the common segment (number 2) and the table of HUD pictures in it."""
        d = self.d
        digits = b"".join(struct.pack(">I", 0x02000000 + k * 0x200) for k in range(10))
        at = d.find(b"MIO0")
        while at >= 0:
            if at % 16 == 0:
                block = mio0(d, at)
                if digits in block:
                    self.segments[0x02] = block
                    self.hud_table = block.find(digits)
                    return
            at = d.find(b"MIO0", at + 4)
        raise BuildError("the HUD pictures were not found in this ROM")


# -- the model's layout -----------------------------------------------------------------------------

# Bytes each layout command takes; None for the ones whose size depends on their flags.
_GEO_SIZE = {
    0x00: 8, 0x01: 4, 0x02: 8, 0x03: 4, 0x04: 4, 0x05: 4, 0x06: 4, 0x07: 4, 0x08: 12, 0x09: 4, 0x0A: None,
    0x0B: 4, 0x0C: 4, 0x0D: 8, 0x0E: 8, 0x0F: 20, 0x10: None, 0x11: None, 0x12: None, 0x13: 12, 0x14: None,
    0x15: 8, 0x16: 8, 0x17: 4, 0x18: 8, 0x19: 8, 0x1A: 8, 0x1C: 12, 0x1D: None, 0x1E: 8, 0x1F: 16, 0x20: 4,
}  # fmt: skip


class _Node:
    """One command of the model's layout, and the nodes it encloses. Only what its command carries is set."""

    def __init__(self, command: int) -> None:
        self.command = command
        self.children: list[_Node] = []
        self.layer = 0  # 0x13, 0x15: the drawing layer its display list belongs to
        self.display_list = 0  # 0x13, 0x15: a segmented address, or 0 for a joint with nothing drawn
        self.offset = (0, 0, 0)  # 0x13: where the part sits in its parent
        self.scale = 1.0  # 0x1D
        self.near = self.far = 0  # 0x0D: the distances this level of detail is drawn between
        self.param = self.func = 0  # 0x18: the number handed to a function of the game, and its RAM address


def _parse_geo(rom: Rom, addr: int) -> tuple[list[_Node], int]:
    """The nodes of a layout from `addr` to its end or return, and where that is; branches are followed in line."""
    segment, p = rom.seg(addr)
    out = []
    while True:
        command = segment[p]
        if command in {0x01, 0x03, 0x05}:  # end, return, close
            return out, p + 4
        if command == 0x04:  # open: what follows belongs to the last node
            children, p = _parse_geo(rom, (addr & 0xFF000000) | (p + 4))
            out[-1].children += children
            continue
        if command == 0x02:  # branch
            target = struct.unpack_from(">I", segment, p + 4)[0]
            out += _parse_geo(rom, target)[0]
            if not segment[p + 1]:  # a jump, not a call: nothing comes back
                return out, p + 8
            p += 8
            continue
        size = _GEO_SIZE.get(command)
        flags = segment[p + 1]
        node = _Node(command)
        if command == 0x0A:
            size = 12 if flags else 8
        elif command == 0x10:  # translate and rotate (only Mario's cap wings)
            size = {0: 16, 1: 8, 2: 8, 3: 4}[flags >> 4 & 7] + (4 if flags & 0x80 else 0)
        elif command in {0x11, 0x12, 0x14}:
            size = 8 + (4 if flags & 0x80 else 0)
        elif command == 0x1D:
            size = 8 + (4 if flags & 0x80 else 0)
            node.scale = struct.unpack_from(">I", segment, p + 4)[0] / 65536.0
        elif command == 0x13:  # an animated part
            node.layer = flags
            node.offset = struct.unpack_from(">3h", segment, p + 2)
            node.display_list = struct.unpack_from(">I", segment, p + 8)[0]
        elif command == 0x15:  # a display list
            node.layer = flags
            node.display_list = struct.unpack_from(">I", segment, p + 4)[0]
        elif command == 0x0D:  # a level of detail: drawn between two distances
            node.near, node.far = struct.unpack_from(">2h", segment, p + 4)
        elif command == 0x18:  # a node the game's code drives: a parameter and a function
            node.param, node.func = struct.unpack_from(">hI", segment, p + 2)
        elif size is None:
            raise BuildError(f"layout command {command:#04x} at {p:#x} is not handled")
        out.append(node)
        p += size


# -- display lists ----------------------------------------------------------------------------------


class Material:
    """What a triangle is drawn with: a flat colour, and perhaps a texture laid over it."""

    def __init__(
        self,
        colour: tuple[int, int, int],
        texture: int,
        size: tuple[int, int],
        clamp: tuple[bool, bool],
    ) -> None:
        self.colour, self.texture, self.size, self.clamp = colour, texture, size, clamp
        self.key = (colour, texture)
        self.name = f"tex_{texture & 0xFFFFFF:06x}" if texture else f"colour_{bytes(colour).hex()}"


class _Layer:
    """
    What the display lists of one drawing layer have left set.

    The game draws a layer's lists one after another without resetting anything, and the model
    relies on it: a thigh is blue because the list before it left blue set.
    """

    def __init__(self, sizes: dict[int, tuple]) -> None:
        self.vertices: dict[int, Corner] = {}
        self.colour = (255, 255, 255)
        self.image = 0  # the segmented address of the texture last named
        self.textured = False
        self.size = (0, 0)
        self.clamp = (True, True)
        self.sizes = sizes  # shared by every layer: {texture: (size, clamp)} as its triangles were drawn


def _run_dl(rom: Rom, addr: int, state: _Layer, tris: list[tuple]) -> None:
    """Follow a Fast3D display list, adding (material key, [(position, normal, st)] * 3) to `tris`."""
    segment, p = rom.seg(addr)
    while True:
        command = segment[p]
        w1 = struct.unpack_from(">I", segment, p + 4)[0]
        if command == 0xB8:  # end
            return
        if command == 0x06:  # another list
            _run_dl(rom, w1, state, tris)
            if segment[p + 1]:
                return
        elif command == 0x04:  # load vertices
            count, first = (segment[p + 1] >> 4) + 1, segment[p + 1] & 0xF
            source, at = rom.seg(w1)
            for i in range(count):
                x, y, z, _, s, t, nx, ny, nz, _ = struct.unpack_from(">3hH2h3bB", source, at + i * 16)
                state.vertices[first + i] = ((x, y, z), (nx, ny, nz), (s, t))
        elif command == 0xBF:  # one triangle; the indices are stored times ten
            key = (state.colour, state.image if state.textured else 0)
            if key[1]:
                state.sizes[key[1]] = (state.size, state.clamp)
            tris.append((key, [state.vertices[segment[p + 5 + k] // 10] for k in range(3)]))
        elif command == 0x03:  # a light: 0x86 is the lit colour, 0x88 the shadowed one
            if segment[p + 1] == 0x86:
                source, at = rom.seg(w1)
                state.colour = tuple(source[at : at + 3])
        elif command == 0xFD:  # texture image
            state.image = w1
        elif command == 0xBB:  # texturing on or off
            state.textured = bool(segment[p + 3] & 1)
        elif command == 0xF5 and (w1 >> 24 & 7) == 0:  # the tile triangles are drawn from: clamping
            state.clamp = (bool(w1 >> 8 & 2), bool(w1 >> 18 & 2))
        elif command == 0xF2:  # its size, in quarter texels
            state.size = ((w1 >> 12 & 0xFFF) // 4 + 1, (w1 & 0xFFF) // 4 + 1)
        p += 8


def _rgba16(segment: bytes, p: int, width: int, height: int) -> Rows:
    """A picture stored as 16 bits a pixel (five each of red, green and blue, one of alpha), as rows of RGBA."""
    rows = []
    for _ in range(height):
        row = []
        for _ in range(width):
            v = segment[p] << 8 | segment[p + 1]
            p += 2
            r, g, b = v >> 11 & 31, v >> 6 & 31, v >> 1 & 31
            row.append((r << 3 | r >> 2, g << 3 | g >> 2, b << 3 | b >> 2, 255 if v & 1 else 0))
        rows.append(row)
    return rows


# -- animations -------------------------------------------------------------------------------------


class Animation:
    """
    One of Mario's animations, by its number in the game's table.

    `frames` is how many it has (frames 0 to `frames` - 1; the game wraps at that count) and `loops`
    whether it starts again when it ends.
    """

    def __init__(self, rom: Rom, number: int) -> None:
        d, table = rom.d, rom.anim_table
        if not 0 <= number < struct.unpack_from(">I", d, table)[0]:
            raise ValueError(f"no animation {number}")
        offset, _ = struct.unpack_from(">II", d, table + 8 + number * 8)
        header = table + offset
        self.number = number
        self.flags, self.y_divisor, self.start, self.loop_start, self.loop_end = struct.unpack_from(">5h", d, header)
        values, index = struct.unpack_from(">II", d, header + 0xC)
        self._values, self._index, self._d = header + values, header + index, d
        self.frames = self.loop_end
        self.loops = not self.flags & 1

    def _channel(self, n: int, frame: int) -> int:
        """Channel `n` at a whole frame. A channel shorter than the animation holds its last value."""
        count, at = struct.unpack_from(">2H", self._d, self._index + n * 4)
        return struct.unpack_from(">h", self._d, self._values + 2 * (at + min(frame, count - 1)))[0]

    def _between(self, n: int, frame: float, *, angle: bool) -> float:
        """
        Channel `n` at any frame.

        The game only ever shows whole frames; between two, this goes straight from one to the next
        (the short way round for an angle), and past the last frame of a looping animation it comes
        back to the first.

        Returns:
            The value, in the channel's own units.
        """
        last = self.frames - 1
        if self.loops:
            frame %= self.frames
        else:
            frame = min(max(frame, 0.0), float(last))
        lo = math.floor(frame)
        hi = lo + 1 if lo < last else (0 if self.loops else last)
        a, b = self._channel(n, lo), self._channel(n, hi)
        step = b - a
        if angle:
            step = (step + 0x8000) % 0x10000 - 0x8000
        return a + step * (frame - lo)

    def root(self, frame: float) -> Vector:
        """The root's position at a frame."""
        return tuple(float(self._between(k, frame, angle=False)) for k in range(3))

    def rotations(self, frame: float) -> dict[str, Vector]:
        """{part: (rx, ry, rz) in radians} at a frame."""
        out = {}
        for i, name in enumerate(PARTS):
            out[name] = tuple(
                self._between(3 + i * 3 + k, frame, angle=True) * (2 * math.pi / 65536.0) for k in range(3)
            )
        return out


class Sequence:
    """
    Several of his animations played one after another, as one that comes round.

    Each plays once through, a frame a frame, and the first follows the last. From the last frame
    of one to the first frame of the next he goes straight across (the short way round for an
    angle), since his own game shows those two on successive frames. So the pieces have to lead
    into each other, as his three standing animations do: he looks left, then right, then ahead
    again, and each ends where the next begins.

    `frames` is how many it has in all; `root` and `rotations` answer as an `Animation`'s do.
    """

    def __init__(self, pieces: list[Animation]) -> None:
        self.pieces = pieces
        self.starts = [sum(piece.frames for piece in pieces[:i]) for i in range(len(pieces))]
        self.frames = sum(piece.frames for piece in pieces)

    def _at(self, frame: int) -> tuple[Animation, int]:
        """The piece that shows on a whole frame, and which frame of it."""
        frame %= self.frames
        i = max(k for k, start in enumerate(self.starts) if start <= frame)
        return self.pieces[i], frame - self.starts[i]

    def _around(self, frame: float) -> tuple[tuple[Animation, int], tuple[Animation, int], float]:
        """The whole frames either side of any frame, and how far it is from the first to the second."""
        frame %= self.frames
        before = math.floor(frame)
        return self._at(before), self._at(before + 1), frame - before

    def root(self, frame: float) -> Vector:
        """The root's position at a frame."""
        (a, at_a), (b, at_b), along = self._around(frame)
        here, there = a.root(float(at_a)), b.root(float(at_b))
        return tuple(x + (y - x) * along for x, y in zip(here, there, strict=True))

    def rotations(self, frame: float) -> dict[str, Vector]:
        """{part: (rx, ry, rz) in radians} at a frame."""
        (a, at_a), (b, at_b), along = self._around(frame)
        here, there = a.rotations(float(at_a)), b.rotations(float(at_b))
        if not along:
            return here

        def short(x: float, y: float) -> float:
            return x + ((y - x + math.pi) % (2 * math.pi) - math.pi) * along

        return {part: tuple(starmap(short, zip(here[part], there[part], strict=True))) for part in here}


# -- Mario --------------------------------------------------------------------------------------------


class Part:
    """One animated part: its name, its parent's number (None for the root) and its offset from that parent."""

    def __init__(self, name: str, parent: int | None, offset: tuple[int, int, int]) -> None:
        self.name, self.parent, self.offset = name, parent, offset
        self.tris: list[tuple[Material, list[Corner]]] = []  # each corner in the part's own space


class Mario:
    """
    Mario's model, as the game holds it.

    `parts` are the animated parts in the order of `PARTS`, `materials` what their triangles are
    drawn with ({(colour, texture): `Material`}), `scale` the model's own scale and `count` the
    number of animations the game has for him. `swell` says which part each of the attack's three
    scale nodes swells ({its number: part name}).
    """

    def __init__(self, path: str | Path | Rom, eyes: str = "open") -> None:
        self.rom = rom = path if isinstance(path, Rom) else Rom(path)
        if eyes not in EYES:
            raise ValueError(f"eyes are one of: {', '.join(EYES)}")
        self.eyes = EYES.index(eyes)
        nodes, _ = _parse_geo(rom, rom.mario_geo)
        self.parts: list[Part] = []
        self.scale = 1.0
        self.swell: dict[int, str] = {}
        self.swell_func = None  # the game function that drives the scale nodes (a RAM address)
        self._sizes: dict[int, tuple] = {}
        self._layers: dict[int, _Layer] = {}
        self._walk(nodes, None)
        if len(self.parts) != len(PARTS):
            raise BuildError(f"expected {len(PARTS)} animated parts in the model, found {len(self.parts)}")
        self.materials: dict[tuple, Material] = {}
        for part in self.parts:
            for i, (key, corners) in enumerate(part.tris):
                if key not in self.materials:
                    size, clamp = self._sizes.get(key[1], ((0, 0), (True, True)))
                    self.materials[key] = Material(key[0], key[1], size, clamp)
                part.tris[i] = (self.materials[key], corners)
        self.count = struct.unpack_from(">I", rom.d, rom.anim_table)[0]

    def _walk(self, nodes: list[_Node], parent: int | None, *, switch: bool = False) -> None:
        """Gather the parts and their triangles from one level of the layout, inside part number `parent`."""
        if switch:
            # A switch draws one of its children. The one between eight bare display lists is his
            # eyes; at every other the first is taken.
            eyes = len(nodes) == len(EYES) and all(node.command == 0x15 for node in nodes)
            nodes = nodes[self.eyes : self.eyes + 1] if eyes else nodes[:1]
        details = [node for node in nodes if node.command == 0x0D]
        nearest = min(details, key=lambda node: node.near) if details else None
        before = None
        for node in nodes:
            last, before = before, node
            if node.command == 0x0D and node is not nearest:
                continue  # keep only the nearest level of detail
            if node.command == 0x10:
                continue  # the wing cap's wings
            here = parent
            if node.command == 0x1D and not self.parts:
                self.scale *= node.scale
            if node.command == 0x1D and last is not None and last.command == 0x18 and parent is not None:
                # A scale node straight after a code-driven one, inside a part: the game's code sets
                # this scale each frame. Mario has three, around the fists and a foot, which swell
                # when he attacks.
                self.swell[last.param] = self.parts[parent].name
                self.swell_func = last.func
            if node.command == 0x13:
                here = len(self.parts)
                self.parts.append(Part(PARTS[here] if here < len(PARTS) else f"part{here}", parent, node.offset))
            if node.command in {0x13, 0x15} and node.display_list:
                if node.layer not in self._layers:
                    self._layers[node.layer] = _Layer(self._sizes)
                _run_dl(self.rom, node.display_list, self._layers[node.layer], self.parts[here].tris)
            self._walk(node.children, here, switch=node.command == 0x0E)

    def animation(self, which: str | int) -> Animation:
        """One of his animations: a name from `ANIMATIONS` or the game's own number."""
        return Animation(self.rom, ANIMATIONS.get(which, which))

    def sequence(self, pieces: tuple[str | int, ...]) -> Sequence:
        """Several of his animations as one that comes round: see `Sequence`."""
        return Sequence([self.animation(which) for which in pieces])

    def swell_table(self) -> dict[str, tuple[float, ...]]:
        """
        How much a fist or foot swells when he attacks: {part name: six scales}.

        The game's code counts a number down by one each frame it draws him and scales the part by
        the entry at that number: a punch starts the count at 4, so the fist goes through entries
        3, 2, 1 and ends at entry 0, which is 1; the kick starts it at 6. The table is the one array
        of bytes the function behind the scale nodes reads, found by following that function's own
        instructions; each entry is tenths.

        Returns:
            The six scales of each of the three parts.

        Raises:
            BuildError: If the scale nodes, or the one table their function reads, are not found.
        """
        d = self.rom.d
        if self.swell_func is None or sorted(self.swell) != [0, 1, 2]:
            raise BuildError("the three scale nodes of Mario's attack were not found in his model")
        entry = struct.unpack_from(">I", d, 8)[0]  # where the game's code is loaded; it starts 0x1000 into the ROM

        def in_rom(address: int) -> int:
            return address - entry + 0x1000

        # Follow the function's instructions for the addresses it reads bytes from. `high` is the
        # registers that hold the high half of an address at the point reached.
        high, tables = {}, []
        for i in range(0, 0x400, 4):
            word = struct.unpack_from(">I", d, in_rom(self.swell_func) + i)[0]
            op, rs, rt, imm = word >> 26, word >> 21 & 31, word >> 16 & 31, word & 0xFFFF
            if word == 0x03E00008:  # jr ra: the function ends
                break
            if op == 0x0F:  # lui: the high half of an address
                high[rt] = imm << 16
            elif op == 0 and word & 0x7FF == 0x21:  # addu: an index added to it
                rd = word >> 11 & 31
                source = high.get(rs, high.get(rt))
                high.pop(rd, None)
                if source is not None:
                    high[rd] = source
            elif op == 0x20 and rs in high:  # lb: a signed byte read from a fixed address
                tables.append(high[rs] + imm - (0x10000 if imm & 0x8000 else 0))
                high.pop(rt, None)
            elif op == 0:  # any other register operation: its result is not an address
                high.pop(word >> 11 & 31, None)
            elif op not in {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x28, 0x29, 0x2B}:
                high.pop(rt, None)  # (branches, jumps and stores write no register)
        if len(tables) != 1:
            found = len(tables)
            raise BuildError(f"expected one table of bytes in the function that swells Mario's fists, found {found}")
        raw = d[in_rom(tables[0]) : in_rom(tables[0]) + 6 * len(self.swell)]
        rows = {self.swell[k]: raw[k * 6 : k * 6 + 6] for k in self.swell}
        if any(row[0] != 10 or not all(10 <= tenths <= 40 for tenths in row) for row in rows.values()):
            raise BuildError(f"the table of swelling does not look like one: {raw.hex()}")
        return {part: tuple(tenths / 10.0 for tenths in row) for part, row in rows.items()}

    def pose(
        self,
        rotations: dict[str, Vector | Matrix] | None = None,
        root: Vector | None = None,
        lean: float = 0.0,
    ) -> Pose:
        """
        [(rotation, position)] for every part in model space, before the model's own scale.

        `rotations` maps part names to angles in radians (parts left out are not turned); with
        nothing given this is the rest pose. A part's turn may be given as a rotation matrix in
        place of its three angles, which is how a pose part of the way between two of his is
        handed over. `lean` (radians) pitches everything above the pelvis
        forward about it, which the game does in code while he walks or runs: up to 30 degrees,
        reached at full speed.

        Returns:
            One (rotation, position) for each part, in the order of `parts`.
        """
        if rotations is None:
            rotations = {name: tuple(math.radians(a) for a in angles) for name, angles in REST.items()}
            root = REST_ROOT if root is None else root
        out = []
        for i, part in enumerate(self.parts):
            turn = rotations.get(part.name, (0.0, 0.0, 0.0))
            turn = turn if isinstance(turn[0], list) else euler_matrix(*turn)
            offset = part.offset if i else tuple(part.offset[k] + (root or (0, 0, 0))[k] for k in range(3))
            if part.parent is None:
                out.append((turn, tuple(float(c) for c in offset)))
            else:
                parent_turn, parent_position = out[part.parent]
                if lean and part.name == "torso":
                    parent_turn = matmul(parent_turn, euler_matrix(0.0, 0.0, lean))
                moved = rotate(parent_turn, offset)
                out.append((matmul(parent_turn, turn), tuple(parent_position[k] + moved[k] for k in range(3))))
        return out

    def image(self, material: Material) -> tuple[int, int, Rows]:
        """(width, height, rows of RGBA) for a material: its texture over its colour, or the colour alone."""
        if not material.texture:
            return 8, 8, [[(*material.colour, 255)] * 8 for _ in range(8)]
        width, height = material.size
        segment, p = self.rom.seg(material.texture)
        rows = _rgba16(segment, p, width, height)
        return width, height, [[(*(px[:3] if px[3] else material.colour), 255) for px in row] for row in rows]

    def triangles(self, pose: Pose | None = None) -> list[tuple[Material, str, list[Corner]]]:
        """
        Every triangle in model space: (material, part name, [(position, normal, (u, v))] * 3).

        Positions carry the model's scale; u and v are in texture widths, v downwards as stored.
        `pose` is what `pose()` gives, and the rest pose if left out.

        Returns:
            The triangles, part by part in the order of `parts`.
        """
        pose = pose or self.pose()
        out = []
        for part, (turn, position) in zip(self.parts, pose, strict=True):
            for material, corners in part.tris:
                width, height = material.size if material.texture else (32, 32)
                triangle = []
                for p, n, (s, t) in corners:
                    q = rotate(turn, p)
                    normal = rotate(turn, n)
                    length = math.sqrt(sum(c * c for c in normal)) or 1.0
                    placed = tuple((q[k] + position[k]) * self.scale for k in range(3))
                    triangle.append((placed, tuple(c / length for c in normal), (s / 32.0 / width, t / 32.0 / height)))
                out.append((material, part.name, triangle))
        return out
