"""
Alph's pictures in the HUD and the menus become Mario's.

What the game has
-----------------
No leader's picture on screen is drawn from his model. They are small ready-made textures, kept in
each layout archive's one texture file (`lyt/<name>.szs`, then `<name>.arc`, then
`timg/__Combined.bntx`) and named by the leader's letter. Layouts pick one by name:

* `nh_PlayerAR0` to `R4`: his head, turned to the right, with five expressions. The round button
  for him at the bottom left of the HUD and the portrait in a message window both pick among these.
* `nh_PlayerAR5`: the same head knocked out, in grey: two channels, brightness and coverage.
* `nh_PlayerAL0`: his head turned to the left: the D-pad menu, and lists in several menus.
* `nh_PlayerStandA`: head and shoulders, fading out below: his place among the Pikmin types when
  another leader has him in the party. `nt_PlayerStandA` is all of him, standing.
* `nh_MmpAlph`: the emblem that marks him on the map.
* `nt_PlayerArun_00` to `19` and `nt_AnimAlfRun_00` to `19`: him running, as a loop.
* `nh_AnimPlayerAWait`, `Run2` and `Jumpb1` (the versus menu): him standing, running and jumping.

What is written
---------------
Every layout archive that has any of those, with each one drawn again as Mario at the same size and
encoded over the old picture in place: same texture, same format, same bytes of file. The other
leaders' pictures are byte for byte the game's, and so is every layout but one.

* The heads, the bust, the standing figure and the loops are his own model from the ROM, shaded the
  way his game shades him. His eyes are one of the eight textures the game switches between, chosen
  per expression in `FACES`.
* The left-facing head and the map emblem are the head from Super Mario 64's own HUD, the one beside
  the count of lives: 16x16 in the ROM, enlarged by whole pixels and not smoothed.
* The one layout: in the versus menu the light of Alph's antenna is drawn over his tall picture, a
  glow that the picture's animations move about with his head. Mario has no antenna, so that light
  is switched off: its panes in Alph's own layout are marked not to be drawn (`put_out_light`).

An archive is not compressed again: the stock stream is kept but for the stretches that cover the
changed textures, so a 70 MB archive takes seconds and comes out a few KB larger.
"""

from __future__ import annotations

import logging
import math
import re
import struct

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from mario.errors import BuildError
from mario.formats import bntx, sarc, yaz0
from mario.sm64 import render
from mario.sm64.rom import REST, REST_ROOT, Mario, Rom

if TYPE_CHECKING:
    from mario.romfs import Mod, RomFs

logger = logging.getLogger(__name__)

LAYOUTS = "lyt"
TEXTURES = "timg/__Combined.bntx"

# Alph's textures, by name without the format suffix, and what to draw for each. A group in the
# pattern is the number of the picture among those of its kind.
PICTURES = tuple(
    (re.compile(pattern + "$"), kind)
    for pattern, kind in (
        (r"nh_PlayerAR(\d)_00", "face"),
        (r"nh_PlayerAL0_00", "glyph"),
        (r"nh_PlayerStandA_00", "bust"),
        (r"nt_PlayerStandA_00", "stand"),
        (r"nh_MmpAlph_00", "emblem"),
        (r"nt_PlayerArun_(\d\d)", "run quarter"),
        (r"nt_AnimAlfRun_(\d\d)", "run side"),
        (r"nh_AnimPlayerAWait_(\d\d)", "big wait"),
        (r"nh_AnimPlayerARun2_(\d\d)", "big run"),
        (r"nh_AnimPlayerAJumpb1_(\d\d)", "big jump"),
    )
)
# Frames in each loop. An archive may hold only one of them (a still), so this cannot be counted there.
LOOPS = {"run side": 20, "run quarter": 20, "big wait": 20, "big run": 26, "big jump": 31}

# His head for each of the leader's face textures: which eyes, and how far the head is turned from
# rest about its own three axes, in degrees (the third nods it: positive is down).
# The leaders' own five are: 0 plain, 1 mouth open in surprise, 2 alert, 3 downcast, 4 a scowl; 5 is
# the grey one shown when he is down. Mario's mouth is under his moustache, so his eyes do the work.
FACES = {
    0: ("open", (0.0, 0.0, 0.0)),
    1: ("up", (0.0, 0.0, -10.0)),
    2: ("right", (0.0, 0.0, 0.0)),
    3: ("down", (0.0, 0.0, 10.0)),
    4: ("half", (0.0, 0.0, 4.0)),
    5: ("dead", (0.0, 8.0, 6.0)),
}
FACE_VIEW = (32.0, 12.0)  # the camera for the heads: degrees he is turned to the right, degrees it looks down
BUST_VIEW = (10.0, 6.0)
BUST_HEAD = 0.56  # the share of the bust picture's height that his head takes
BUST_FADE = 0.30  # the share of its height, at the bottom, over which it fades out
STAND_VIEW = (10.0, 6.0)
STAND = ("idle_left", 0.0)  # the animation and frame he stands in: looking straight ahead
RUN = ("running", 30.0)  # the animation of the loops, and the lean his game gives him at a run
RUN_VIEWS = {"side": (-90.0, 5.0), "quarter": (-55.0, 10.0)}
# The versus menu's tall pictures (194x342): he stands low in them with room to jump. Shares of the
# picture: his height standing, and where his feet are (across, down), as the leaders' own have it.
BIG_VIEW = (25.0, 8.0)
BIG_HEIGHT = 122 / 342.0
BIG_FEET = (100 / 194.0, 326 / 342.0)
BIG_WAIT = ("idle_left", 9.0)  # waiting: his head turns this many frames into the animation and back
BIG_RUN_LEAN = 15.0  # running on the spot, towards the viewer: less lean than at a real run
BIG_RUN_LOOP = (5, 20)  # the layouts loop frames 5 to 24 of the 26
# The jump (31 frames; the leaders' own leave the ground at 2 and land at 20): a crouch, one turn of
# his backflip while a parabola carries him up by this share of his height, then his landing.
BIG_JUMP_AIR = (2, 20)
BIG_JUMP_RISE = 0.85
BIG_JUMP_CROUCH = ("land_single_jump", (30.0, 24.0))
BIG_JUMP_FLIP = ("backflip", 0.0, 13.0)
BIG_JUMP_LAND = ("land_single_jump", 24.0, 37.0)
GLYPH = 16  # the size of the head in his own game's HUD
EMBLEM_ZOOM = 3  # the map emblem: the HUD head at this many pixels a pixel, with a shadow
EMBLEM_SHADOW = ((-3, 3), (72, 8, 8))

# The light of the leader's antenna in the versus menu. His tall picture there is a small layout of
# its own, with the picture twice (itself and its reflection) and over the head of each a glow, with
# the light itself under it as a part. A layout is a run of sections; a pane is a section of at
# least 0x54 bytes with its flags 8 bytes in (bit 0: it is drawn), its opacity 10 in and its name
# 12 in.
LIGHT_LAYOUT = "blyt/PaAnimIconPlayerA_00.bflyt"
LIGHT_PANES = (re.compile(r"P_Player[A-E]glow_\d\d$"), re.compile(r"Pa_PlayerLight_\d\d$"))
PANES = (b"pan1", b"pic1", b"txt1", b"wnd1", b"bnd1", b"prt1")
PANE_SIZE, PANE_FLAGS, PANE_ALPHA, PANE_NAME = 0x54, 8, 10, 12

Pixel = tuple[int, int, int, int]
Rows = list[list[Pixel]]


def to_linear(value: float) -> float:
    """An sRGB value (0-255) as linear light (0-255): how the game stores its grey pictures."""
    value /= 255.0
    return 255.0 * (value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)


def grey(rows: Rows) -> Rows:
    """
    A colour picture as the two channels of a grey texture: brightness and coverage.

    The knocked-out faces are stored this way. Measured against the same leaders' colour faces,
    their brightness is the colour's luma taken out of sRGB, which is what a renderer that expects
    linear light wants of a texture it does not convert.

    Returns:
        Rows of (brightness, coverage, 0, 255).
    """
    return [
        [(int(to_linear(0.299 * r + 0.587 * g + 0.114 * b) + 0.5), a, 0, 255) for r, g, b, a in row] for row in rows
    ]


def blank(rows: Rows) -> Rows:
    """Fully transparent pixels far from the figure made white, as the game's own pictures have them."""
    return [[px if px[3] or px[:3] != (0, 0, 0) else (255, 255, 255, 0) for px in row] for row in rows]


def paste(rows: Rows, picture: Rows, x: int, y: int) -> Rows:
    """Lay `picture` on `rows` at (x, y), copying only its opaque pixels."""
    for j, line in enumerate(picture):
        for i, px in enumerate(line):
            if px[3]:
                rows[y + j][x + i] = px
    return rows


class Artist:
    """Draws every kind of picture, from one ROM, at whatever size a texture is."""

    def __init__(self, rom: Rom, samples: int = 4):
        self.rom = rom
        self.samples = samples  # points averaged per pixel, each way
        self._marios: dict[str, Mario] = {}
        self._pictures: dict[tuple, Rows] = {}
        self._loops: dict[tuple, tuple] = {}

    def mario(self, eyes: str = "open") -> Mario:
        """His model with one of his sets of eyes."""
        if eyes not in self._marios:
            self._marios[eyes] = Mario(self.rom, eyes=eyes)
        return self._marios[eyes]

    def picture(self, kind: str, number: int, width: int, height: int, count: int = 1) -> Rows:
        """
        Draw one texture.

        Args:
            kind: What to draw, from `PICTURES`.
            number: The number in the texture's name.
            width: The texture's width.
            height: The texture's height.
            count: How many pictures there are of its kind (the length of a loop).

        Returns:
            Rows of RGBA.

        Raises:
            BuildError: For a kind nobody knows how to draw.
        """
        key = (kind, number, width, height, count)
        if key not in self._pictures:
            match kind.split():
                case ["face"]:
                    rows = self._face(width, height, number)
                case ["bust"]:
                    rows = self._bust(width, height)
                case ["stand"]:
                    rows = self._stand(width, height)
                case ["run", style]:
                    rows = self._run(width, height, number, count, style)
                case ["big", what]:
                    rows = self._big(width, height, number, count, what)
                case ["glyph"]:
                    rows = self._glyph(width, height)
                case ["emblem"]:
                    rows = self._emblem(width, height)
                case _:
                    raise BuildError(f"no way to draw a {kind!r}")
            self._pictures[key] = blank(rows)
        return self._pictures[key]

    # -- from his model ------------------------------------------------------------------------

    def _face(self, width: int, height: int, number: int) -> Rows:
        if number not in FACES:
            raise BuildError(f"no face is set for texture R{number}")
        eyes, turned = FACES[number]
        mario = self.mario(eyes)
        rest = {part: tuple(math.radians(a) for a in angles) for part, angles in REST.items()}
        camera = render.view(*FACE_VIEW)
        # Framed on the head at rest, so that it does not shift from one expression to the next.
        scale, centre = render.fit(
            render.Figure(mario, mario.pose(rest, REST_ROOT), ("head",)).bounds(camera),
            width,
            height,
            margin=width * 3 / 64.0,
        )
        rest["head"] = tuple(math.radians(a) for a in turned)
        figure = render.Figure(mario, mario.pose(rest, REST_ROOT), ("head",))
        return render.draw(figure, camera, width, height, scale, centre, self.samples)

    def _standing(self) -> tuple[Mario, list]:
        mario = self.mario()
        name, frame = STAND
        anim = mario.animation(name)
        return mario, mario.pose(anim.rotations(frame), anim.root(frame))

    def _bust(self, width: int, height: int) -> Rows:
        mario, pose = self._standing()
        camera = render.view(*BUST_VIEW)
        left, right, bottom, top = render.Figure(mario, pose, ("head",)).bounds(camera)
        scale = BUST_HEAD * height / (top - bottom)
        margin = height * 3 / 80.0
        centre = ((left + right) / 2.0, top - (height / 2.0 - margin) / scale)
        rows = render.draw(render.Figure(mario, pose), camera, width, height, scale, centre, self.samples)
        start = height * (1.0 - BUST_FADE)
        for y in range(int(start), height):
            keep = max(0.0, 1.0 - (y + 0.5 - start) / (height - 1.0 - start))
            rows[y] = [(r, g, b, int(a * keep + 0.5)) for r, g, b, a in rows[y]]
        return rows

    def _stand(self, width: int, height: int) -> Rows:
        mario, pose = self._standing()
        camera = render.view(*STAND_VIEW)
        figure = render.Figure(mario, pose)
        scale, centre = render.fit(figure.bounds(camera), width, height, margin=2.0)
        return render.draw(figure, camera, width, height, scale, centre, self.samples)

    def _run(self, width: int, height: int, number: int, count: int, style: str) -> Rows:
        mario = self.mario()
        name, lean = RUN
        anim = mario.animation(name)
        camera = render.view(*RUN_VIEWS[style])
        key = (style, count)
        if key not in self._loops:
            # One cycle of his run over the loop's frames, all framed alike so that he bobs as he does.
            figures = []
            for k in range(count):
                t = k * anim.frames / float(count)
                figures.append(render.Figure(mario, mario.pose(anim.rotations(t), anim.root(t), math.radians(lean))))
            boxes = [figure.bounds(camera) for figure in figures]
            box = (
                min(b[0] for b in boxes),
                max(b[1] for b in boxes),
                min(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            self._loops[key] = (figures, box)
        figures, box = self._loops[key]
        scale, centre = render.fit(box, width, height, margin=2.0)
        return render.draw(figures[number], camera, width, height, scale, centre, self.samples)

    def _big(self, width: int, height: int, number: int, count: int, what: str) -> Rows:
        mario, standing = self._standing()
        camera = render.view(*BIG_VIEW)
        left, right, bottom, top = render.Figure(mario, standing).bounds(camera)
        scale = BIG_HEIGHT * height / (top - bottom)
        centre = (
            (left + right) / 2.0 - (BIG_FEET[0] - 0.5) * width / scale,
            bottom + (BIG_FEET[1] - 0.5) * height / scale,
        )
        tall = (top - bottom) / mario.scale  # his height in the units animations move him by

        def posed(name: str, t: float, lean: float = 0.0, lift: float = 0.0) -> list:
            anim = mario.animation(name)
            root = anim.root(t)
            return mario.pose(anim.rotations(t), (root[0], root[1] + lift, root[2]), math.radians(lean))

        up, down = BIG_JUMP_AIR
        if what == "wait":
            name, turn = BIG_WAIT
            pose = posed(name, turn * (1.0 - math.cos(2.0 * math.pi * number / count)) / 2.0)
        elif what == "run":
            first, length = BIG_RUN_LOOP
            cycle = mario.animation(RUN[0]).frames
            pose = posed(RUN[0], ((number - first) % length) * cycle / float(length), BIG_RUN_LEAN)
        elif number < up:
            name, frames = BIG_JUMP_CROUCH
            pose = posed(name, frames[min(number, len(frames) - 1)])
        elif number < down:
            name, a, b = BIG_JUMP_FLIP
            s = (number - up + 0.5) / float(down - up)
            lift = 4.0 * BIG_JUMP_RISE * tall * s * (1.0 - s)
            pose = posed(name, a + (b - a) * (number - up) / float(down - up - 1), lift=lift)
        else:
            name, a, b = BIG_JUMP_LAND
            pose = posed(name, a + (b - a) * (number - down) / float(count - 1 - down))
        return render.draw(render.Figure(mario, pose), camera, width, height, scale, centre, self.samples)

    # -- from his game's HUD -------------------------------------------------------------------

    def _glyph(self, width: int, height: int) -> Rows:
        """The HUD's head as large as whole pixels allow, centred."""
        zoom = min(width, height) // GLYPH
        if not zoom:
            raise BuildError(f"a {width}x{height} texture is too small for the HUD head")
        head = render.enlarge(self.rom.hud_glyph("mario"), zoom)
        rows = [[(255, 255, 255, 0)] * width for _ in range(height)]
        return paste(rows, head, (width - GLYPH * zoom) // 2, (height - GLYPH * zoom) // 2)

    def _emblem(self, width: int, height: int) -> Rows:
        """The HUD's head over a dark copy of its own outline, the way the leaders' map emblems are."""
        head = render.enlarge(self.rom.hud_glyph("mario"), EMBLEM_ZOOM)
        (dx, dy), colour = EMBLEM_SHADOW
        size = GLYPH * EMBLEM_ZOOM
        x, y = (width - size - abs(dx)) // 2 + max(0, -dx), (height - size - abs(dy)) // 2 + max(0, -dy)
        if x < 0 or y < 0:
            raise BuildError(f"a {width}x{height} texture is too small for the map emblem")
        rows = [[(*colour, 0)] * width for _ in range(height)]
        shadow = [[(*colour, px[3]) for px in row] for row in head]
        return paste(paste(rows, shadow, x + dx, y + dy), head, x, y)


class Layout:
    """
    One layout archive, unpacked.

    Attributes:
        raw: The `.szs` file.
        plain: The archive inside it.
        inner: The first and last byte, within `plain`, of the archive in it that holds its layouts
            and its texture file, or None if it has none.
        where: The first and last byte of its texture file within `plain`, or None if it has none.
        textures: The textures of that file, by name.
    """

    def __init__(self, name: str, raw: bytes):
        self.name, self.raw = name, raw
        self.plain = yaz0.decompress(raw)
        outer = sarc.Sarc(self.plain)
        self.inner = self.where = None
        for _, start, end in outer.entries if outer.is_archive else []:
            if self.plain[start : start + 4] != b"SARC":
                continue
            inner = sarc.Sarc(self.plain[start:end])
            if TEXTURES in inner.sizes():
                lo, hi = inner.span(TEXTURES)
                self.inner, self.where = (start, end), (start + lo, start + hi)
        self.textures = bntx.Bntx(self.plain[self.where[0] : self.where[1]]).textures if self.where else {}


def plan(layout: Layout) -> list[tuple[bntx.Texture, str, int, int]]:
    """
    The leader's textures in a layout archive, in the order they lie in the file.

    Returns:
        (texture, kind, number, how many there are of its kind) for each.

    Raises:
        BuildError: If a loop has more frames than it is taken to have.
    """
    found = []
    for pattern, kind in PICTURES:
        count = LOOPS.get(kind, 1)
        for texture in layout.textures.values():
            match = pattern.match(texture.name.split("^")[0])
            if not match:
                continue
            number = int(match.group(1)) if match.groups() else 0
            if kind in LOOPS and number >= count:
                raise BuildError(f"{layout.name}: {texture.name} is frame {number} of a loop taken to have {count}")
            found.append((texture, kind, number, count))
    return sorted(found, key=lambda entry: entry[0].offsets[0])


def encoded(texture: bntx.Texture, rows: Rows) -> Rows:
    """
    A picture in the channels the texture's format keeps.

    Returns:
        The rows to hand the encoder.

    Raises:
        BuildError: For a format none of the leaders' pictures is known to use.
    """
    if texture.codec == "BC3":
        return rows
    if texture.codec == "BC5" and not texture.signed:
        return grey(rows)
    raise BuildError(f"{texture.name} is {texture.describe()}: only colour (BC3) and grey (BC5) pictures are known")


def redraw(layout: Layout, artist: Artist) -> tuple[bytes, list[tuple[bntx.Texture, Rows]]]:
    """
    Draw over every picture of the leader's in a layout archive, in place.

    Returns:
        The archive's new bytes, and each texture changed with the picture drawn for it.
    """
    lo, hi = layout.where
    data = bytearray(layout.plain[lo:hi])
    drawn = []
    for texture, kind, number, count in plan(layout):
        rows = artist.picture(kind, number, texture.width, texture.height, count)
        bntx.replace(data, texture, encoded(texture, rows), search=True)
        drawn.append((texture, rows))
    return layout.plain[:lo] + bytes(data) + layout.plain[hi:], drawn


def light_panes(layout: bytes) -> list[tuple[int, str]]:
    """
    The panes of the leader's antenna light in the layout of his picture in the versus menu.

    Returns:
        Where each one starts in the layout, and its name: the glows, and the lights under them.

    Raises:
        BuildError: If the file is not a layout.
    """
    if layout[:4] != b"FLYT" or len(layout) < 0x14:
        raise BuildError(f"{LIGHT_LAYOUT} is not a layout")
    panes = []
    at = struct.unpack_from("<H", layout, 6)[0]
    for _ in range(struct.unpack_from("<H", layout, 0x10)[0]):
        size = struct.unpack_from("<I", layout, at + 4)[0] if at + 8 <= len(layout) else 0
        if size < 8 or at + size > len(layout):
            raise BuildError(f"{LIGHT_LAYOUT}: a section runs past the end of the file")
        if layout[at : at + 4] in PANES and size >= PANE_SIZE:
            name = layout[at + PANE_NAME : at + PANE_NAME + 24].split(b"\0")[0].decode("latin-1")
            if any(pattern.match(name) for pattern in LIGHT_PANES):
                panes.append((at, name))
        at += size
    return panes


def put_out_light(layout: Layout, plain: bytes) -> tuple[bytes, int]:
    """
    Switch off the leader's antenna light where a layout archive draws it over his picture.

    Only the versus menu's archive does. Each pane of the light is marked not to be drawn and given
    no opacity: two bytes a pane, in place. The light itself is one part that all five leaders'
    pictures share, and it is not touched; nor is any other leader's layout.

    Returns:
        The archive's bytes, and how many panes were switched off: none if it has no such layout.

    Raises:
        BuildError: If the layout does not have a glow, and a light under it, for each picture.
    """
    start, end = layout.inner
    inner = sarc.Sarc(plain[start:end])
    if LIGHT_LAYOUT not in inner.sizes():
        return plain, 0
    first, last = inner.span(LIGHT_LAYOUT)
    panes = light_panes(plain[start + first : start + last])
    glows = [name for _, name in panes if name.startswith("P_")]
    if not glows or len(panes) != 2 * len(glows):
        raise BuildError(f"{layout.name}: {LIGHT_LAYOUT} does not have a glow and a light under it for each picture")
    data = bytearray(plain)
    for at, _ in panes:
        data[start + first + at + PANE_FLAGS] &= 0xFE
        data[start + first + at + PANE_ALPHA] = 0
    return bytes(data), len(panes)


def build(rom: Rom, romfs: RomFs, mod: Mod) -> None:
    """
    Write every layout archive that shows the leader into the mod folder, redrawn.

    Raises:
        BuildError: If no archive has a picture of him.
    """
    logger.info("pictures:")
    artist = Artist(rom)
    done = 0
    for name in romfs.names(LAYOUTS, ".szs"):
        layout = Layout(PurePosixPath(name).stem, romfs.read(name))
        if not layout.where or not plan(layout):
            continue
        plain, drawn = redraw(layout, artist)
        plain, lights = put_out_light(layout, plain)
        mod.write(name, yaz0.patch(layout.raw, plain))
        logger.info("  %-16s %d of its %d textures", layout.name, len(drawn), len(layout.textures))
        if lights:
            logger.info("  %-16s the light of his antenna switched off over his picture: %d panes", "", lights)
        done += 1
    if not done:
        raise BuildError("no layout archive has a picture of the leader")
