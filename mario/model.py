"""
Alph becomes Mario: the model, his animations, and the game's list of clips.

What is written
---------------
* `player/pikminAll.szs`: the one model file every leader and Pikmin lives in, with Alph's model
  rebuilt as Mario and nine clips of his own added.
* `player/pikminAll_indirect.szs`: the game's list of animation clips, with those nine on the end.
* `effect/effect_common.szs`: the effects, with the light of Alph's antenna put out.

How
---
* **Skeleton.** The leaders' clips (shared with every Pikmin) turn and scale each bone but move
  only the root, so where a limb joins the body comes from each model's own skeleton. Alph's joints
  are moved to Mario's hips, neck and shoulders, their rest rotations kept, and Mario is bound in a
  rest pose that matches a leader's: standing, arms straight out.
* **Bones of his own.** A leader's arm or leg is one bone; Mario's has an elbow or a knee. Each of
  his 15 parts gets a bone of its own (`m_pelvis` ... `m_right_foot`), hung under the leader bone it
  would otherwise have been welded to. Every clip in the file then needs a track that tells each new
  bone to stay at rest, so he moves exactly as if welded. The file has to grow for that, which is
  what `formats/bfres_edit.py` is for.
* **Clips of his own.** In the walk and the run, the new bones carry curves as well: for every
  frame, where Mario's own animation puts that part, in the space of the leader bone above it. The
  leader's bones still do the leaders' walk underneath; his cancel it and replace it. Other leaders
  and the Pikmin have no bones by those names, so the same clips play on them unchanged.
* **Moves.** His punches, kick, jumps and landings are new clips, laid end to end from pieces of his
  animations. A clip is played by its number in the game's list, so the list gains an entry for
  each; `moves.py` is the code that plays them.
* **Mesh.** Alph's body and head shapes are overwritten with Mario's triangles, each vertex
  following one bone. Every other shape (helmet, antenna, backpack, eyes) is emptied.
* **Textures.** His materials are packed into one image per shape and encoded over Alph's colour
  texture, at its size and format. The normal map is flattened and the specular and roughness maps
  are set to one matte value.
* **Antenna light.** With the antenna gone its light would float over his head, so Alph's five light
  effects are made invisible.
"""

from __future__ import annotations

import logging
import math
import re
import struct

from dataclasses import dataclass
from operator import itemgetter
from typing import NamedTuple

from mario.errors import BuildError
from mario.formats import bfres_edit, bntx, sarc, vfxb, yaz0
from mario.formats.bfres import Bfres, Model, Shape, SkelAnim, Skeleton
from mario.geometry import (
    Matrix,
    Vector,
    cross,
    dot,
    euler_matrix,
    matmul,
    matrix_euler,
    sub,
    to_local,
    to_model,
    transpose,
    unit,
)
from mario.romfs import Mod, RomFs, repack
from mario.sm64.rom import Mario, Material, Rom

logger = logging.getLogger(__name__)

ARCHIVE = "player/pikminAll.szs"
SETTINGS = "player/pikminAll_indirect.szs"
EFFECTS = "effect/effect_common.szs"
MODEL_FILE = "model.bfres"
CLIP_LIST = "anim_define.txt"
EFFECT_FILE = "effect_common.ptcl"

LEADER = "A"  # Alph, whose model is `playerA`
HEIGHT = 21.3  # how tall Mario stands, in the game's units. Alph is 16.3 with his helmet on
# The lights the game hangs on the tip of a leader's antenna: the steady glow, and the flares for
# the whistle, dismiss, charge and dodge. Each leader has a copy of each, so one's can be put out.
LIGHTS = (
    "c3_player{}_light",
    "c3_light_whistle_per_player{}",
    "c3_light_kaisan_per_player{}",
    "c3_light_totsugeki_per_player{}",
    "c3_light_sidestep_per_player{}",
)
# The shapes that receive the new mesh, by material name; the first one present is used.
BODY_MATERIALS = ("body_m",)
HEAD_MATERIALS = ("head_m", "face_m")

# A part of Mario -> the leader bone that pivots there. Every other part follows its nearest listed
# ancestor, so a forearm follows the upper arm's bone.
RIG = {
    "pelvis": "sebonjnt",
    "head": "headjnt",
    "left_upperarm": "lhandjnt",
    "right_upperarm": "rhandjnt",
    "left_thigh": "llegjnt",
    "right_thigh": "rlegjnt",
}
OWN = "m_"  # the prefix of the bones added for his parts

# A clip of the leaders' -> (Mario's animation, how far he leans forward in it, in degrees).
# The leaders pick a clip by speed: walking up to 45, running above. His own game leans him in
# code, by his speed, up to 30 degrees at a run. The leaders' slowest clip, the step, cannot be done
# this way: it squashes and stretches their leg bones along their length, and a bone hung under one
# that is stretched along one axis cannot turn freely without shearing.
CLIPS = {
    "walk": ("walking", 15.0),
    "run2": ("running", 30.0),
}


class Piece(NamedTuple):
    """Frames `first` to `last` of one of his animations, the last then held for `held` more frames."""

    animation: str
    first: int
    last: int
    held: int = 0


@dataclass(frozen=True)
class Move:
    """
    A clip of his own, which no clip of the leaders' stands behind.

    Attributes:
        pieces: What it is laid end to end from.
        hit: The frame on which the punch state looks for something to hit (the clip's "action"
            marker), if the clip has one.
        more: The frame from which another press of the button is taken, if one is.
        swell: The part that swells, the count his game sets for it, and the frame it is set on.
        spark: The part the spark of a hit belongs on, and the frame of the clip to find it at. A
            punch or kick has it on the part that swells, on the frame it lands.
    """

    pieces: tuple[Piece, ...]
    hit: int | None = None
    more: int | None = None
    swell: tuple[str, int, int] | None = None
    spark: tuple[str, int] | None = None


# What his own game does in code is restated here.
#   * A punch is its animation up to two frames short of the end, then the "fast" animation that
#     draws the fist back, to its end. A press of the button during the second starts the next move.
#   * The fist swells from the punch's last frame on, the kicking foot from the kick's first: the
#     game sets a count there (4 for a punch, 6 for the kick) and runs it down a frame at a time
#     through a table of sizes (`Mario.swell_table`).
#   * A jump's animation plays once and its last frame is held for as long as he is in the air. The
#     double jump is two animations, the second from the top of the jump on, which here is 14 frames
#     after he leaves the ground. A landing is its animation, whole (the first jump's begins at
#     frame 22 of data it shares with another). No frame of these clips is "the one that lands":
#     the code decides that, so they carry no markers.
# The order is the order the code counts on: `moves.CLIPS`.
MOVES = {
    "m_punch1": Move(
        pieces=(Piece("first_punch", 0, 3), Piece("first_punch_fast", 0, 9)),
        hit=3,
        more=4,
        swell=("right_hand", 4, 3),
    ),
    "m_punch2": Move(
        pieces=(Piece("second_punch", 0, 2), Piece("second_punch_fast", 0, 12)),
        hit=2,
        more=3,
        swell=("left_hand", 4, 2),
    ),
    "m_kick": Move(pieces=(Piece("ground_kick", 0, 25),), hit=3, swell=("right_foot", 6, 0)),
    "m_jump1": Move(pieces=(Piece("single_jump", 0, 11, held=40),), spark=("right_foot", 11)),
    "m_jump2": Move(
        pieces=(Piece("double_jump_rise", 0, 5, held=8), Piece("double_jump_fall", 0, 8, held=30)),
        spark=("right_foot", 22),
    ),
    "m_jump3": Move(pieces=(Piece("triple_jump", 0, 35, held=24),), spark=("right_foot", 35)),
    "m_land1": Move(pieces=(Piece("land_single_jump", 22, 37),), spark=("right_foot", 0)),
    "m_land2": Move(pieces=(Piece("land_double_jump", 0, 17),), spark=("right_foot", 0)),
    "m_land3": Move(pieces=(Piece("triple_jump_land", 0, 31),), spark=("right_foot", 0)),
}
MOVE_MODEL = "punch"  # the stock clip a new one takes its header from: played once, not looped
# The leaders' punch draws its spark 4 units along the right arm's bone. In a move that bone has
# nothing to carry, so it is put where the spark belongs: at the fist or foot that lands.
HIT_BONE, HIT_REACH = "rhandjnt", 4.0
ACTION, MORE = 1024, 2048  # marker flags in the game's list of clips
LOOP_START, LOOP_END = 2, 4

# Flat material values for Mario (0-255). The specular map's alpha masks diffuse light on the stock
# leaders (0 only on the glass of the suit's gauges), so it stays near their usual value.
SPECULAR = (64, 64, 64, 240)
ROUGHNESS = (255, 255, 255, 255)
FLAT_NORMAL = (128, 128, 255, 255)

SOLID_TILE = 32  # pixels a single-colour material gets in the atlas
GUTTER = 2  # pixels between tiles
MARGIN = 2  # source texels baked around the part of a texture the triangles use

# Curve targets in a clip's track: the offset of the value each curve drives.
SCALE_CURVES = (0x4, 0x8, 0xC)
TRANSLATE_CURVES = (0x10, 0x14, 0x18)
ROTATE_CURVES = (0x20, 0x24, 0x28)

Frame = tuple[Matrix, Vector]  # where something sits: its rotation and its origin
Corner = tuple[Vector, Vector, tuple[float, float], Material, str]  # position, normal, uv, material, part
Triangle = list[Corner]
Plan = dict[int, tuple[Vector, Vector]]  # bone index -> its translation from its parent, its origin
Pose = dict[str, dict[str, list[float]]]  # bone -> {"S" | "R" | "T": values}, for the bones a clip moves
Target = tuple[Matrix, Vector, Matrix, Vector]  # the leader bone above a part: rotation, origin; the part's
Track = tuple[str, Vector, Vector, Vector | None, dict[int, list[float]]]  # bone, scale, rotation, place, curves
Image = tuple[int, int, list[list[tuple[int, int, int, int]]]]  # width, height, rows of RGBA


class Character:
    """
    Mario, read from the ROM and stood in the game's units.

    Attributes:
        mario: The model as his own game has it.
        tris: Every triangle at rest, as corners.
        frames: Where each part that has triangles sits at rest.
        stock: The leader bone each of those parts follows.
        pivot: Where each leader bone of the rig must sit.
        top: The highest point of his head.
    """

    def __init__(self, rom: Rom, height: float):
        self.mario = mario = Mario(rom)
        rest = mario.pose()
        tris = mario.triangles(rest)
        ys = [p[1] for _, _, tri in tris for p, _, _ in tri]
        self.source_height = max(ys) - min(ys)
        self.scale = height / self.source_height
        self.lift = -min(ys) * self.scale
        self.tris: list[Triangle] = [
            [(self.place(p), n, (uv[0], 1.0 - uv[1]), material, part) for p, n, uv in tri]
            for material, part, tri in tris
        ]
        self.frames: dict[str, Frame] = {
            part.name: (rot, self.place_part(pos))
            for part, (rot, pos) in zip(mario.parts, rest, strict=True)
            if part.tris
        }
        names = [p.name for p in mario.parts]
        self.stock: dict[str, str] = {}
        for part in mario.parts:
            if not part.tris:
                continue  # a bare joint: the root, a shoulder, a hip
            i = names.index(part.name)
            while names[i] not in RIG:
                if mario.parts[i].parent is None:
                    raise BuildError(f"part {part.name} has no ancestor in the rig")
                i = mario.parts[i].parent
            self.stock[part.name] = RIG[names[i]]
        self.pivot = {bone: self.place_part(rest[names.index(part)][1]) for part, bone in RIG.items()}
        self.top = max((c[0] for tri in self.tris for c in tri if self.stock[c[4]] == "headjnt"), key=itemgetter(1))

    def place(self, point: Vector) -> Vector:
        """A point of the model, already at the model's own scale, in the game's units."""
        return (point[0] * self.scale, point[1] * self.scale + self.lift, point[2] * self.scale)

    def place_part(self, position: Vector) -> Vector:
        """A part's position as `Mario.pose` gives it, in the game's units."""
        return self.place(tuple(c * self.mario.scale for c in position))


# -- the skeleton ----------------------------------------------------------------------------------


def plan_skeleton(skel: Skeleton, char: Character) -> tuple[list[Frame], Plan]:
    """
    New rest positions for the leader's bones.

    Rotations are kept, so each bone's rest orientation, and with it what an animation's angles
    mean, is what it was.

    Returns:
        The stock skeleton's rest pose, and the plan.

    Raises:
        BuildError: If the skeleton is not the one the rig was written for.
    """
    world = skel.world()
    names = [b["name"] for b in skel.bones]
    for b in skel.bones:
        if any(abs(s - 1.0) > 1e-6 for s in b["scale"]):
            raise BuildError(f"bone {b['name']} has a rest scale, which is not handled")
    missing = sorted(set(RIG.values()) - set(names))
    if missing:
        raise BuildError(f"the leader has no bone named {', '.join(missing)}")
    target = dict(char.pivot)
    target["legcentre"] = char.pivot["sebonjnt"]  # the legs' parent turns about the pelvis too
    # The antenna is gone; fold its bones onto the top of the head so anything the game hangs on
    # the tip sits there. A little length is left in case code normalises it.
    x, y, z = char.top
    target["happajnt1"], target["happajnt2"], target["happajnt3"] = (x, y, z), (x, y + 0.1, z), (x, y + 0.2, z)
    plan: Plan = {}
    for i, b in enumerate(skel.bones):
        parent = b["parent"]
        if parent < 0:
            # The root's translation is animated (it is the hip height), so it is left alone.
            plan[i] = (tuple(b["trans"]), tuple(world[i][1]))
            continue
        prot, porigin = world[parent][0], plan[parent][1]
        if b["name"] in target:
            origin = target[b["name"]]
            local = to_local(prot, porigin, origin)
        else:
            local = tuple(b["trans"])
            origin = to_model(prot, porigin, local)
        plan[i] = (local, origin)
    return world, plan


def write_skeleton(f: Bfres, skel: Skeleton, plan: Plan) -> tuple[Skeleton, list[Frame]]:
    """
    Store the leader bones' new translations, then every skinned bone's inverse rest matrix.

    Returns:
        The skeleton read back as stored, and its rest pose.
    """
    for i, (local, _) in plan.items():
        b = skel.bones[i]
        struct.pack_into("<3f", f.d, b["off"] + 0x54, *local)
        # A stock bone at its parent's origin carries a "translation is zero" hint; clear it once it moves.
        flags = b["flags"] & ~0x08000000 if any(abs(c) > 1e-9 for c in local) else b["flags"]
        struct.pack_into("<I", f.d, b["off"] + 0x34, flags)
    skel = Skeleton(f, skel.off)
    world = skel.world()
    for k, bone in enumerate(skel.matrix_bones[: skel.nsmooth]):
        rot, origin = world[bone]
        rows = []
        for r in range(3):
            rows += [
                rot[0][r],
                rot[1][r],
                rot[2][r],
                -(rot[0][r] * origin[0] + rot[1][r] * origin[1] + rot[2][r] * origin[2]),
            ]
        struct.pack_into("<12f", f.d, skel.inverse_off + k * 48, *rows)
    return skel, world


# -- clips -----------------------------------------------------------------------------------------


def clip_loops(clip_list: bytes) -> dict[str, tuple[int, int]]:
    """The first and last frame of each clip's loop for the leaders, from the markers in the game's list."""
    text = clip_list.decode("utf-8", "replace")
    loops: dict[str, tuple[int, int]] = {}
    for define, clip, body in re.findall(r'\{\s*"([^"]+)"\s*"([^"]+)"\s*\d+[ \t]*(.*?)\n\t\}', text, re.DOTALL):
        events = [(float(a), int(b)) for a, b in re.findall(r'^\s*([0-9.]+) (\d+) "', body, re.MULTILINE)]
        start = [frame for frame, flag in events if flag == LOOP_START]
        end = [frame for frame, flag in events if flag == LOOP_END]
        if start and end and (define.startswith("player_") or clip not in loops):
            loops[clip] = (int(start[0]), int(end[0]))
    return loops


def define_moves(clip_list: bytes, moves: dict[str, Move]) -> tuple[bytes, int]:
    """
    The game's list of clips with an entry for each move on the end of its skeletal section.

    A clip is played by its place in this list. Entries are only ever added at the end, so every
    number the game's own code plays a clip by still means what it did. Each move's tag is its
    clip's name, with a marker on the frame it lands and, if another move may follow it, one on the
    frame from which the next press is taken.

    Returns:
        The new list, and the number of the first move in it.

    Raises:
        BuildError: If the list is not laid out as expected.
    """
    head, opened, rest = clip_list.partition(b"\r\n{\r\n")
    body, closed, tail = rest.partition(b"\r\n}\r\n")
    lines = head.split(b"\r\n")
    counts = lines[1].split() if len(lines) == 3 else []
    if not opened or not closed or not counts or not counts[0].isdigit():
        raise BuildError(f"{CLIP_LIST} is not laid out as expected")
    first = int(counts[0])
    added = ""
    for name, move in moves.items():
        marks = sorted(
            (float(frame), flag) for frame, flag in ((move.hit, ACTION), (move.more, MORE)) if frame is not None
        )
        added += f'\r\n\t# \r\n\t{{\r\n\t\t"{name}" \r\n\t\t"{name}" \r\n\t\t{len(marks)} '
        for frame, flag in marks:
            added += f'\r\n\t\t{frame:.8f} {flag} "" '
        added += "\r\n\t}"
    counts[0] = str(first + len(moves)).encode()
    lines[1] = b" ".join(counts) + b" "
    return b"\r\n".join(lines) + opened + body + added.encode() + closed + tail, first


def stock_world(skel: Skeleton, plan: Plan, pose: Pose) -> list[Frame]:
    """
    The leader's own bones under a clip's pose, on the planned skeleton.

    Returns:
        Where each bone sits, in the skeleton's order.

    Raises:
        BuildError: If the clip scales a bone, which a part hung under it could not cancel.
    """
    out: list[Frame] = []
    for i in sorted(plan):
        b = skel.bones[i]
        p = pose.get(b["name"], {})
        scale = p.get("S", b["scale"])
        if any(abs(c - 1.0) > 1e-4 for c in scale):
            raise BuildError(f"a clip scales bone {b['name']}; a part hung under it could not cancel that")
        rot = euler_matrix(*p.get("R", b["rot"])[:3])
        trans = p.get("T", plan[i][0])
        if b["parent"] >= 0:
            prot, porigin = out[b["parent"]]
            out.append((matmul(prot, rot), to_model(prot, porigin, trans)))
        else:
            out.append((rot, tuple(trans)))
    return out


def clip_targets(
    char: Character,
    clip: SkelAnim,
    loop: tuple[int, int],
    which: str,
    lean: float,
    skel: Skeleton,
    plan: Plan,
) -> tuple[int, list[dict[str, Target]]]:
    """
    For each frame of a leaders' clip, where Mario's animation wants every part.

    His cycle is stretched over the clip's loop and repeated outside it, and slid in time so that
    his legs swing forward when the leaders' do: the footstep sounds and dust are timed to theirs.

    Returns:
        The frame of his cycle that the loop starts on, and the targets frame by frame.
    """
    mario = char.mario
    anim = mario.animation(which)
    names = [p.name for p in mario.parts]
    first, last = loop
    span = float(last - first)
    stock = [stock_world(skel, plan, clip.pose(float(frame))) for frame in range(clip.frames + 1)]
    bone = {b["name"]: i for i, b in enumerate(skel.bones)}

    def tau(frame: int, phase: int) -> float:
        return phase + (frame - first) * anim.frames / span

    def poses(t: float) -> list[Frame]:
        return mario.pose(anim.rotations(t), anim.root(t), math.radians(lean))

    def centred(row: list[float]) -> list[float]:
        return [v - sum(row) / len(row) for v in row]

    # how far forward each leg points, theirs per clip frame and his per frame of his own cycle
    theirs = [centred([stock[frame][bone[b]][0][2][0] for frame in range(first, last)]) for b in ("llegjnt", "rlegjnt")]
    his = []
    for side in ("left", "right"):
        row = []
        for t in range(anim.frames):
            p = poses(float(t))
            row.append(p[names.index(side + "_foot")][1][2] - p[names.index(side + "_thigh")][1][2])
        his.append(centred(row))

    def score(phase: int) -> float:
        return sum(
            theirs[s][frame - first] * his[s][round(tau(frame, phase)) % anim.frames]
            for s in range(2)
            for frame in range(first, last)
        )

    phase = max(range(anim.frames), key=score)
    frames = []
    for frame in range(clip.frames + 1):
        p = poses(tau(frame, phase))
        row = {}
        for part in char.frames:
            rot, pos = p[names.index(part)]
            prot, porigin = stock[frame][bone[char.stock[part]]]
            row[part] = (prot, porigin, rot, char.place_part(pos))
        frames.append(row)
    return phase, frames


def move_steps(move: Move) -> list[tuple[str, int]]:
    """The (animation, frame) shown on each frame of a move."""
    steps = []
    for piece in move.pieces:
        steps += [(piece.animation, frame) for frame in range(piece.first, piece.last + 1)]
        steps += [(piece.animation, piece.last)] * piece.held
    return steps


def move_spark(move: Move) -> tuple[str, int]:
    """The part the spark of the move's hit belongs on, and the frame of the clip to find it at."""
    return move.spark or (move.swell[0], move.hit)


def move_swell(mario: Mario, move: Move, count: int) -> list[float]:
    """The size of the move's swelling part at each of `count` frames."""
    sizes = [1.0] * count
    if move.swell:
        part, start, at = move.swell
        table = mario.swell_table()[part]
        for k in range(start):  # the count runs start-1, start-2, ... 0 from frame `at`
            if at + k < count:
                sizes[at + k] = table[start - 1 - k]
    return sizes


def move_targets(char: Character, move: Move, skel: Skeleton, plan: Plan) -> tuple[Pose, list[dict[str, Target]]]:
    """
    For each frame of a move, where his animation wants every part, and where the leader's bones are.

    Returns:
        What the clip tells the leader's own bones (to rest, but for the bone the punch draws its
        spark on, which is moved so that the spark's point is the middle of the part that lands, on
        the frame it lands), and the targets frame by frame.
    """
    mario = char.mario
    names = [p.name for p in mario.parts]
    bone = {b["name"]: i for i, b in enumerate(skel.bones)}
    poses = [
        mario.pose(mario.animation(which).rotations(float(frame)), mario.animation(which).root(float(frame)))
        for which, frame in move_steps(move)
    ]
    part, at = move_spark(move)
    corners = [p for _, name, tri in mario.triangles(poses[at]) if name == part for p, _, _ in tri]
    point = char.place(tuple(sum(p[k] for p in corners) / len(corners) for k in range(3)))
    rest = stock_world(skel, plan, {})
    i = bone[HIT_BONE]
    parent = skel.bones[i]["parent"]
    origin = to_model(rest[i][0], point, (-HIT_REACH, 0.0, 0.0))
    pose = {HIT_BONE: {"T": list(to_local(rest[parent][0], rest[parent][1], origin))}}
    stock = stock_world(skel, plan, pose)
    frames = []
    for p in poses:
        row = {}
        for name in char.frames:
            rot, pos = p[names.index(name)]
            prot, porigin = stock[bone[char.stock[name]]]
            row[name] = (prot, porigin, rot, char.place_part(pos))
        frames.append(row)
    return pose, frames


def move_rest(skel: Skeleton, plan: Plan, pose: Pose) -> list[Track]:
    """
    A move's tracks for the leader's own bones: every one at rest, but for what `pose` says.

    As in the stock clips, only the root (and here the spark's bone) is told where to be; the others
    keep the place their model gives them.

    Returns:
        One track for each bone, in the skeleton's order.
    """
    tracks = []
    for i, b in enumerate(skel.bones):
        trans = pose.get(b["name"], {}).get("T", plan[i][0] if b["parent"] < 0 else None)
        tracks.append((b["name"], (1.0, 1.0, 1.0), tuple(b["rot"][:3]), None if trans is None else tuple(trans), {}))
    return tracks


def _quarter_turns() -> list[Matrix]:
    """The 24 ways to turn a cube onto itself."""
    seen, out = set(), []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                m = euler_matrix(a * math.pi / 2, b * math.pi / 2, c * math.pi / 2)
                m = [[float(round(v)) for v in row] for row in m]
                key = tuple(tuple(row) for row in m)
                if key not in seen:
                    seen.add(key)
                    out.append(m)
    return out


def choose_turn(samples: list[Matrix]) -> Matrix:
    """
    A fixed turn for a part's bone that keeps its angles away from the pole.

    Three angles cannot describe a rotation smoothly when the middle one nears 90 degrees: the other
    two swing wildly from frame to frame, and the game, which blends between frames, would show it.
    Which rotations land there depends on how the bone's own axes are set, and that is free: the
    mesh is bound to whatever they are. `samples` are the rotations the bone will be asked for.

    Returns:
        The quarter turn that keeps the samples furthest from the pole.
    """
    best = None
    for turn in _quarter_turns():
        worst = max(abs(matmul(rot, turn)[2][0]) for rot in samples)
        if best is None or worst < best[0] - 1e-9:
            best = (worst, turn)
    return best[1]


def plan_own(
    char: Character,
    skel: Skeleton,
    world: list[Frame],
    plan: Plan,
    targets: dict[str, list[dict[str, Target]]],
    swells: dict[str, dict[str, list[float]]],
) -> tuple[list[tuple], dict[str, list[Track]], list[Track]]:
    """
    Mario's own bones, and their tracks for every clip in `targets`.

    `swells` gives, for a clip, a part's size at each frame.

    Returns:
        The bones as `Grower.add_bones` takes them, each clip's tracks for them, and the tracks that
        hold them at rest in every other clip.
    """
    names = [b["name"] for b in skel.bones]
    bones: list[tuple] = []
    tracks: dict[str, list[Track]] = {clip: [] for clip in targets}
    for part in sorted(char.frames, key=lambda p: (names.index(char.stock[p]), p)):
        parent = names.index(char.stock[part])
        prot, porigin = world[parent][0], plan[parent][1]
        rot0, origin = char.frames[part]
        samples = [matmul(transpose(prot), rot0)]
        for frames in targets.values():
            samples += [matmul(transpose(row[part][0]), row[part][2]) for row in frames]
        turn = choose_turn(samples)
        angles = matrix_euler(matmul(samples[0], turn))
        trans = to_local(prot, porigin, origin)
        bones.append((OWN + part, parent, angles, trans, True))
        for clip, frames in targets.items():
            rots, places, near = [], [], angles
            for p_rot, p_origin, want_rot, want_pos in (row[part] for row in frames):
                near = matrix_euler(matmul(matmul(transpose(p_rot), want_rot), turn), near)
                rots.append(near)
                places.append(to_local(p_rot, p_origin, want_pos))
            sizes = swells.get(clip, {}).get(part, [1.0] * len(frames))
            curves = {target: [r[k] for r in rots] for k, target in enumerate(ROTATE_CURVES)}
            curves.update({target: [t[k] for t in places] for k, target in enumerate(TRANSLATE_CURVES)})
            curves.update(dict.fromkeys(SCALE_CURVES, sizes))
            curves = {target: values for target, values in curves.items() if max(values) - min(values) > 1e-6}
            tracks[clip].append((OWN + part, (sizes[0],) * 3, rots[0], places[0], curves))
    still = [(name, (1.0, 1.0, 1.0), angles, trans, {}) for name, _, angles, trans, _ in bones]
    return bones, tracks, still


# -- textures --------------------------------------------------------------------------------------


def _sample(image: Image, u: float, v: float) -> tuple[int, int, int, int]:
    """A bilinear sample of an image at a texture coordinate (v up), clamped at the edges."""
    w, h, rows = image
    x = min(max(u * w - 0.5, 0.0), w - 1.0)
    y = min(max((1.0 - v) * h - 0.5, 0.0), h - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    a, b, c, e = rows[y0][x0], rows[y0][x1], rows[y1][x0], rows[y1][x1]
    return (
        *(round((a[i] * (1 - fx) + b[i] * fx) * (1 - fy) + (c[i] * (1 - fx) + e[i] * fx) * fy) for i in range(3)),
        255,
    )


class Atlas:
    """One image holding every material a shape uses, and the mapping from the old coordinates."""

    def __init__(self, char: Character, tris: list[Triangle], width: int, height: int):
        """
        Pack and draw the materials of `tris` into a `width` by `height` image.

        Raises:
            BuildError: If they do not fit even at their own size.
        """
        self.width, self.height = width, height
        images: dict[Material, Image] = {}
        used: dict[Material, list[float]] = {}
        for tri in tris:
            for _, _, uv, material, _ in tri:
                if material not in images:
                    images[material] = char.mario.image(material)
                r = used.setdefault(material, [uv[0], uv[0], uv[1], uv[1]])
                r[0], r[1], r[2], r[3] = min(r[0], uv[0]), max(r[1], uv[0]), min(r[2], uv[1]), max(r[3], uv[1])
        self.solid: dict[Material, tuple[int, int, int]] = {}
        for material, (_, _, rows) in images.items():
            first = tuple(rows[0][0][:3])
            if all(tuple(p[:3]) == first for row in rows for p in row):
                self.solid[material] = first
        self.tiles: dict[Material, tuple] = {}
        for zoom in (4, 3, 2, 1):
            if self._pack(images, used, zoom):
                break
        else:
            raise BuildError(f"the materials do not fit a {width}x{height} texture")
        self.zoom = zoom
        background = (*next(iter(self.solid.values()), (128, 128, 128)), 255)
        self.rows = [[background] * width for _ in range(height)]
        for material, (x, y, w, h, u0, u1, v0, v1) in self.tiles.items():
            for py in range(h):
                v = v1 - (py + 0.5) / h * (v1 - v0)
                row = self.rows[y + py]
                for px in range(w):
                    if material in self.solid:
                        row[x + px] = (*self.solid[material], 255)
                    else:
                        row[x + px] = _sample(images[material], u0 + (px + 0.5) / w * (u1 - u0), v)

    def _pack(self, images: dict[Material, Image], used: dict[Material, list[float]], zoom: int) -> bool:
        """Shelf-pack the tiles, tallest first; False if they overflow at this zoom."""
        want = []
        for material, (w, h, _) in images.items():
            if material in self.solid:
                want.append((SOLID_TILE, SOLID_TILE, material.name, material, 0.0, 1.0, 0.0, 1.0))
                continue
            u0, u1, v0, v1 = used[material]
            u0, u1, v0, v1 = u0 - MARGIN / w, u1 + MARGIN / w, v0 - MARGIN / h, v1 + MARGIN / h
            want.append(
                (
                    math.ceil((u1 - u0) * w * zoom),
                    math.ceil((v1 - v0) * h * zoom),
                    material.name,
                    material,
                    u0,
                    u1,
                    v0,
                    v1,
                ),
            )
        self.tiles = {}
        x = y = GUTTER
        shelf = 0
        for tw, th, _, material, u0, u1, v0, v1 in sorted(want, key=lambda t: (-t[1], t[2])):
            if x + tw + GUTTER > self.width:
                x, y, shelf = GUTTER, y + shelf + GUTTER, 0
            if x + tw + GUTTER > self.width or y + th + GUTTER > self.height:
                return False
            self.tiles[material] = (x, y, tw, th, u0, u1, v0, v1)
            x += tw + GUTTER
            shelf = max(shelf, th)
        return True

    def map(self, material: Material, uv: tuple[float, float]) -> tuple[float, float]:
        """An old coordinate (v up) as the atlas's (v down, as the game's textures are addressed)."""
        x, y, w, h, u0, u1, v0, v1 = self.tiles[material]
        if material in self.solid:
            return ((x + w / 2) / self.width, (y + h / 2) / self.height)  # one point: never blurs or bleeds
        return ((x + (uv[0] - u0) / (u1 - u0) * w) / self.width, (y + (v1 - uv[1]) / (v1 - v0) * h) / self.height)


# -- one shape -------------------------------------------------------------------------------------


def weld(tris: list[Triangle], atlas: Atlas, palette: dict[str, int]) -> tuple[list[tuple], list[int]]:
    """
    Turn triangles of corners into vertices and indices.

    Returns:
        The vertices, each (position, normal, uv, matrix number, part), and three indices a triangle.
    """
    verts, index, indices = [], {}, []
    for tri in tris:
        for pos, nrm, uv, material, part in tri:
            vert = (pos, nrm, atlas.map(material, uv), palette[part], part)
            key = (
                tuple(round(c, 4) for c in pos),
                tuple(round(c, 3) for c in nrm),
                tuple(round(c, 5) for c in vert[2]),
                part,
            )
            if key not in index:
                index[key] = len(verts)
                verts.append(vert)
            indices.append(index[key])
    return verts, indices


def write_shape(shape: Shape, verts: list[tuple], indices: list[int], frames: dict[str, Frame]) -> None:
    """
    Overwrite a shape's vertices, triangles and bounds. `frames` is where each part's bone rests.

    Raises:
        BuildError: If the new mesh does not fit the shape's buffers.
    """
    if len(verts) > shape.nvert:
        raise BuildError(f"{shape.name}: {len(verts)} vertices do not fit in {shape.nvert}")
    tangents, bitangents = [], []
    for _, nrm, _, _, _ in verts:
        # The normal map is flat, so any frame square to the normal lights the same.
        tangent = unit(cross((0.0, 1.0, 0.0), nrm)) if abs(nrm[1]) < 0.99 else (1.0, 0.0, 0.0)
        tangents.append(tangent)
        bitangents.append(unit(cross(nrm, tangent)))
    values = {
        "_p0": [v[0] for v in verts],
        "_n0": [v[1] for v in verts],
        "_t0": tangents,
        "_b0": bitangents,
        "_c0": [(1.0, 1.0, 1.0, 1.0)] * len(verts),
        "_u0": [v[2] for v in verts],
        "_i0": [(v[3], v[3]) for v in verts],
        "_w0": [(1.0, 0.0)] * len(verts),
    }
    for name in shape.attrs:
        if name not in values:
            raise BuildError(f"{shape.name}: no rule for vertex attribute {name}")
        shape.write(name, values[name])
    shape.set_indices(indices)
    # Bounds, measured the way the stock ones are: each vertex in the space of the bone it follows.
    local = [to_local(frames[v[4]][0], frames[v[4]][1], v[0]) for v in verts]
    lo = [min(p[i] for p in local) for i in range(3)]
    hi = [max(p[i] for p in local) for i in range(3)]
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = max(math.sqrt(dot(sub(p, centre), sub(p, centre))) for p in local)
    shape.set_bounds(centre, [(hi[i] - lo[i]) / 2 for i in range(3)], radius)


# -- the build -------------------------------------------------------------------------------------


def shape_of(model: Model, materials: tuple[str, ...]) -> Shape:
    """
    The first shape of a model drawn with one of `materials`.

    Returns:
        The shape.

    Raises:
        BuildError: If the model has no such shape.
    """
    for shape in model.shapes:
        if model.mats[shape.mat].name in materials:
            return shape
    raise BuildError(f"{model.name} has no shape with material {' or '.join(materials)}")


def body_and_head(model: Model) -> tuple[Shape, Shape]:
    """The two shapes of a leader's model that take the new mesh."""
    return shape_of(model, BODY_MATERIALS), shape_of(model, HEAD_MATERIALS)


def split(char: Character) -> tuple[list[Triangle], list[Triangle]]:
    """His triangles for the body shape, and for the head shape."""
    body, head = [], []
    for tri in char.tris:
        (head if all(char.stock[c[4]] == "headjnt" for c in tri) else body).append(tri)
    return body, head


def remodel(stock: bytes, char: Character, loops: dict[str, tuple[int, int]]) -> bytes:
    """
    The leaders' model file with Alph rebuilt as Mario.

    `loops` is what `clip_loops` read from the game's list of clips.

    Returns:
        The new model file.

    Raises:
        BuildError: If the stock model is not put together the way this expects.
    """
    f0 = Bfres(stock)
    model0 = f0.model("player" + LEADER)
    skel0 = model0.skel
    world0, plan = plan_skeleton(skel0, char)
    body_tris, head_tris = split(char)
    stock_clips = {clip.name: clip for clip in f0.skel_anims()}

    # Where his animations put each part, in every clip that will carry curves for his bones.
    targets, swells, poses = {}, {}, {}
    for name, (which, lean) in CLIPS.items():
        clip = stock_clips[name]
        loop = loops.get(name, (0, clip.frames))
        phase, targets[name] = clip_targets(char, clip, loop, which, lean, skel0, plan)
        logger.info("  clip %-8s his %s, from frame %d of it, leaning %g degrees", name, which, phase, lean)
    for name, move in MOVES.items():
        poses[name], targets[name] = move_targets(char, move, skel0, plan)
        if move.swell:
            swells[name] = {move.swell[0]: move_swell(char.mario, move, len(targets[name]))}
        pieces = " + ".join(f"{piece.animation} {piece.first}-{piece.last}" for piece in move.pieces)
        logger.info("  move %-8s %2d frames: %s", name, len(targets[name]), pieces)
    bones, tracks, still = plan_own(char, skel0, world0, plan, targets, swells)

    # Grow the file: his bones, a track for each in every clip, then the new clips.
    grower = bfres_edit.Grower(stock)
    first = grower.add_bones(skel0.off, bones)[0]
    index = {name: first + k for k, (name, *_) in enumerate(bones)}
    for shape, tris in zip(body_and_head(model0), (body_tris, head_tris), strict=True):
        grower.set_skin_bones(shape.off, sorted({index[OWN + c[4]] for tri in tris for c in tri}))
    for clip in f0.skel_anims():
        grower.add_tracks(clip.o, tracks.get(clip.name, still))
    # Every change to a stock clip is made; from here the clip list is a copy with the moves on its end.
    like = stock_clips[MOVE_MODEL].o
    new_clips = grower.add_clips([(name, len(targets[name]) - 1, like) for name in MOVES])
    for offset, name in zip(new_clips, MOVES, strict=True):
        grower.add_tracks(offset, move_rest(skel0, plan, poses[name]) + tracks[name])
    data = bytearray(grower.finish())
    logger.info(
        "  %d bones and %d clips added; the model file grows by %d bytes",
        len(bones),
        len(MOVES),
        len(data) - len(stock),
    )

    # Then, in place: the leader's joints, the mesh and the textures.
    f = Bfres(data)
    model = f.model("player" + LEADER)
    skel, world = write_skeleton(f, model.skel, plan)
    names = [b["name"] for b in skel.bones]
    palette, frames = {}, {}
    for part in char.frames:
        bone = names.index(OWN + part)
        if bone not in skel.matrix_bones[: skel.nsmooth]:
            raise BuildError(f"bone {names[bone]} has no skinning matrix in {model.name}")
        palette[part] = skel.matrix_bones.index(bone)
        frames[part] = world[bone]
    body, head = body_and_head(model)
    textures = bntx.Bntx(data)
    for shape, tris in ((body, body_tris), (head, head_tris)):
        material = model.mats[shape.mat]
        loose = {OWN + c[4] for tri in tris for c in tri} - {names[b] for b in shape.skinbones}
        if loose:
            raise BuildError(f"{shape.name} is not skinned to {', '.join(sorted(loose))}")
        colour = textures[material.role["_a0"]]
        atlas = Atlas(char, tris, colour.width, colour.height)
        verts, indices = weld(tris, atlas, palette)
        write_shape(shape, verts, indices, frames)
        bntx.replace(data, colour, atlas.rows)
        for role, value in (("_n0", FLAT_NORMAL), ("_s0", SPECULAR), ("_r0", ROUGHNESS)):
            if role in material.role:
                bntx.fill(data, textures[material.role[role]], value)
        logger.info(
            "  %-22s %d triangles, %d vertices, %d materials",
            shape.name,
            len(tris),
            len(verts),
            len(atlas.tiles),
        )
    for shape in model.shapes:
        if shape is not body and shape is not head:
            shape.set_indices([])
    return bytes(data)


def hide_lights(ptcl: bytes) -> bytes:
    """
    The effect file with the lights of the leader's antenna made invisible.

    Returns:
        The new effect file, the same size as the old.

    Raises:
        BuildError: If the effects are not found, or do not look like what is expected.
    """
    data = bytearray(ptcl)
    wanted = [name.format(LEADER) for name in LIGHTS]
    for _, name, emitters in vfxb.emitter_blocks(data):
        if name not in wanted:
            continue
        wanted.remove(name)
        if not emitters:
            raise BuildError(f"effect {name} has no emitters")
        for emitter, block in emitters:
            colours = struct.unpack_from("<8f", data, block + vfxb.EMITTER_COLOR0)
            size = struct.unpack_from("<3f", data, block + vfxb.EMITTER_SCALE)
            # Refuse anything that does not hold colours and a size where a light's emitter does.
            if not (all(0.0 <= c <= 4.0 for c in colours) and all(0.5 <= x <= 500.0 for x in size)):
                raise BuildError(f"effect {name} / {emitter} does not hold colours and a size where expected")
            vfxb.hide_emitter(data, block)
    if wanted:
        raise BuildError(f"effects not found: {', '.join(wanted)}")
    return bytes(data)


def build(rom: Rom, romfs: RomFs, mod: Mod) -> int:
    """
    Write the model, the list of clips and the effects into the mod folder.

    Returns:
        The number of his first move in the game's list of clips.
    """
    char = Character(rom, HEIGHT)
    logger.info(
        "model: %d triangles, %.1f units tall in his own game, %g here",
        len(char.tris),
        char.source_height,
        HEIGHT,
    )
    archive, settings, effects = romfs.read(ARCHIVE), romfs.read(SETTINGS), romfs.read(EFFECTS)
    clip_list = sarc.Sarc(yaz0.decompress(settings)).files()[CLIP_LIST]
    model = remodel(sarc.Sarc(yaz0.decompress(archive)).files()[MODEL_FILE], char, clip_loops(clip_list))
    logger.info("  compressing %s (this is the slow part)", ARCHIVE)
    mod.write(ARCHIVE, repack(archive, {MODEL_FILE: model}))
    new_list, first = define_moves(clip_list, MOVES)
    mod.write(SETTINGS, repack(settings, {CLIP_LIST: new_list}))
    logger.info("  his moves are clips %#x to %#x of the game's list", first, first + len(MOVES) - 1)
    ptcl = sarc.Sarc(yaz0.decompress(effects)).files()[EFFECT_FILE]
    mod.write(EFFECTS, repack(effects, {EFFECT_FILE: hide_lights(ptcl)}))
    return first
