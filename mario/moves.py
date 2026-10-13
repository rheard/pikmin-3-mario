"""
The code this mod adds to the game: Mario's punch, punch, kick, his three jumps, and his fall.

Everything else the mod does is data the game already reads. A new move is new behaviour, so this
part is code: an IPS patch of the game's program, which a loader applies before the game starts.
The code itself is the assembly in `asm/`; this module gives it its numbers, assembles it and
writes the patch.

What the game has
-----------------
* **A punch.** A leader with no Pikmin, pressing A beside an enemy, enters a punch state. Entering
  it plays the punch clip; each frame the state asks the animation which markers it passed, on the
  "action" marker it searches a sphere just ahead of him and hits the nearest enemy, and when the
  clip ends the state ends.
* **No jump.** But a leader is a body in the game's physics: it has a velocity the walk state
  steers, gravity pulls it down, and a byte says whether it is touching the ground. Setting the
  upward part of that velocity is a jump, and the game brings him down again.
* **A hit from above.** A thrown Pikmin landing on an enemy sends it a hit of its own kind, which
  kills a Dwarf Bulborb outright. Each enemy's parameters say how many such hits it takes.
* **No fall.** A leader who walks off an edge stays in the walk state, walking on the air, and
  cannot be steered until he lands.

What the code does, for the one leader whose model is Mario's and for nobody else
---------------------------------------------------------------------------------
* **Punch, punch, kick** (`asm/punch.s`). Entering the punch state plays his first punch. Once a
  clip has passed its "another press is taken from here" marker, a press of A plays the next move
  and keeps the state alive. The game's own code still does the hitting, on each clip's action
  marker, with his numbers in place of a leader's: harder, and reaching further.
* **Jumps** (`asm/jump.s`). D-pad Down, which the game maps to nothing, or a click of the right
  stick, enters the same punch state with different answers: the clip is a jump's, "look for
  something to hit" is every frame he is coming down, the place searched is under him, and the
  state ends when he has landed. A press within a few frames of landing is the next jump up; the
  third needs a run-up, as in his own game. In the air the walk state's own steering is called, so
  the stick moves him as it does on the ground.
* **Landing on an enemy** sends the hit a thrown Pikmin's landing makes and, if the enemy's life
  did not move, an ordinary blow twice as strong as his punch. Either way he bounces off.
* **Falling** (the end of `asm/jump.s`). While he walks, his frames off the ground are counted.
  After a few of them, coming down, the walk state asks for a fall the way a press asks for a
  jump, and the same state is entered with no leap: he holds his falling pose, the stick steers
  him, and he lands on enemies as from a jump, with a landing of its own. A press as he lands is
  the second jump. A step down too small to count never gets that far.

Where it lives
--------------
Ten words of the walk and punch states are changed to call the new code (`asm/hooks.s`). The code
sits in stretches of the program that can never run (`ROOM`): this is a Wii U port, and the blocks
that served the GamePad are still there behind checks that are now always false; so are the
functions of two kinds of mission rule that no mission in the game uses. Ten bytes at the end of
the program's zeroed memory hold its state.

The patch fits one build of the game (`BUILD_ID`) and is named after it, so a loader will not apply
it to any other.
"""

from __future__ import annotations

import logging
import math
import struct

from pathlib import Path
from typing import TYPE_CHECKING

from mario import arm
from mario.errors import BuildError

if TYPE_CHECKING:
    from mario.romfs import Mod

logger = logging.getLogger(__name__)

BUILD_ID = "D467F5AD367BBEE88C51F8C468143CF23486A44B"  # the program of Pikmin 3 Deluxe 1.2.0
BASE = 0x60000000  # what the addresses here count from; a patch counts from 0x100 before the code's start
ASM = Path(__file__).parent / "asm"
SOURCES = ("punch", "jump", "hooks")  # assembled in this order, each seeing the labels of those before it
MARIO = 8  # what a leader's actor holds at +0x81c when its model is Alph's, which is now Mario's

# The clips the code plays, in the order it counts on: each is the one before plus one.
PUNCHES = ("m_punch1", "m_punch2", "m_kick")
JUMPS = ("m_jump1", "m_jump2", "m_jump3")
LANDINGS = ("m_land1", "m_land2", "m_land3")
FALLS = ("m_fall", "m_fland")  # the fall, and the landing from it
CLIPS = PUNCHES + JUMPS + LANDINGS + FALLS

# The game's own functions and data that the code calls or reads, by address.
GAME = {
    "anim_flags": 0x60E2B77C,  # (animation): the markers it passed this frame; bit 31 at its end
    "player_of": 0x60BEDB3C,  # (leader): 0 or 1, the player controlling it, or -1
    "shared_mode": 0x60B12218,  # whether pads are assigned another way just now ...
    "shared_pad": 0x60E0A874,  # ... and if so, whose pad it is
    "mode_object": 0x6189C330,
    "game_object": 0x61EA24BC,
    "pads": 0x6189DE18,  # player n's pad is 0x10 + n * 0x4f0 into this
    "steer": 0x60C4E3B8,  # (state, leader, 0): the walk state's stick-to-movement
    "walk_return": 0x60C50358,  # where the walk state goes after changing state itself
    "hold": 0x60A2D1A8,  # (slot, leader): take the leader's lock
    "release": 0x60A2D1C4,  # (slot): give it back
    "drop_pending": 0x609B9A78,  # (state machine): discard a state waiting to be entered
    "make_punch": 0x60C7C2F0,  # (state machine, 0x30): the punch state with no enemy named
}
# Where the new code goes: the first address and the size in words of each stretch that cannot run.
ROOM = {
    "room_update": (0x60C5ED70, 35),  # four blocks that served the Wii U GamePad
    "room_pad": (0x60C50A88, 33),
    "room_next": (0x60C50438, 29),
    "room_reach": (0x60C5CABC, 27),
    "room_jump": (0x60E422A8, 342),  # the functions of mission rule types 1 and 2, which no mission uses
    "room_fall": (0x60E375E4, 236),  # more of those functions
}
# The code's memory, past the program's last variable. The punches: moves made so far, and "a press
# is being taken". The jump: what was asked for (1 a jump, 2 a fall); what this state is, in the same
# numbers; which of the three jumps; 1 in the air, 2 landing; frames of that (two bytes; while he
# walks, his frames off the ground); frames left in which a press is the next jump up; and the
# body's "on the ground" byte while the steering borrows it.
PUNCH_STATE = 0x61EB7F00
JUMP_STATE = 0x61EB7F08
JUMP_BYTES = {"ASKED": 0, "IS_JUMP": 1, "WHICH": 2, "PHASE": 3, "FRAMES": 4, "WINDOW": 6, "GROUND": 7}

# A leader's punch: a sphere of `radius` centred `ahead` of him is searched for enemies, and the
# nearest within `furthest` of him is hit with `power`.
THEIRS = {"ahead": 4.0, "radius": 10.0, "furthest": 20.0, "power": 3.0}
POWER = 4.5  # the strength of a punch or kick of his
REACH = 1.5  # how many times as wide his search is, and how many times as far away a hit may land
STOMP = 9.0  # the strength of the blow when he lands on an enemy that cannot be squashed
STOMP_REACH = 12.0  # how near an enemy his feet must come, coming down, to land on it

# His own game's numbers: the speed he leaves the ground with for each jump and off an enemy, in its
# units a frame; its gravity; the speed the third jump needs, of his top running speed; his height.
SM64 = {
    "jumps": (42.0, 52.0, 69.0),
    "bounce": 30.0,
    "gravity": 4.0,
    "run_up": 20.0,
    "top_speed": 32.0,
    "height": 154.49,
}
GRAVITY = 400.0  # this game's, in units a second each second
TOP_SPEED = 80.0  # a leader's, as the number kept at leader + 0x828
SETTLE = 3  # frames after leaving the ground before touching it counts as landing
LONGEST = 512  # frames in the air after which a jump is given up, whatever is under him
CHAIN = 9  # frames after a landing in which a press is the next jump up
RUN_ON = 4  # frames of a landing before the stick cuts it short
FALL_AFTER = 5  # frames off the ground, walking, before it is a fall, so that a small step down is not one
FALL_SPEED = 30.0  # and how fast he must be coming down by then, in units a second


def bits(value: float) -> int:
    """A number as the 32 bits the game keeps it in."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def his_hit() -> dict[str, float]:
    """
    What a punch or kick of his reaches, and how hard it lands.

    The sphere searched grows forward and to the sides; its back edge stays where a leader's is, so
    that he does not start hitting what is behind him.

    Returns:
        The same four numbers as `THEIRS`.
    """
    radius = THEIRS["radius"] * REACH
    return {
        "power": POWER,
        "radius": radius,
        "ahead": radius - (THEIRS["radius"] - THEIRS["ahead"]),
        "furthest": THEIRS["furthest"] * REACH,
    }


def leap(speed: float, height: float) -> tuple[float, float]:
    """
    A jump of his own game's, for a Mario `height` tall under this game's gravity.

    Returns:
        The upward speed that takes him as high, for his size, as leaving the ground at `speed` does
        there; and how high that is.
    """
    apex = speed * speed / (2.0 * SM64["gravity"]) * (height / SM64["height"])
    return math.sqrt(2.0 * GRAVITY * apex), apex


def fall_drop() -> float:
    """How far he has dropped by the frame a fall begins: the smallest step down that shows one."""
    return 0.5 * GRAVITY * (FALL_AFTER / 30.0) ** 2


def symbols(first_clip: int, height: float) -> dict[str, int]:
    """Every name the assembly uses that is not one of its own labels."""
    hit = his_hit()
    floats = {
        "RADIUS_THEIRS": THEIRS["radius"],
        "FURTHEST_THEIRS": THEIRS["furthest"],
        "POWER_THEIRS": THEIRS["power"],
        "RADIUS": hit["radius"],
        "AHEAD": hit["ahead"],
        "FURTHEST": hit["furthest"],
        "POWER": hit["power"],
        "SPEED_1": leap(SM64["jumps"][0], height)[0],
        "SPEED_2": leap(SM64["jumps"][1], height)[0],
        "SPEED_3": leap(SM64["jumps"][2], height)[0],
        "BOUNCE": leap(SM64["bounce"], height)[0],
        "RUN_UP": TOP_SPEED * SM64["run_up"] / SM64["top_speed"],
        "STOMP_RADIUS": STOMP_REACH,
        "STOMP_FURTHEST": STOMP_REACH,
        "STOMP_POWER": STOMP,
        "FALL_SPEED": -FALL_SPEED,  # as his upward speed
    }
    return {
        **GAME,
        **{name: start for name, (start, _) in ROOM.items()},
        **JUMP_BYTES,
        **{name: bits(value) for name, value in floats.items()},
        "MARIO": MARIO,
        "PUNCH_STATE": PUNCH_STATE,
        "JUMP_STATE": JUMP_STATE,
        "LAST_MOVE": len(PUNCHES) - 1,
        "PUNCH_CLIP": first_clip,
        "JUMP_CLIP": first_clip + len(PUNCHES),
        "LAND_CLIP": first_clip + len(PUNCHES) + len(JUMPS),
        "FALL_CLIP": first_clip + len(PUNCHES) + len(JUMPS) + len(LANDINGS),
        "SETTLE": SETTLE,
        "LONGEST": LONGEST,
        "CHAIN": CHAIN,
        "RUN_ON": RUN_ON,
        "FALL_AFTER": FALL_AFTER,
    }


def program(first_clip: int, height: float) -> dict[int, int]:
    """
    Assemble the code.

    Args:
        first_clip: The number of his first punch in the game's list of clips. The rest of `CLIPS`
            must follow it, one after another.
        height: How tall he is in this game, which sets how high he jumps.

    Returns:
        The words to write, by address.

    Raises:
        BuildError: If the code does not fit the room set aside for it.
    """
    if not 0 < first_clip <= 0x100 - len(CLIPS):
        raise BuildError(f"{first_clip:#x} is not a clip number the code can count from: each must fit a byte")
    names = symbols(first_clip, height)
    words: dict[int, int] = {}
    for source in SOURCES:
        text = (ASM / f"{source}.s").read_text(encoding="utf-8")
        try:
            new, labels = arm.assemble(text, names)
        except arm.AsmError as error:
            raise BuildError(f"asm/{source}.s, {error}") from error
        if words.keys() & new.keys():
            raise BuildError(f"asm/{source}.s places a word where another file already has one")
        if source != "hooks":
            _check_room(source, new)
        words.update(new)
        names.update(labels)
    return words


def _check_room(source: str, words: dict[int, int]) -> None:
    """Refuse code that is not one unbroken run from the start of each stretch it uses."""
    left = set(words)
    for start, size in ROOM.values():
        inside = sorted(a for a in left if start <= a < start + 4 * size)
        if inside != list(range(start, start + 4 * len(inside), 4)):
            raise BuildError(f"asm/{source}.s leaves a hole in the room at {start:#x}")
        left.difference_update(inside)
    if left:
        raise BuildError(f"asm/{source}.s has {len(left)} words outside the room set aside for the code")


def ips(words: dict[int, int]) -> bytes:
    """
    An IPS patch of the program holding `words`.

    Each run of words is one record at its offset in the program file as a loader sees it, which is
    0x100 past the start of the code.

    Returns:
        The patch file's bytes.

    Raises:
        BuildError: If a run cannot be held in the format.
    """
    patch = bytearray(b"PATCH")
    run: list[int] = []
    for address in [*sorted(words), None]:
        if run and address != run[-1] + 4:
            offset = run[0] - BASE + 0x100
            data = b"".join(struct.pack("<I", words[a]) for a in run)
            if offset >= 0x1000000 or offset == int.from_bytes(b"EOF", "big") or len(data) > 0xFFFF:
                raise BuildError(f"the run of code at {run[0]:#x} does not fit the patch format")
            patch += offset.to_bytes(3, "big") + len(data).to_bytes(2, "big") + data
            run = []
        if address is not None:
            run.append(address)
    return bytes(patch + b"EOF")


def read_ips(patch: bytes) -> dict[int, int]:
    """
    The words of a patch, read the way a loader reads it: the inverse of `ips`.

    Returns:
        The words by address.

    Raises:
        BuildError: If it is not an IPS patch of whole words.
    """
    if patch[:5] != b"PATCH":
        raise BuildError("not an IPS patch")
    at, words = 5, {}
    while patch[at : at + 3] != b"EOF":
        offset = int.from_bytes(patch[at : at + 3], "big")
        size = int.from_bytes(patch[at + 3 : at + 5], "big")
        if not size or size % 4 or at + 5 + size > len(patch):
            raise BuildError(f"a record of {size} bytes is not whole words")
        for k in range(0, size, 4):
            words[offset - 0x100 + BASE + k] = struct.unpack_from("<I", patch, at + 5 + k)[0]
        at += 5 + size
    if at + 3 != len(patch):
        raise BuildError("there are bytes after the end of the patch")
    return words


def in_room(address: int) -> bool:
    """Whether an address is in one of the stretches set aside for the code."""
    return any(start <= address < start + 4 * size for start, size in ROOM.values())


def build(mod: Mod, first_clip: int, height: float) -> None:
    """Write the patch into the mod folder. `first_clip` and `height` are as for `program`."""
    words = program(first_clip, height)
    hit = his_hit()
    heights = [leap(speed, height)[1] for speed in SM64["jumps"]]
    code = sum(in_room(address) for address in words)
    logger.info(
        "his moves: %d words of code in %d unused stretches of the program, called from %d changed words",
        code,
        len(ROOM),
        len(words) - code,
    )
    logger.info(
        "  a punch: strength %g (a leader's is %g), reaching %g ahead of him (a leader's %g)",
        hit["power"],
        THEIRS["power"],
        hit["radius"] + hit["ahead"],
        THEIRS["radius"] + THEIRS["ahead"],
    )
    logger.info(
        "  his jumps: %s units high (he is %g tall); landing on an enemy is a blow of strength %g",
        ", ".join(f"{h:.1f}" for h in heights),
        height,
        STOMP,
    )
    logger.info(
        "  his fall: %d frames after he walks off an edge, by when he has dropped %.1f units",
        FALL_AFTER,
        fall_drop(),
    )
    mod.write_patch(BUILD_ID, ips(words))
