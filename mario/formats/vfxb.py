"""
VFXB, the particle-effect file (an nn::vfx `.ptcl`): the game's is `effect_common.ptcl`.

The file starts with the letters `VFXB` and from 0x40 on is a tree of sections. Each section opens
with a little-endian header, whose offsets count from the section's own start::

    0x00  four letters naming the kind of section
    0x08  u32  offset of its first child (0xFFFFFFFF: it has none)
    0x0C  u32  offset of the next section beside it (0 or 0xFFFFFFFF: it is the last)
    0x14  u32  offset of its binary block
    0x1C  u16  how many children it has

The effects are in the `ESTA` section. Its children are the emitter sets, and a set's `EMTR`
children are its emitters, each of which draws one kind of particle. A set's or an emitter's name is
0x10 bytes into its binary block.

A set's place in the list is its id, the number the game's code passes when it plays an effect
(`FUN_608e44b0`, at +0x10 of its parameters): the whistle is 0x369, the punch 0x23c. The ids are
written into the code, so a set added to the file could only go at the end. Every set in this file
is loaded when the game starts. The file is about 52 MB once decompressed.

`emitter_blocks` lists the sets and where each emitter's binary block is; `hide_emitter` makes one
emitter draw nothing.
"""

from __future__ import annotations

import struct

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

NONE = 0xFFFFFFFF  # an offset that leads nowhere

# Three fields of an emitter, as offsets into its binary block. They are for this file's version of
# the format (0x16). In the leaders' antenna lights the two colours are each leader's own (Alph
# blue, Brittany pink, Charlie green) and the size is 9 for a glow, 500 for an explosion's light.
# Values that make sense sit at these offsets in 3,171 of the file's 3,180 emitters, so look at what
# an emitter holds there before writing over it.
EMITTER_COLOR0 = 0x9A0  # r, g, b, a: a multiplier on the particle's colour
EMITTER_COLOR1 = 0x9B0  # the second colour, same form
EMITTER_SCALE = 0x9C0  # x, y, z: the particle's base size


def header(d: bytes | bytearray, off: int) -> dict:
    """The header of the section at `off`: its name, where it is, and the offsets and count it carries."""
    _, child, following, _, binary, _, count, _ = struct.unpack_from("<IIIIIIHH", d, off + 4)
    return {
        "magic": d[off : off + 4].decode("ascii", "replace"),
        "off": off,
        "child": child,
        "next": following,
        "bin": binary,
        "count": count,
    }


def siblings(d: bytes | bytearray, off: int) -> Iterator[dict]:
    """The header of the section at `off` and of each section beside it, in order."""
    while True:
        section = header(d, off)
        yield section
        if section["next"] in {0, NONE}:
            return
        off += section["next"]


def cstr(d: bytes | bytearray, off: int, n: int = 64) -> str:
    """The text at `off`, up to its zero byte or `n` characters."""
    text = d[off : off + n]
    end = text.find(b"\0")
    return text[: end if end >= 0 else n].decode("ascii", "replace")


def emitter_blocks(d: bytes | bytearray) -> list[tuple[int, str, list[tuple[str, int]]]]:
    """[(set id, set name, [(emitter name, offset of its binary block)])] in file order."""
    out = []
    for top in siblings(d, 0x40):
        if top["magic"] != "ESTA":
            continue
        for index, group in enumerate(siblings(d, top["off"] + top["child"])):
            found = []
            if group["child"] != NONE and group["count"]:
                for emitter in siblings(d, group["off"] + group["child"]):
                    if emitter["magic"] == "EMTR":
                        block = emitter["off"] + emitter["bin"]
                        found.append((cstr(d, block + 0x10), block))
            out.append((index, cstr(d, group["off"] + group["bin"] + 0x10), found))
    return out


def hide_emitter(d: bytearray, block: int) -> tuple[int, int]:
    """
    Make one emitter draw nothing, in place: no colour, no alpha, and particles too small to see.

    The size is left a hair above zero in case anything divides by it.

    Returns:
        (start, end) of the bytes written.
    """
    struct.pack_into("<8f", d, block + EMITTER_COLOR0, *([0.0] * 8))  # both colours
    struct.pack_into("<3f", d, block + EMITTER_SCALE, 1e-4, 1e-4, 1e-4)
    return block + EMITTER_COLOR0, block + EMITTER_SCALE + 12
