"""
Turn Alph into Mario.

Builds a Pikmin 3 Deluxe mod from two things only you can supply: your own Super Mario 64 ROM and
the files of your own copy of Pikmin 3 Deluxe. Nothing of either game is kept in this repository.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from mario import icons, model, moves, voice
from mario.romfs import Mod, RomFs
from mario.sm64.rom import Rom

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def build(rom: str | Path, romfs: str | Path, out: str | Path) -> None:
    """
    Build the whole mod.

    Args:
        rom: A Super Mario 64 ROM, in any of the three byte orders.
        romfs: Pikmin 3 Deluxe's extracted RomFS.
        out: The mod folder to write.
    """
    rom, romfs, mod = Rom(rom), RomFs(romfs), Mod(out)
    first_clip = model.build(rom, romfs, mod)
    voice.build(rom, romfs, mod)
    icons.build(rom, romfs, mod)
    moves.build(mod, first_clip, model.HEIGHT)
    logger.info("done: the mod is in %s", mod.root)
