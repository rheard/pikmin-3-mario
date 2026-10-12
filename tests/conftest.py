"""
Fixtures for the tests that need the two games.

Those tests are skipped unless `SM64_ROM` names a Super Mario 64 ROM and `PIKMIN3_ROMFS` names
Pikmin 3 Deluxe's extracted RomFS. With both set, the mod is built once for the whole run, which
takes a while; `MARIO_MOD` may name a folder built earlier to read that back instead.
"""

import os

from pathlib import Path

import pytest

import mario

from mario.romfs import RomFs
from mario.sm64.rom import Rom

NEEDED = {"SM64_ROM": "a Super Mario 64 ROM", "PIKMIN3_ROMFS": "Pikmin 3 Deluxe's extracted RomFS"}
BUILT = "MARIO_MOD"


def game_file(variable: str) -> str:
    """The path an environment variable names, or skip the test."""
    path = os.environ.get(variable)
    if not path:
        pytest.skip(f"set {variable} to {NEEDED[variable]} to run this")
    return path


@pytest.fixture(scope="session")
def rom() -> Rom:
    """Your Super Mario 64 ROM."""
    return Rom(game_file("SM64_ROM"))


@pytest.fixture(scope="session")
def romfs() -> RomFs:
    """The game's own files."""
    return RomFs(game_file("PIKMIN3_ROMFS"))


@pytest.fixture(scope="session")
def mod(tmp_path_factory: pytest.TempPathFactory) -> RomFs:
    """The built mod, read the way the game's own files are."""
    rom, romfs = game_file("SM64_ROM"), game_file("PIKMIN3_ROMFS")
    built = os.environ.get(BUILT)
    if not built:
        built = tmp_path_factory.mktemp("mod") / "Mario"
        mario.build(rom, romfs, built)
    return RomFs(Path(built))
