"""
Where the files are: the game's own, read from an extracted RomFS, and the mod folder being written.

A mod folder mirrors the game's layout. A loader that finds `romfs/CMCmn/player/pikminAll.szs` in
it hands the game that file in place of its own, and applies `exefs/<build id>.ips` to the program
whose build ID that is.
"""

from __future__ import annotations

from pathlib import Path

from mario.errors import BuildError
from mario.formats import sarc, yaz0

COMMON = "CMCmn"  # the folder of the RomFS that every file this mod touches is in
LANDMARK = "player/pikminAll.szs"


class RomFs:
    """The game's files, extracted from your own copy. Nothing is ever written here."""

    def __init__(self, path: str | Path):
        """
        Find the game's files at `path`: the RomFS folder itself, or a folder holding it as `romfs`.

        Raises:
            BuildError: If the game's files are not there.
        """
        path = Path(path)
        for root in (path / COMMON, path / "romfs" / COMMON, path):
            if (root / LANDMARK).is_file():
                self.root = root
                return
        raise BuildError(f"{path} does not look like Pikmin 3 Deluxe's RomFS: there is no {COMMON}/{LANDMARK} in it")

    def read(self, name: str) -> bytes:
        """The bytes of one file, named from inside `CMCmn/` with forward slashes."""
        file = self.root / name
        if not file.is_file():
            raise BuildError(f"the RomFS has no {COMMON}/{name}")
        return file.read_bytes()

    def names(self, folder: str, suffix: str) -> list[str]:
        """The files of one folder that end in `suffix`, named as `read` takes them."""
        return sorted(f"{folder}/{file.name}" for file in (self.root / folder).glob(f"*{suffix}"))


class Mod:
    """The mod folder being written."""

    def __init__(self, path: str | Path):
        self.root = Path(path)

    def write(self, name: str, data: bytes) -> Path:
        """Write a file the game will load in place of its own `CMCmn/<name>`."""
        return self._write(self.root / "romfs" / COMMON / name, data)

    def write_patch(self, build_id: str, data: bytes) -> Path:
        """Write an IPS patch of the program with this build ID."""
        return self._write(self.root / "exefs" / f"{build_id}.ips", data)

    @staticmethod
    def _write(file: Path, data: bytes) -> Path:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        return file


def repack(stock: bytes, changes: dict[str, bytes]) -> bytes:
    """
    A `.szs` archive of the game's with some of the files inside it replaced.

    The archive is laid out and compressed again, under the stock file's header: the game allocates
    the unpacked archive with the alignment written there.

    Returns:
        The new `.szs` file.

    Raises:
        BuildError: If the stock archive is not laid out the way `sarc.rebuild` lays one out, in
            which case a rebuilt one could not be trusted.
    """
    plain = yaz0.decompress(stock)
    if sarc.rebuild(plain, {}) != plain:
        raise BuildError("the archive is not laid out the way this tool lays one out")
    return yaz0.compress(sarc.rebuild(plain, changes), yaz0.alignment(stock))
