"""
SARC, the archive inside most of the game's `.szs` files.

Pikmin 3 Deluxe keeps almost everything in `.szs` files: a Yaz0 stream (`mario.formats.yaz0`) around
a little-endian SARC archive. A few `.szs` files hold a single compressed file instead of an archive.

An archive is three tables and then the files' bytes::

    SARC  0x14 bytes: its own length (u16 at 4), the byte order mark (FF FE for little-endian, at 6),
          the size of the archive (u32 at 8) and where the files' bytes start (u32 at 0xC)
    SFAT  12 bytes holding the count of files (u16 at 6) and the multiplier of the name hash (u32
          at 8), then a 16-byte node for each file: the hash of its name, its attributes, and the
          start and end of its bytes
    SFNT  8 bytes, then the names, each ending in a zero byte and starting on a multiple of four

A node's start and end count from where the files' bytes start. Attributes with the top byte set
mean the file has a name, found in the name table at four times the low 24 bits. A file without
one is known only by its hash and is listed here as `#` and the hash in hex.

`Sarc` reads an archive and `rebuild` gives a copy of one with files changed. `rebuild` works on
decompressed bytes: pack the result with `yaz0.compress`.
"""

from __future__ import annotations

import struct

from mario.formats.yaz0 import decompress

NODE = 16  # the size of one file's entry in the SFAT table


class Sarc:
    """
    An archive read from its bytes, compressed or not.

    `data` is the decompressed bytes and `entries` lists (name, start, end) for each file in them.
    An archive also has `endian` (the `struct` prefix for its numbers), `data_start` (where the
    files' bytes start) and `nodes` (where the first file's SFAT node is).

    Bytes that are not an archive are taken as one file called `name`: `is_archive` is False,
    `entries` is empty, and `files()` and `sizes()` give that one file.
    """

    def __init__(self, data: bytes | bytearray, name: str = "file") -> None:
        self.data = plain = decompress(bytes(data))
        self.name = name
        self.is_archive = plain[:4] == b"SARC"
        self.entries: list[tuple[str, int, int]] = []
        if not self.is_archive:
            return
        self.endian = endian = "<" if plain[6:8] == b"\xff\xfe" else ">"
        sfat, _, _, self.data_start = struct.unpack_from(f"{endian}HHII", plain, 4)
        if plain[sfat : sfat + 4] != b"SFAT":
            raise ValueError("no SFAT")
        count = struct.unpack_from(f"{endian}H", plain, sfat + 6)[0]
        self.nodes = sfat + 12
        sfnt = self.nodes + NODE * count
        if plain[sfnt : sfnt + 4] != b"SFNT":
            raise ValueError("no SFNT")
        names = sfnt + 8
        for index in range(count):
            name_hash, attributes, start, end = struct.unpack_from(f"{endian}IIII", plain, self.nodes + NODE * index)
            if attributes >> 24:
                at = names + (attributes & 0xFFFFFF) * 4
                entry = plain[at : plain.index(b"\0", at)].decode("utf-8", errors="replace")
            else:
                entry = f"#{name_hash:08x}"
            self.entries.append((entry, self.data_start + start, self.data_start + end))

    def files(self) -> dict[str, bytes]:
        """The bytes of every file, by name."""
        if not self.is_archive:
            return {self.name: self.data}
        return {name: self.data[start:end] for name, start, end in self.entries}

    def sizes(self) -> dict[str, int]:
        """The length of every file, by name."""
        if not self.is_archive:
            return {self.name: len(self.data)}
        return {name: end - start for name, start, end in self.entries}

    def span(self, name: str) -> tuple[int, int]:
        """
        Find one file inside the decompressed archive, for editing it where it lies.

        Returns:
            Its (start, end) in `data`.

        Raises:
            KeyError: If the archive has no file of that name.
        """
        for entry, start, end in self.entries:
            if entry == name:
                return start, end
        raise KeyError(name)


def rebuild(plain: bytes | bytearray, changes: dict[str, bytes]) -> bytes:
    """
    Replace some files of a decompressed archive with others of any length: {name: bytes}.

    The table of files and their names is kept; only the data is laid out again, in the order it
    was in, each file starting on the boundary the stock layout keeps (the largest power of two that
    every file's offset is a multiple of), with zeros between. A change of nothing gives the same
    bytes when the stock archive pads that way, which the caller should check.

    Returns:
        The new archive, decompressed.

    Raises:
        ValueError: If `plain` is not an archive.
        KeyError: If `changes` names a file the archive does not have.
    """
    arc = Sarc(plain)
    if not arc.is_archive:
        raise ValueError("not an archive")
    known = arc.sizes()
    missing = [name for name in changes if name not in known]
    if missing:
        raise KeyError(f"archive has no file named {', '.join(missing)}")
    align = 0x2000
    while any(start % align for _, start, _ in arc.entries):
        align >>= 1
    out = bytearray(plain[: arc.data_start])
    for index in sorted(range(len(arc.entries)), key=lambda index: arc.entries[index][1]):
        name, start, end = arc.entries[index]
        body = changes.get(name, plain[start:end])
        out += bytes(-len(out) % align)
        at = len(out) - arc.data_start
        struct.pack_into(f"{arc.endian}II", out, arc.nodes + NODE * index + 8, at, at + len(body))
        out += body
    struct.pack_into(f"{arc.endian}I", out, 8, len(out))
    return bytes(out)
