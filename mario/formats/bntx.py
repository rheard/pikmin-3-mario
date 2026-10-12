"""
BNTX, the Switch's texture file: on its own in a layout archive, or embedded in a model (BFRES).

A BNTX starts with the letters `BNTX`. At 0x20 is an `NX  ` block holding the count of textures and
a pointer to a table of pointers, one to each texture's `BRTI` record. Pointers are 64-bit and count
from the start of the BNTX. Of a BRTI, this module reads::

    0x16  u16  how many mip levels
    0x1C  u32  the format: the codec in the second byte, the kind of number in the lowest
    0x24  u32  width, and at 0x28 height
    0x34  u32  log2 of the block height of the layout described below
    0x50  u32  the size of its data, all mips together
    0x60  u64  pointer to its name (a u16 length, then the characters)
    0x70  u64  pointer to a table of pointers, one to each mip's data

The data is 4x4 blocks compressed as BC1, BC3, BC4 or BC5: the four codecs handled here, unsigned,
signed and sRGB. The blocks are not stored row by row but in the Tegra's block-linear order: the
surface is cut into tiles of 512 bytes (64 across, 8 rows down), and the tiles are stacked `block
height` deep before the layout moves across (`_address`). The mips follow one another, and the block
height halves with each mip too short to fill it.

`Bntx(buffer)` finds the BNTX inside any bytes, a bare `.bntx` or a whole `.bfres`, and the offsets
its textures carry are into that buffer. `decode` reads a mip as a picture. `replace` encodes a
picture into a texture in place and `fill` does the same with one flat colour: same size, same
format, same mip count, and every byte outside that texture's data untouched. Nothing in the file
moves, which is what makes them safe to use on a file full of pointers.

A picture is rows of (r, g, b, a) pixels, each channel 0 to 255.
"""

from __future__ import annotations

import struct

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

Pixel = tuple[int, ...]  # (r, g, b, a), or (r, g, b) where coverage does not matter
Rows = list[list[Pixel]]

FORMATS = {0x1A: ("BC1", 8), 0x1C: ("BC3", 16), 0x1D: ("BC4", 8), 0x1E: ("BC5", 16)}  # codec: (name, block bytes)
TYPES = {1: "UNORM", 2: "SNORM", 6: "SRGB"}
TILE_WIDTH, TILE_ROWS = 64, 8  # a tile of the block-linear layout: bytes across, rows down

# The directions along which `encode_bc1_search` looks for a block's furthest-apart colours.
_AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1))


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pow2_up(value: int) -> int:
    power = 1
    while power < value:
        power *= 2
    return power


def _u16(d: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<H", d, offset)[0]


def _u32(d: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<I", d, offset)[0]


def _u64(d: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from("<Q", d, offset)[0]


# -- the file ---------------------------------------------------------------------------------------


class Texture:
    """One texture's record: what it is, and where its data lies in the buffer it was read from."""

    def __init__(self, d: bytes | bytearray, base: int, record: int) -> None:
        if d[record : record + 4] != b"BRTI":
            raise ValueError(f"no BRTI at {record:#x}")
        name_at = base + _u64(d, record + 0x60)
        self.name = bytes(d[name_at + 2 : name_at + 2 + _u16(d, name_at)]).decode()
        self.mips = _u16(d, record + 0x16)
        self.format = _u32(d, record + 0x1C)
        self.width, self.height = _u32(d, record + 0x24), _u32(d, record + 0x28)
        self.block_height_log2 = _u32(d, record + 0x34)
        self.size = _u32(d, record + 0x50)
        table = base + _u64(d, record + 0x70)
        self.offsets = [base + _u64(d, table + 8 * level) for level in range(self.mips)]  # into the buffer given

    @property
    def codec(self) -> str:
        """The name of its block compression, or the format's number in hex if it is not one handled here."""
        return FORMATS.get(self.format >> 8, (f"{self.format >> 8:#x}", 0))[0]

    @property
    def signed(self) -> bool:
        """Whether its values are signed, as a normal map's often are."""
        return (self.format & 0xFF) == 2

    def describe(self) -> str:
        """Its format in words, such as `BC1 SRGB`."""
        return f"{self.codec} {TYPES.get(self.format & 0xFF, '?')}"

    def mip_size(self, level: int) -> tuple[int, int]:
        """(width, height) of one mip level, in pixels."""
        return max(1, self.width >> level), max(1, self.height >> level)

    def mip_span(self, level: int) -> tuple[int, int]:
        """(start, end) of one mip level's bytes in the buffer."""
        end = self.offsets[level + 1] if level + 1 < self.mips else self.offsets[0] + self.size
        return self.offsets[level], end

    def mip_block_height(self, level: int) -> int:
        """
        Block height (log2) the layout uses for a mip: it shrinks as the mips get short.

        A stack of the texture's own block height is so many rows of blocks tall. The height halves
        once for this level, and once for each level above it, whose own rows of blocks (rounded up
        to a power of two) would not fill that.

        Returns:
            The log2 of the block height, never below zero.
        """
        full = TILE_ROWS << self.block_height_log2
        short = sum(_pow2_up(_div_up(self.mip_size(above)[1], 4)) < full for above in range(level + 1))
        return max(0, self.block_height_log2 - short)


class Bntx:
    """The textures of the BNTX found in a buffer, by name and in the file's order."""

    def __init__(self, d: bytes | bytearray) -> None:
        self.base = base = d.find(b"BNTX")
        if base < 0:
            raise ValueError("no BNTX in this file")
        nx = base + 0x20
        if d[nx : nx + 4] != b"NX  ":
            raise ValueError("BNTX without an NX header")
        count = _u32(d, nx + 4)
        pointers = base + _u64(d, nx + 8)
        self.textures: dict[str, Texture] = {}
        for index in range(count):
            texture = Texture(d, base, base + _u64(d, pointers + 8 * index))
            self.textures[texture.name] = texture

    def __getitem__(self, name: str) -> Texture:
        return self.textures[name]


# -- block-linear layout ----------------------------------------------------------------------------


def _address(x: int, y: int, width: int, bpp: int, block_height: int) -> int:
    """
    Byte address of block (x, y) in a block-linear surface `width` blocks wide.

    Tiles are numbered down each stack of `block_height`, then stack by stack across the surface,
    then on to the next row of stacks. Inside a tile the bits of the byte column and of the row are
    woven together: column bits 0 to 3, row bit 0, column bit 4, row bits 1 and 2, column bit 5.

    Returns:
        The offset of the block's first byte from the start of the surface.
    """
    column = x * bpp
    stacks_wide = _div_up(width * bpp, TILE_WIDTH)
    stack_rows = TILE_ROWS * block_height
    stack = (y // stack_rows) * stacks_wide + column // TILE_WIDTH
    tile = stack * block_height + (y % stack_rows) // TILE_ROWS
    inside = (column & 0xF) | (y & 1) << 4 | (column & 0x10) << 1 | (y & 6) << 5 | (column & 0x20) << 3
    return tile * TILE_WIDTH * TILE_ROWS + inside


def _mip_blocks(tex: Texture, level: int) -> tuple[int, int]:
    """How many blocks wide and high a mip level is."""
    width, height = tex.mip_size(level)
    return _div_up(width, 4), _div_up(height, 4)


def _block_offsets(tex: Texture, level: int) -> Iterator[tuple[int, int]]:
    """For each block of a mip level, row by row: its offset in row order and its offset in the buffer."""
    bpp = FORMATS[tex.format >> 8][1]
    wide, high = _mip_blocks(tex, level)
    block_height = 1 << tex.mip_block_height(level)
    start, end = tex.mip_span(level)
    for y in range(high):
        for x in range(wide):
            at = start + _address(x, y, wide, bpp, block_height)
            if at + bpp > end:
                raise ValueError(f"{tex.name} mip {level}: block ({x}, {y}) falls outside the mip")
            yield (y * wide + x) * bpp, at


def read_mip(d: bytes | bytearray, tex: Texture, level: int = 0) -> bytes:
    """One mip level's compressed blocks in plain row order."""
    bpp = FORMATS[tex.format >> 8][1]
    wide, high = _mip_blocks(tex, level)
    out = bytearray(wide * high * bpp)
    for row_order, at in _block_offsets(tex, level):
        out[row_order : row_order + bpp] = d[at : at + bpp]
    return bytes(out)


def write_mip(d: bytearray, tex: Texture, level: int, blocks: bytes | bytearray) -> None:
    """
    Store row-ordered compressed blocks as one mip level, in place.

    Raises:
        ValueError: If `blocks` is not the size of the mip.
    """
    bpp = FORMATS[tex.format >> 8][1]
    wide, high = _mip_blocks(tex, level)
    if len(blocks) != wide * high * bpp:
        raise ValueError(f"{tex.name} mip {level}: {len(blocks)} bytes of blocks, expected {wide * high * bpp}")
    for row_order, at in _block_offsets(tex, level):
        d[at : at + bpp] = blocks[row_order : row_order + bpp]


# -- block codecs -----------------------------------------------------------------------------------


def _to565(c: Pixel) -> int:
    return ((c[0] * 31 + 127) // 255) << 11 | ((c[1] * 63 + 127) // 255) << 5 | ((c[2] * 31 + 127) // 255)


def _from565(value: int) -> Pixel:
    r, g, b = (value >> 11) & 31, (value >> 5) & 63, value & 31
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def _bc1_palette(c0: int, c1: int, *, four: bool) -> list[Pixel]:
    """The four colours a block's two end colours stand for; without `four`, their order picks the mode."""
    a, b = _from565(c0), _from565(c1)
    if four or c0 > c1:
        return [a, b, tuple((2 * a[i] + b[i]) // 3 for i in range(3)), tuple((a[i] + 2 * b[i]) // 3 for i in range(3))]
    return [a, b, tuple((a[i] + b[i]) // 2 for i in range(3)), (0, 0, 0)]


def _nearest(p: Pixel, pal: list[Pixel]) -> int:
    """Which of a palette's four colours is closest to a pixel."""
    return min(range(4), key=lambda k: (p[0] - pal[k][0]) ** 2 + (p[1] - pal[k][1]) ** 2 + (p[2] - pal[k][2]) ** 2)


def encode_bc1(px: list[Pixel]) -> bytes:
    """16 (r, g, b[, a]) pixels -> an 8-byte colour block, always in four-colour mode."""
    first = px[0][:3]
    if all(p[:3] == first for p in px):
        c = _to565(first)
        return struct.pack("<HHI", c, c, 0)
    # Endpoints: the two pixels furthest apart along the axis of greatest spread.
    lo = [min(p[i] for p in px) for i in range(3)]
    hi = [max(p[i] for p in px) for i in range(3)]
    axis = [hi[i] - lo[i] for i in range(3)]
    proj = [p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2] for p in px]
    c0, c1 = _to565(px[proj.index(max(proj))]), _to565(px[proj.index(min(proj))])
    if c0 == c1:
        return struct.pack("<HHI", c0, c1, 0)
    if c0 < c1:
        c0, c1 = c1, c0
    pal = _bc1_palette(c0, c1, four=True)
    bits = 0
    for i, p in enumerate(px):
        bits |= _nearest(p, pal) << (2 * i)
    return struct.pack("<HHI", c0, c1, bits)


def encode_bc1_search(px: list[Pixel]) -> bytes:
    """
    16 (r, g, b[, a]) pixels -> an 8-byte colour block, by trying many pairs of end colours.

    `encode_bc1` takes the two pixels furthest apart and is done, which is fine for skin and cloth
    and poor for a small picture with red, blue, black and skin in one block. Here every pair of the
    block's own colours is tried when there are eight or fewer, and otherwise every pair of the
    colours at the ends of its spread along seven directions; the pair whose four-colour palette
    misses the pixels by least is kept. Pixels with no coverage (alpha 0) do not count towards the
    choice, only take the nearest colour afterwards.

    Returns:
        The block.
    """
    count = {}
    for p in px:
        if len(p) < 4 or p[3]:
            count[p[:3]] = count.get(p[:3], 0) + 1
    if not count:
        count = {p[:3]: 1 for p in px}
    colours = list(count)
    ends = sorted({_to565(c) for c in colours})
    if len({p[:3] for p in px}) == 1:
        return struct.pack("<HHI", ends[0], ends[0], 0)
    if len(ends) > 8:
        ends = set()
        for axis in _AXES:
            proj = [c[0] * axis[0] + c[1] * axis[1] + c[2] * axis[2] for c in colours]
            ends.add(_to565(colours[proj.index(max(proj))]))
            ends.add(_to565(colours[proj.index(min(proj))]))
        ends = sorted(ends)
    weighted = [(c[0], c[1], c[2], n) for c, n in count.items()]
    pairs = [(c0, c1) for i, c1 in enumerate(ends) for c0 in ends[i + 1 :]]  # c0 > c1: the four-colour mode
    quick = struct.unpack_from("<HH", encode_bc1(px))  # so this is never the worse of the two
    pairs += [(max(quick), min(quick)), (ends[0], ends[0])]
    best, best_pair = None, None
    for c0, c1 in pairs:
        pal = _bc1_palette(c0, c1, four=True)
        err = 0
        for r, g, b, n in weighted:
            err += n * min((r - q[0]) ** 2 + (g - q[1]) ** 2 + (b - q[2]) ** 2 for q in pal)
            if best is not None and err >= best:
                break
        else:
            best, best_pair = err, (c0, c1)
    c0, c1 = best_pair
    pal = _bc1_palette(c0, c1, four=True)
    bits = 0
    for i, p in enumerate(px):
        bits |= _nearest(p, pal) << (2 * i)
    return struct.pack("<HHI", c0, c1, bits)


def _bc4_palette(a0: int, a1: int) -> list[float]:
    """The values a block's two end values stand for, less the two fixed extremes of the six-value mode."""
    if a0 > a1:
        return [a0, a1] + [((7 - i) * a0 + i * a1) / 7 for i in range(1, 7)]
    return [a0, a1] + [((5 - i) * a0 + i * a1) / 5 for i in range(1, 5)]


def _closest(value: int, pal: list[float]) -> int:
    """Which of a palette's eight values is closest to a value."""
    return min(range(8), key=lambda k: abs(value - pal[k]))


def encode_bc4(vals: list[int], *, signed: bool = False) -> bytes:
    """16 single-channel values (0-255, or -127..127 when signed) -> an 8-byte block."""
    hi, lo = max(vals), min(vals)
    if hi == lo:
        return struct.pack("<bb" if signed else "<BB", hi, hi) + bytes(6)
    pal = _bc4_palette(hi, lo)
    bits = 0
    for i, value in enumerate(vals):
        bits |= _closest(value, pal) << (3 * i)
    return struct.pack("<bb" if signed else "<BB", hi, lo) + bits.to_bytes(6, "little")


def decode_bc4(blk: bytes | bytearray, *, signed: bool = False) -> list[float]:
    """An 8-byte block -> 16 single-channel values (0-255, or -127..127 when signed)."""
    a0, a1 = struct.unpack_from("<bb" if signed else "<BB", blk)
    pal = _bc4_palette(a0, a1)
    if a0 <= a1:
        pal += [-127 if signed else 0, 127 if signed else 255]
    bits = int.from_bytes(blk[2:8], "little")
    return [pal[(bits >> (3 * i)) & 7] for i in range(16)]


def _channel(px: list[Pixel], channel: int, *, signed: bool) -> list[int]:
    """One channel of 16 pixels as `encode_bc4` takes it: as it is, or about zero when signed."""
    if signed:
        return [max(-127, min(127, p[channel] - 128)) for p in px]
    return [p[channel] for p in px]


def encode_block(codec: str, px: list[Pixel], *, signed: bool = False, search: bool = False) -> bytes:
    """
    16 RGBA pixels (0-255) -> one block of `codec`. BC4 takes the red channel, BC5 red and green.

    Signed BC4/BC5 use the same 0-255 scale that `decode_block` returns, with 128 meaning zero, so a
    normal map round-trips as the picture it looks like. `search` picks colour blocks the slow,
    close way (`encode_bc1_search`).

    Returns:
        The block: 8 bytes for BC1 and BC4, 16 for BC3 and BC5.

    Raises:
        ValueError: If `codec` is not one of the four.
    """
    if codec == "BC1":
        return encode_bc1_search([p[:3] for p in px]) if search else encode_bc1(px)
    if codec == "BC3":
        return encode_bc4([p[3] for p in px]) + (encode_bc1_search(px) if search else encode_bc1(px))
    if codec == "BC4":
        return encode_bc4(_channel(px, 0, signed=signed), signed=signed)
    if codec == "BC5":
        red, green = _channel(px, 0, signed=signed), _channel(px, 1, signed=signed)
        return encode_bc4(red, signed=signed) + encode_bc4(green, signed=signed)
    raise ValueError(f"cannot encode {codec}")


def decode_block(codec: str, blk: bytes | bytearray, *, signed: bool = False) -> list[Pixel]:
    """One block -> 16 (r, g, b, a) pixels. Signed BC4/BC5 come back shifted to 0-255 (0 -> 128)."""
    if codec in {"BC1", "BC3"}:
        col = blk[8:] if codec == "BC3" else blk
        c0, c1, bits = struct.unpack("<HHI", col)
        pal = _bc1_palette(c0, c1, four=codec == "BC3")
        alpha = decode_bc4(blk[:8]) if codec == "BC3" else None
        out = []
        for i in range(16):
            k = (bits >> (2 * i)) & 3
            a = int(alpha[i]) if alpha else (0 if k == 3 and c0 <= c1 else 255)
            out.append((*pal[k], a))
        return out
    bias = 128 if signed else 0
    r = decode_bc4(blk[:8], signed=signed)
    if codec == "BC4":
        return [(int(v) + bias,) * 3 + (255,) for v in r]
    g = decode_bc4(blk[8:], signed=signed)
    return [(int(r[i]) + bias, int(g[i]) + bias, 255, 255) for i in range(16)]


# -- whole pictures ---------------------------------------------------------------------------------


def decode(d: bytes | bytearray, tex: Texture, level: int = 0) -> Rows:
    """One mip level as rows of (r, g, b, a)."""
    codec, bpp = FORMATS[tex.format >> 8]
    width, height = tex.mip_size(level)
    wide, high = _mip_blocks(tex, level)
    blocks = read_mip(d, tex, level)
    rows = [[None] * (wide * 4) for _ in range(high * 4)]
    for by in range(high):
        for bx in range(wide):
            at = (by * wide + bx) * bpp
            px = decode_block(codec, blocks[at : at + bpp], signed=tex.signed)
            for i in range(16):
                rows[by * 4 + i // 4][bx * 4 + i % 4] = px[i]
    return [row[:width] for row in rows[:height]]


def _halve(rows: Rows) -> Rows:
    """Box-filter a picture down one mip level."""
    height, width = len(rows), len(rows[0])
    out = []
    for y in range(max(1, height // 2)):
        y0, y1 = min(2 * y, height - 1), min(2 * y + 1, height - 1)
        row = []
        for x in range(max(1, width // 2)):
            x0, x1 = min(2 * x, width - 1), min(2 * x + 1, width - 1)
            a, b, c, e = rows[y0][x0], rows[y0][x1], rows[y1][x0], rows[y1][x1]
            row.append(tuple((a[i] + b[i] + c[i] + e[i] + 2) // 4 for i in range(4)))
        out.append(row)
    return out


def replace(d: bytearray, tex: Texture, rows: Rows, *, search: bool = False) -> None:
    """
    Encode a picture (the texture's own size) into `tex`, every mip, in place.

    `search` encodes colour blocks the slow, close way: for small pictures seen up close.

    Raises:
        ValueError: If the picture is not the texture's size.
    """
    codec = FORMATS[tex.format >> 8][0]
    if len(rows) != tex.height or len(rows[0]) != tex.width:
        raise ValueError(f"{tex.name} is {tex.width}x{tex.height}; the image is {len(rows[0])}x{len(rows)}")
    for level in range(tex.mips):
        width, height = tex.mip_size(level)
        wide, high = _mip_blocks(tex, level)
        blocks = bytearray()
        for by in range(high):
            for bx in range(wide):
                px = [rows[min(by * 4 + i // 4, height - 1)][min(bx * 4 + i % 4, width - 1)] for i in range(16)]
                blocks += encode_block(codec, px, signed=tex.signed, search=search)
        write_mip(d, tex, level, blocks)
        if level + 1 < tex.mips:
            rows = _halve(rows)


def fill(d: bytearray, tex: Texture, rgba: Pixel) -> None:
    """
    Make every pixel of every mip one colour (0-255 per channel; 128 is zero in a signed format).

    Raises:
        ValueError: If the texture's data is not a whole number of blocks.
    """
    codec, bpp = FORMATS[tex.format >> 8]
    block = encode_block(codec, [tuple(rgba)] * 16, signed=tex.signed)
    start, end = tex.offsets[0], tex.offsets[0] + tex.size
    if (end - start) % bpp:
        raise ValueError(f"{tex.name}: data size is not a whole number of blocks")
    # Every block is the same, so the layout does not matter: fill the whole span, padding included.
    d[start:end] = block * ((end - start) // bpp)
