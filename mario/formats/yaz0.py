"""
Yaz0, the compression around the game's files (the `.szs` extension).

A Yaz0 file is a 16-byte big-endian header (the letters `Yaz0`, the size of the data once
decompressed, an alignment, four zero bytes) and then a run of groups. A group is a flag byte and up
to eight pieces, one for each bit from the top down: a set bit is one plain byte, a clear bit is a
copy of 3 to 0x111 bytes from 1 to 0x1000 bytes back in what has been written so far. A copy is two
bytes, `NR RR`: N + 2 bytes from RRR + 1 back. When N is 0 a third byte follows, and the length is
that byte + 0x12. A copy may run into the bytes it is itself writing, which is how a long run of one
value is stored.

The alignment matters. The game allocates the buffer it decompresses into with it, and when it is
zero asks the resource being loaded for a default, through a virtual call that model archives do not
implement: "Pure virtual function called!", and the game stops (FUN_610dbf10). So a file made here
must carry the stock file's value, which `alignment` reads: 0x20 for the archives of text files,
0x1000 for the leaders' model archive, 0x2000 for the common layout archive and 0x4000 for the
effect file.

`decompress` and `compress` handle whole files. `patch` is for changing a few places in a large one:
it keeps the stock file's stream and rewrites only the groups near a change.
"""

from __future__ import annotations

import struct

MAGIC = b"Yaz0"
HEADER = 16
FURTHEST = 0x1000  # how far back a copy may reach
LONGEST = 0x111  # how many bytes a copy may write

Piece = tuple[bool, bytes]  # (is a plain byte, its bytes in the stream)


def decompress(data: bytes | bytearray) -> bytes:
    """Decompress a Yaz0 stream; anything that is not one comes back unchanged."""
    if data[:4] != MAGIC:
        return data
    size = struct.unpack_from(">I", data, 4)[0]
    out = bytearray()
    src = HEADER
    n_data = len(data)
    while len(out) < size and src < n_data:
        code = data[src]
        src += 1
        for bit in range(7, -1, -1):
            if len(out) >= size:
                break
            if code >> bit & 1:
                out.append(data[src])
                src += 1
            else:
                b1, b2 = data[src], data[src + 1]
                src += 2
                dist = ((b1 & 0x0F) << 8 | b2) + 1
                n = b1 >> 4
                if n == 0:
                    n = data[src] + 0x12
                    src += 1
                else:
                    n += 2
                start = len(out) - dist
                if dist >= n:
                    out += out[start : start + n]
                else:
                    # Overlapping copy: the source repeats with period `dist`.
                    chunk = out[start:]
                    reps, rest = divmod(n, dist)
                    out += chunk * reps + chunk[:rest]
    return bytes(out)


def alignment(raw: bytes | bytearray) -> int:
    """The alignment a Yaz0 file asks for its decompressed data (0 when it is not a Yaz0 file)."""
    return struct.unpack_from(">I", raw, 8)[0] if raw[:4] == MAGIC else 0


def compress(data: bytes | bytearray, alignment: int, tries: int = 4) -> bytes:
    """
    Compress bytes into a Yaz0 stream the game will read.

    `alignment` goes in the header and must be the stock file's: see the top of this module.

    Matching is greedy against the last 4 KB: the `tries` nearest earlier copies of the next three
    bytes are tried and the longest kept. That is slower and a little looser than Nintendo's packer
    (about 20 seconds for a 14 MB model file, 5% larger), which is fine for a file built once.

    Returns:
        The whole file, header included.
    """
    data = bytes(data)
    n = len(data)
    out = bytearray(MAGIC + struct.pack(">II", n, alignment) + bytes(4))
    pos = 0
    rfind = data.rfind
    while pos < n:
        code_at = len(out)
        out.append(0)
        code = 0
        for bit in range(7, -1, -1):
            if pos >= n:
                break
            best, best_at = 0, 0
            limit = min(LONGEST, n - pos)
            if limit >= 3:
                key = data[pos : pos + 3]
                lo = pos - FURTHEST if pos > FURTHEST else 0
                at = rfind(key, lo, pos + 2)  # a copy may overlap the bytes it produces
                left = tries
                while at >= 0:
                    # Longest common run of data[at:] and data[pos:], by bisection on slice equality.
                    a, b = 3, limit
                    while a < b:
                        mid = (a + b + 1) >> 1
                        if data[at : at + mid] == data[pos : pos + mid]:
                            a = mid
                        else:
                            b = mid - 1
                    if a > best:
                        best, best_at = a, at
                        if a == limit:
                            break
                    left -= 1
                    if not left:
                        break
                    at = rfind(key, lo, at + 2)
            if best >= 3:
                out += _copy(pos - best_at, best)
                pos += best
            else:
                code |= 1 << bit
                out.append(data[pos])
                pos += 1
        out[code_at] = code
    return bytes(out)


def differences(old: bytes | bytearray, new: bytes | bytearray, step: int = 0x10000) -> list[tuple[int, int]]:
    """
    Find where two equally long buffers differ.

    Returns:
        (start, end) of each run of differing bytes, in order.

    Raises:
        ValueError: If the buffers are not the same length.
    """
    if len(old) != len(new):
        raise ValueError(f"the buffers differ in length ({len(old)} and {len(new)})")
    spans = []
    for base in range(0, len(old), step):
        a, b = old[base : base + step], new[base : base + step]
        if a == b:
            continue
        i = 0
        while i < len(a):
            if a[i] == b[i]:
                i += 1
                continue
            j = i
            while j < len(a) and a[j] != b[j]:
                j += 1
            if spans and spans[-1][1] == base + i:
                spans[-1] = (spans[-1][0], base + j)
            else:
                spans.append((base + i, base + j))
            i = j
    return spans


def patch(stock: bytes | bytearray, plain: bytes | bytearray) -> bytes:
    """
    Make a Yaz0 stream for `plain` out of the stock stream of a file of the same length.

    This is for changing a few places in a large file: compressing 70 MB with `compress` takes
    minutes, and all but a sliver of the result would say what the stock stream already says. A
    group can be kept as it is unless one of its pieces writes changed bytes or copies from them,
    which only happens within 0x1000 bytes after a change. So the groups near a change are taken
    apart: plain bytes become the new bytes, a copy that is no longer true becomes plain bytes, and
    the pieces are packed into groups again. That leaves the count of pieces out of step with the
    stock groups that follow, so a few of their copies are cut in two (same distance, the same bytes
    written) until it is a multiple of eight again, and from there the rest of the stream is the
    stock's, byte for byte.

    The header (the size, and the alignment the game allocates with) is the stock file's. Check the
    result as for any packing: `decompress(result) == plain`.

    Returns:
        The whole file, header included.

    Raises:
        ValueError: If `stock` is not a Yaz0 file, or does not unpack to the length of `plain`.
    """
    if stock[:4] != MAGIC:
        raise ValueError("not a Yaz0 file")
    size = struct.unpack_from(">I", stock, 4)[0]
    plain = bytes(plain)
    if size != len(plain):
        raise ValueError(f"the stock file unpacks to {size} bytes; the new data is {len(plain)}")
    spans = differences(decompress(stock), plain)
    if not spans:
        return bytes(stock)
    zones = _zones(spans)
    out = bytearray(stock[:HEADER])
    queue: list[Piece] = []  # pieces waiting for a group
    src, pos, zone = HEADER, 0, 0
    while pos < size:
        end, at = _group_end(stock, src, pos, size)
        while zone < len(zones) and zones[zone][1] <= pos:
            zone += 1
        hot = zone < len(zones) and zones[zone][0] < at
        if not hot and not queue:
            out += stock[src:end]
        else:
            # Outside the changed places every copy is still true, and there the pieces short of a
            # whole group are made up by cutting copies in two.
            short = 0 if hot else (8 - len(queue)) % 8
            queue += _take_apart(stock, plain, src, pos, short)
            _flush(queue, out)
        src, pos = end, at
    _flush(queue, out, everything=True)
    return bytes(out) + bytes(stock[src:])


def _copy(distance: int, n: int) -> bytes:
    """One Yaz0 copy: `n` bytes (3 to 0x111) from `distance` bytes back (1 to 0x1000)."""
    d = distance - 1
    if n >= 0x12:
        return bytes((d >> 8, d & 0xFF, n - 0x12))
    return bytes(((n - 2) << 4 | d >> 8, d & 0xFF))


def _zones(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The stretches of output whose pieces a change can touch, merged where they meet."""
    # A piece at output position p writes [p, p + n) and reads no further back than p - 0x1000.
    zones = []
    for start, end in spans:
        lo, hi = start - LONGEST, end + FURTHEST
        if zones and lo <= zones[-1][1]:
            zones[-1] = (zones[-1][0], hi)
        else:
            zones.append((lo, hi))
    return zones


def _group_end(stock: bytes | bytearray, src: int, pos: int, size: int) -> tuple[int, int]:
    """Where the group at `src`, which writes from `pos`, ends: in the stream and in the output."""
    code = stock[src]
    if code == 0xFF and pos + 8 <= size:
        return src + 9, pos + 8
    p, at = src + 1, pos
    for bit in range(7, -1, -1):
        if at >= size:
            break
        if code >> bit & 1:
            p += 1
            at += 1
        elif stock[p] >> 4:
            at += (stock[p] >> 4) + 2
            p += 2
        else:
            at += stock[p + 2] + 0x12
            p += 3
    return p, at


def _take_apart(stock: bytes | bytearray, plain: bytes, src: int, pos: int, short: int) -> list[Piece]:
    """The pieces of the group at `src`, made true of `plain`, with up to `short` copies cut in two."""
    size = len(plain)
    code = stock[src]
    pieces: list[Piece] = []
    p, at = src + 1, pos
    for bit in range(7, -1, -1):
        if at >= size:
            break
        if code >> bit & 1:
            pieces.append((True, plain[at : at + 1]))
            p += 1
            at += 1
            continue
        if stock[p] >> 4:
            n, length = (stock[p] >> 4) + 2, 2
        else:
            n, length = stock[p + 2] + 0x12, 3
        distance = ((stock[p] & 0x0F) << 8 | stock[p + 1]) + 1
        if plain[at : at + n] != plain[at - distance : at - distance + n]:
            pieces.extend((True, plain[k : k + 1]) for k in range(at, at + n))
        elif short and n >= 6:
            pieces += [(False, _copy(distance, 3)), (False, _copy(distance, n - 3))]
            short -= 1
        else:
            pieces.append((False, stock[p : p + length]))
        p += length
        at += n
    return pieces


def _flush(queue: list[Piece], out: bytearray, *, everything: bool = False) -> None:
    """Move whole groups of eight pieces from the queue to the stream; with `everything`, the rest too."""
    while len(queue) >= 8 or (everything and queue):
        group, code = queue[:8], 0
        del queue[:8]
        for k, (literal, _) in enumerate(group):
            if literal:
                code |= 0x80 >> k
        out.append(code)
        for _, body in group:
            out.extend(body)
