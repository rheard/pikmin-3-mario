"""The file formats, on data made here: nothing in these tests needs a game."""

import random
import struct

import pytest

from mario.formats import bfsar, bntx, sarc, yaz0

ALIGNMENT = 0x1000


def noisy(size: int, seed: int = 64) -> bytes:
    """Bytes with runs, repeats and noise in them, the way real files have."""
    rng = random.Random(seed)
    words = [b"Pikmin", b"Mario", bytes(8), bytes(rng.randrange(256) for _ in range(40))]
    out = bytearray()
    while len(out) < size:
        out += rng.choice(words) if rng.random() < 0.7 else bytes([rng.randrange(256)])
    return bytes(out[:size])


def make_sarc(files: dict[str, bytes], align: int = 0x100) -> bytes:
    """A little-endian SARC archive holding `files`, each on a multiple of `align`."""
    names, nodes, data = bytearray(), [], bytearray()
    for name, body in files.items():
        hashed = 0
        for byte in name.encode():
            hashed = (hashed * 0x65 + byte) & 0xFFFFFFFF
        data += bytes(-len(data) % align)
        nodes.append((hashed, 0x01000000 | len(names) // 4, len(data), len(data) + len(body)))
        data += body
        names += name.encode() + b"\0"
        names += bytes(-len(names) % 4)
    table = b"SFAT" + struct.pack("<HHI", 0xC, len(nodes), 0x65) + b"".join(struct.pack("<4I", *node) for node in nodes)
    table += b"SFNT" + struct.pack("<HH", 8, 0) + names
    start = 0x14 + len(table)
    start += -start % align
    header = b"SARC" + struct.pack("<HHIIHH", 0x14, 0xFEFF, start + len(data), start, 0x100, 0)
    return (header + table).ljust(start, b"\0") + bytes(data)


# -- Yaz0 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [bytes(3000), b"abc" * 2000, noisy(20000), bytes(range(256)) * 4],
    ids=["zeros", "repeats", "noisy", "ramp"],
)
def test_yaz0_round_trip(data: bytes):
    """What is compressed comes back, under a header that carries the alignment."""
    packed = yaz0.compress(data, ALIGNMENT)
    assert packed[:4] == b"Yaz0"
    assert yaz0.alignment(packed) == ALIGNMENT
    assert yaz0.decompress(packed) == data


def test_yaz0_compresses():
    """Repeats cost next to nothing."""
    assert len(yaz0.compress(b"abc" * 2000, ALIGNMENT)) < 200


def test_yaz0_leaves_other_data_alone():
    """Anything that is not a Yaz0 stream comes back as it was."""
    assert yaz0.decompress(b"SARC and so on") == b"SARC and so on"


def test_yaz0_patch():
    """A stream patched for a small change unpacks to the changed data, and is mostly the old stream."""
    data = noisy(400000)
    stock = yaz0.compress(data, ALIGNMENT)
    changed = bytearray(data)
    changed[200000:200040] = bytes(range(40))
    patched = yaz0.patch(stock, bytes(changed))
    assert yaz0.decompress(patched) == changed
    assert patched[:16] == stock[:16]
    same = sum(a == b for a, b in zip(patched[: len(stock) // 3], stock, strict=False))
    assert same == len(stock) // 3  # everything before the change is the stock stream, untouched


# -- SARC ------------------------------------------------------------------------------------------


def test_sarc_reads_what_was_packed():
    """Files come back by name, with where they lie."""
    files = {"one.txt": b"first", "dir/two.bin": bytes(range(200)), "three": b""}
    archive = sarc.Sarc(make_sarc(files))
    assert archive.is_archive
    assert archive.files() == files
    assert archive.sizes() == {name: len(body) for name, body in files.items()}
    lo, hi = archive.span("dir/two.bin")
    assert archive.data[lo:hi] == files["dir/two.bin"]


def test_sarc_rebuild():
    """Rebuilding changes the files asked for and nothing else; changing nothing changes nothing."""
    files = {"one.txt": b"first", "two.bin": bytes(range(200)), "three": b"third"}
    plain = make_sarc(files)
    assert sarc.rebuild(plain, {}) == plain
    longer = bytes(1000)
    rebuilt = sarc.Sarc(sarc.rebuild(plain, {"two.bin": longer}))
    assert rebuilt.files() == {**files, "two.bin": longer}
    assert all(start % 0x100 == 0 for _, start, _ in rebuilt.entries)


def test_single_file_is_not_an_archive():
    """A `.szs` may hold one bare file; that is not mistaken for an archive."""
    assert not sarc.Sarc(b"FRES....").is_archive


# -- BNTX block compression ------------------------------------------------------------------------


def worst(a: list, b: list, channels: tuple[int, ...]) -> int:
    """The largest difference between two blocks of pixels over some channels."""
    return max(abs(p[c] - q[c]) for p, q in zip(a, b, strict=True) for c in channels)


@pytest.mark.parametrize("search", [False, True])
def test_bc3_keeps_a_two_colour_block(*, search: bool):
    """A block of two colours and smooth coverage survives to within what 5:6:5 loses."""
    block = [((200, 30, 30, 255) if i % 4 < 2 else (20, 40, 200, 16 * i)) for i in range(16)]
    back = bntx.decode_block("BC3", bntx.encode_block("BC3", block, search=search))
    assert worst(block, back, (0, 1, 2)) <= 8
    assert worst(block, back, (3,)) <= 18


def test_bc5_keeps_two_channels():
    """A grey picture's two channels survive; the other two are not stored."""
    block = [(i * 16, 255 - i * 16, 0, 255) for i in range(16)]
    back = bntx.decode_block("BC5", bntx.encode_block("BC5", block))
    assert worst(block, back, (0, 1)) <= 18


def test_bc4_ends_are_exact():
    """A block's lowest and highest values are stored as they are."""
    values = [10, 240] + [100] * 14
    back = bntx.decode_bc4(bntx.encode_bc4(values))
    assert round(back[0]) == 10
    assert round(back[1]) == 240


# -- BFSAR -----------------------------------------------------------------------------------------


def test_wave_round_trip():
    """A wave made here reads back sample for sample."""
    samples = [round(12000 * ((i % 50) / 25 - 1)) for i in range(2000)]
    wave = bfsar.read_wave(bfsar.make_wave(samples, 16000))
    assert wave["samples"] == samples
    assert wave["rate"] == 16000
    assert wave["channels"] == 1
    assert not wave["loops"]


def test_wave_archive_round_trip():
    """A wave archive hands back each wave it was given."""
    waves = [bfsar.make_wave([i] * 100, 32000) for i in (1, 2, 3)]
    archive = bfsar.make_wave_archive(waves)
    found = bfsar.read_wave_archive(archive)
    assert len(found) == len(waves)
    assert [bfsar.read_wave(archive, offset)["samples"][0] for offset, _ in found] == [1, 2, 3]


def test_wave_sounds_round_trip():
    """A wave sound file names the wave archive and wave of each of its sounds, in order."""
    named = [(0x05000000 | 13, i) for i in range(4)]
    assert [(archive, wave) for archive, wave, _ in bfsar.read_wave_sounds(bfsar.make_wave_sounds(named))] == named


def test_user_fields_round_trip():
    """The bit fields that pick a sound by who is making it can each be set and read alone."""
    params = bfsar.with_fields((0, 0), variants=3, weight=5, gap=7, gap_spread=8, spaced=1, kind=2)
    assert [
        bfsar.user_field(params, name) for name in ("variants", "weight", "gap", "gap_spread", "spaced", "kind")
    ] == [
        3,
        5,
        7,
        8,
        1,
        2,
    ]
    assert bfsar.user_field(bfsar.with_fields(params, weight=1), "variants") == 3
