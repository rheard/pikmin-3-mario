"""The parts of the builders that can be checked without a game."""

import math
import random
import re

from types import SimpleNamespace

import pytest

from mario import model, voice
from mario.errors import BuildError
from mario.geometry import euler_matrix, matmul, matrix_euler, to_local, to_model, transpose
from mario.sm64.rom import Sequence

CLIP_LIST = (
    b"20121218 \r\n3 0 0 4 0 0 0 0 \r\n# skeletal \r\n{\r\n"
    b'\t# \r\n\t{\r\n\t\t"wait" \r\n\t\t"wait" \r\n\t\t0 \r\n\t}\r\n'
    b'\t# \r\n\t{\r\n\t\t"player_wait" \r\n\t\t"wait" \r\n\t\t2 \r\n\t\t0.00000000 2 "" \r\n\t\t19.00000000 4 "" '
    b"\r\n\t}\r\n"
    b'\t# \r\n\t{\r\n\t\t"player_walk" \r\n\t\t"walk" \r\n\t\t2 \r\n\t\t5.00000000 2 "" \r\n\t\t25.00000000 4 "" '
    b"\r\n\t}\r\n}\r\n# colour \r\n{\r\n}\r\n"
)
# Enough of Mario for the list: his three standing animations are 30 frames each.
HIS = SimpleNamespace(sequence=lambda pieces: SimpleNamespace(frames=30 * len(pieces)))


class Turning:
    """A stand-in for one of his animations: one part turning, and the root rising, steadily from `start`."""

    def __init__(self, frames: int, start: float, step: float) -> None:
        self.frames, self.start, self.step = frames, start, step

    def root(self, frame: float) -> tuple[float, float, float]:
        """The root's position at a frame."""
        return (0.0, self.start + self.step * frame, 0.0)

    def rotations(self, frame: float) -> dict[str, tuple[float, float, float]]:
        """The one part's angles at a frame."""
        return {"head": (self.start + self.step * frame, 0.0, 0.0)}


def close(a: list, b: list, tolerance: float = 1e-9) -> bool:
    """Whether two rotations are the same to within rounding."""
    return all(abs(a[i][j] - b[i][j]) < tolerance for i in range(3) for j in range(3))


def test_angles_round_trip():
    """Any rotation's angles build the same rotation again."""
    rng = random.Random(64)
    for _ in range(200):
        rotation = euler_matrix(*(rng.uniform(-math.pi, math.pi) for _ in range(3)))
        assert close(euler_matrix(*matrix_euler(rotation)), rotation)
        assert close(matmul(rotation, transpose(rotation)), euler_matrix(0.0, 0.0, 0.0))


def test_angles_stay_near_the_last_ones():
    """Asked for angles near a previous triple, the same rotation comes back without a jump of a turn."""
    near = (3.0, 0.2, -3.1)
    rotation = euler_matrix(3.2, 0.25, -3.2)  # past half a turn on two axes
    angles = matrix_euler(rotation, near)
    assert close(euler_matrix(*angles), rotation)
    assert max(abs(a - n) for a, n in zip(angles, near, strict=True)) < 0.3


def test_bone_space_round_trip():
    """A point taken into a bone's space and back is where it was."""
    rotation, origin, point = euler_matrix(0.3, -1.1, 2.0), (1.0, 20.0, -3.0), (4.0, 5.0, 6.0)
    back = to_model(rotation, origin, to_local(rotation, origin, point))
    assert back == pytest.approx(point)


def test_moves_are_added_to_the_end_of_the_list():
    """The clip list gains an entry for each move after its own, and nothing else in it changes."""
    new, first = model.define_moves(CLIP_LIST, model.MOVES)
    assert first == 3
    assert new.split(b"\r\n")[1] == b"%d 0 0 4 0 0 0 0 " % (3 + len(model.MOVES))
    body, tail = CLIP_LIST[CLIP_LIST.index(b"{") :].split(b"\r\n}\r\n", 1)
    assert body in new
    assert new.endswith(b"\r\n}\r\n" + tail)
    added = new[new.index(body) + len(body) : -len(tail) - 5].decode()
    assert re.findall(r'\{\r\n\t\t"(\w+)" \r\n\t\t"(\w+)" ', added) == [(name, name) for name in model.MOVES]
    assert (
        '\r\n\t\t3.00000000 1024 "" \r\n\t\t4.00000000 2048 "" ' in added
    )  # the first punch lands, then takes a press
    assert model.clip_loops(new) == {"wait": (0, 19), "walk": (5, 25)}


def test_a_list_that_is_not_one_is_refused():
    """Text that is not laid out as the game's list is refused, not guessed at."""
    with pytest.raises(BuildError):
        model.define_moves(b"not a list", model.MOVES)


def test_a_sequence_comes_round():
    """Animations play one after another, a frame a frame, and the last leads back into the first."""
    sequence = Sequence([Turning(3, 0.0, 1.0), Turning(2, 3.0, 1.5)])
    assert sequence.frames == 5
    assert [sequence.rotations(float(frame))["head"][0] for frame in range(6)] == [0.0, 1.0, 2.0, 3.0, 4.5, 0.0]
    assert sequence.rotations(2.5)["head"][0] == pytest.approx(2.5)  # from the end of one to the start of the next
    assert sequence.root(3.5)[1] == pytest.approx(3.75)
    # From the last frame back to the first, an angle goes the short way round and a place straight back.
    assert sequence.rotations(4.5)["head"][0] == pytest.approx(4.5 + (2 * math.pi - 4.5) / 2)
    assert sequence.root(4.5)[1] == pytest.approx(2.25)


def test_a_stand_takes_over_the_leaders_entry():
    """The standing entry names his clip and loops over all of it; its place and every other line stay."""
    stand = model.STANDS["m_wait"]
    assert model.stand_loop(CLIP_LIST, stand) == 19
    assert model.stand_length(HIS, stand, 19) == 95  # five rounds of the leaders' loop are the nearest to his ninety
    new = model.define_stands(CLIP_LIST, HIS)
    changed = [line for old, line in zip(CLIP_LIST.split(b"\r\n"), new.split(b"\r\n"), strict=True) if old != line]
    assert changed == [b'\t\t"m_wait" ', b'\t\t95.00000000 4 "" ']
    assert model.clip_entries(new) == [
        ("wait", "wait", []),
        ("player_wait", "m_wait", [(0.0, 2), (95.0, 4)]),
        ("player_walk", "walk", [(5.0, 2), (25.0, 4)]),
    ]


@pytest.mark.parametrize(
    "broken",
    [
        CLIP_LIST.replace(b'"player_wait"', b'"player_rest"'),  # no such entry
        CLIP_LIST.replace(b'"player_wait" \r\n\t\t"wait"', b'"player_wait" \r\n\t\t"walk"'),  # it plays another clip
        CLIP_LIST.replace(b"0.00000000 2", b"1.00000000 2"),  # its loop does not start on the first frame
        CLIP_LIST.replace(b"\t\t19.00000000 4", b"\t\t19.00000000 8"),  # it does not loop at all
    ],
)
def test_a_standing_entry_that_is_not_the_games_is_refused(broken: bytes):
    """A list whose standing entry is not the one the game has is refused, not pointed somewhere anyway."""
    with pytest.raises(BuildError):
        model.define_stands(broken, HIS)


def test_moves_are_as_long_as_their_pieces():
    """A move shows each frame of each piece once, and holds the last for as long as it says."""
    assert model.move_steps(model.MOVES["m_punch1"]) == [("first_punch", f) for f in range(4)] + [
        ("first_punch_fast", f) for f in range(10)
    ]
    jump = model.move_steps(model.MOVES["m_jump1"])
    assert len(jump) == 12 + 40
    assert jump[-41:] == [("single_jump", 11)] * 41
    for move in model.MOVES.values():
        part, frame = model.move_spark(move)
        assert part in {"right_hand", "left_hand", "right_foot"}
        assert 0 <= frame < len(model.move_steps(move))


def tone(frequency: float, rate: int, count: int = 4000) -> list[int]:
    """A sine wave as samples."""
    return [round(10000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(count)]


def test_halving_keeps_what_the_new_rate_can_hold():
    """A low tone comes through at full strength; one the new rate cannot hold is taken out."""
    low, high = voice.halve(tone(1000, 32000)), voice.halve(tone(12000, 32000))
    assert len(low) == 2000
    assert max(low[100:-100]) == pytest.approx(10000, rel=0.02)
    assert max(abs(sample) for sample in high[100:-100]) < 300


def test_a_line_is_named_for_its_clip_and_frame():
    """Each line he says gets a record name of its own in the game's table."""
    assert voice.Line("throw", 1, []).name == "Pikmin/throw/LeaderVoice/1"
