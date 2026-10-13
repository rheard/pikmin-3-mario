"""The code patch: it assembles, fits the room set aside for it, and survives the patch format."""

import pytest

from mario import model, moves, voice
from mario.errors import BuildError

FIRST_CLIP = 0x87  # where the game's list puts his first punch


def called(address: int, word: int) -> int | None:
    """Where the call at `address` goes, or None if the word there is not a call."""
    if word >> 24 != 0xEB:
        return None
    return address + 8 + 4 * (((word & 0xFFFFFF) ^ 0x800000) - 0x800000)


def test_the_builders_agree_on_the_clips():
    """The code counts clips in the order the model adds them, and every line of voice has a clip."""
    assert tuple(model.MOVES) == moves.CLIPS
    assert {clip for clip, _ in voice.MOVES} | {clip for clip, _ in voice.JUMPS} <= set(model.MOVES)


def test_program_fits_its_room():
    """All the code is inside the unused stretches; only the hooks are outside."""
    words = moves.program(FIRST_CLIP, model.HEIGHT)
    hooks = sorted(address for address in words if not moves.in_room(address))
    assert len(hooks) == 10
    for address in hooks:  # each changed word of the game's is a call into the new code
        target = called(address, words[address])
        assert target is not None
        assert moves.in_room(target)
        assert target in words


def test_a_fall_is_reached_through_the_jump():
    """The fall's code is a stretch of its own, called from five places in the jump's and from nowhere else."""
    words = moves.program(FIRST_CLIP, model.HEIGHT)
    (fall, fall_size), (jump, jump_size) = moves.ROOM["room_fall"], moves.ROOM["room_jump"]
    assert any(fall <= address < fall + 4 * fall_size for address in words)
    calls = [address for address, word in words.items() if fall <= (called(address, word) or 0) < fall + 4 * fall_size]
    assert len(calls) == 5
    assert all(jump <= address < jump + 4 * jump_size for address in calls)
    assert moves.bits(-moves.FALL_SPEED) in words.values()  # what his upward speed is held against is a downward one


def test_a_step_down_is_not_a_fall():
    """A fall waits until he has dropped a little way, and not so far that he walks on the air."""
    assert model.HEIGHT / 5 < moves.fall_drop() < model.HEIGHT / 3
    assert moves.GRAVITY * moves.FALL_AFTER / 30.0 > moves.FALL_SPEED  # by then, falling freely, he is fast enough


def test_patch_round_trip():
    """The patch file holds exactly the program's words, and a damaged one is not read."""
    words = moves.program(FIRST_CLIP, model.HEIGHT)
    patch = moves.ips(words)
    assert moves.read_ips(patch) == words
    for damaged in (patch[1:], patch[:-4], patch + b"\0"):
        with pytest.raises(BuildError):
            moves.read_ips(damaged)


def test_clip_numbers_reach_the_code():
    """Another place in the list changes the code, and only a handful of immediates in it."""
    here, there = moves.program(FIRST_CLIP, model.HEIGHT), moves.program(FIRST_CLIP + 1, model.HEIGHT)
    assert here.keys() == there.keys()
    changed = [address for address in here if here[address] != there[address]]
    assert 0 < len(changed) < 10
    assert all((here[address] ^ there[address]) < 0x100 for address in changed)


def test_clip_numbers_must_fit_a_byte():
    """A clip number the instructions cannot hold is refused."""
    with pytest.raises(BuildError):
        moves.program(0x100 - len(moves.CLIPS) + 1, model.HEIGHT)


def test_jumps_grow_with_him():
    """Each jump is higher than the last, and all of them scale with his height."""
    heights = [moves.leap(speed, model.HEIGHT)[1] for speed in moves.SM64["jumps"]]
    assert heights == sorted(heights)
    assert heights[0] > model.HEIGHT  # even the first clears his own height
    speed, apex = moves.leap(moves.SM64["jumps"][0], model.HEIGHT)
    assert apex == pytest.approx(speed * speed / (2 * moves.GRAVITY))
    assert moves.leap(moves.SM64["jumps"][0], 2 * model.HEIGHT)[1] == pytest.approx(2 * apex)
