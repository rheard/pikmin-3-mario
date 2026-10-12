"""
Mario's voice: what he says when he throws a Pikmin, punches, kicks, jumps and lands.

What is written
---------------
* `audio/audiores/CarrotAudio.bfsar`: the game's sound archive, with entries for his sounds.
* `audio/audiores/GROUP_STATIC.bfgrp`: the one file all the game's sounds are loaded from, with his
  on the end.
* `command/command_table.szs`: the table that ties sounds to frames of animation clips.

How, all of it data the game already reads
------------------------------------------
* **When.** The game ties sounds to frames of animation clips in a text table. The leaders' punch
  is the pattern: one record plays a grunt at frame 0 of the punch clip. A copy of that record does
  the same for the throw, and one each for his moves' clips.
* **Who.** The punch's grunt is a family of sounds, one run for each kind of Pikmin and each leader,
  and the game picks the run by who is making the sound. Each new family has Mario in Alph's run
  and a moment of silence at no volume in all the others, so nobody else changes.
* **Which.** A sound can say "pick among the next N at random, with these weights", which is how
  each leader already has two punch grunts. His run uses that where his own game picks among
  several, and the game's own spacing rule keeps the throw from speaking on every Pikmin of a fast
  volley.
* **What.** The sounds themselves are new: a wave archive and a wave sound file added to the end of
  the group file, and entries for them added to the sound archive. No stock sound is touched.
"""

from __future__ import annotations

import logging
import math
import re
import struct

from typing import TYPE_CHECKING, NamedTuple

from mario.errors import BuildError
from mario.formats import bfsar, sarc, yaz0
from mario.romfs import repack
from mario.sm64.audio import Voice

if TYPE_CHECKING:
    from mario.romfs import Mod, RomFs
    from mario.sm64.rom import Rom

logger = logging.getLogger(__name__)

ARCHIVE = "audio/audiores/CarrotAudio.bfsar"
GROUP = "audio/audiores/GROUP_STATIC.bfgrp"
COMMANDS = "command/command_table.szs"
TABLE = "sound_table.txt"

LEADER = "A"  # Alph
PATTERN = "Pikmin/punch/Attack/0"  # the leaders' punch grunt: the record and the family that are copied
KIND = 2  # the pattern's way of choosing a run by who makes the sound: `bfsar.TYPE_SLOTS`
SILENCE = 512  # samples of nothing for everyone else: 16 ms
SILENCE_RATE = 32000
WAVE_ARCHIVE_ITEM = 0x05000000  # an item number's top byte says what kind of thing it names
SOUND_ITEM = 0x01000000
MAIN_GROUP = [0x06000000]

# What he says. Each line is a clip, the frame of it, and the sounds picked among, by weight.
# A throw: one of three, the first being the one his own game throws with, and not again for 7 to
# 15 frames. (Frame 1 is where the clip's own throw marker is.)
THROW = ("throw", 1, (("wah2", 2), ("punch_yah", 1), ("punch_hoo", 1)))
THROW_GAP, THROW_GAP_SPREAD = 7, 8
# A punch or kick: what his own game has him say as it starts, with the swish it adds a tick later.
MOVES = (("m_punch1", "punch_yah"), ("m_punch2", "punch_wah"), ("m_kick", "punch_hoo"))
# A jump, picked as his own game picks: one of three for the first; "hoo hoo" for the double; for
# the triple "yahoo" three times in five, else "waha" or "yippee"; and "haha" as he lands from it.
JUMPS = (
    ("m_jump1", (("jump_yah", 1), ("jump_wah", 1), ("jump_hoo", 1))),
    ("m_jump2", (("hoohoo", 1),)),
    ("m_jump3", (("yahoo", 3), ("waha", 1), ("yippee", 1))),
    ("m_land3", (("haha", 1),)),
)
# The jumps' sounds are five seconds in all, and every sound of the game's is loaded into one pool
# of memory at start-up whose spare room is not known. They are kept at half the rate the others are
# made at, which halves their size: his samples are recorded at 13 to 17 kHz, so nothing that was in
# them is above what 16 kHz holds.
JUMP_RATE = 16000


class Take(NamedTuple):
    """One sound he may make: its name, its weight among a line's takes, its samples and their rate."""

    name: str
    weight: int
    samples: list[int]
    rate: int


class Line:
    """
    One thing he says: at `frame` of the clip tagged `clip`, one of `takes`, picked by weight.

    `gap` and `gap_spread` keep it from being said again for that many frames, and up to that many
    more at random.
    """

    def __init__(self, clip: str, frame: int, takes: list[Take], gap: int = 0, gap_spread: int = 0):
        self.clip, self.frame, self.takes, self.gap, self.gap_spread = clip, frame, list(takes), gap, gap_spread
        self.name = f"Pikmin/{clip}/LeaderVoice/{frame}"


def halve(samples: list[int]) -> list[int]:
    """
    A sound at half its rate.

    It is smoothed first so that nothing above the new rate's reach folds back into it (a 31-point
    windowed low-pass at just under a quarter of the old rate), then every other sample is kept.

    Returns:
        Half as many samples.
    """
    taps = []
    for k in range(-15, 16):
        ideal = 0.46 if k == 0 else math.sin(0.46 * math.pi * k) / (math.pi * k)
        taps.append(ideal * (0.54 + 0.46 * math.cos(math.pi * k / 15.0)))
    total = sum(taps)
    taps = [t / total for t in taps]
    count, out = len(samples), []
    for i in range(0, count, 2):
        acc = 0.0
        for j, tap in enumerate(taps):
            k = i + j - 15
            if 0 <= k < count:
                acc += tap * samples[k]
        out.append(max(-32768, min(32767, round(acc))))
    return out


def lines(voice: Voice) -> list[Line]:
    """
    Everything he says, rendered from the ROM.

    Returns:
        A line for the throw, then one for each move, then one for each jump.

    Raises:
        BuildError: If a jump's sound is not made at the rate `halve` expects.
    """
    clip, frame, said = THROW
    takes = [Take(name, weight, *_sound(voice, name)) for name, weight in said]
    out = [Line(clip, frame, takes, THROW_GAP, THROW_GAP_SPREAD)]
    for clip, name in MOVES:
        out.append(Line(clip, 0, [Take(name + "_whole", 1, *_sound(voice, name, whole=True))]))
    for clip, said in JUMPS:
        takes = []
        for name, weight in said:
            samples, rate = _sound(voice, name)
            if rate != 2 * JUMP_RATE:
                raise BuildError(f"{name} is made at {rate} Hz, not twice {JUMP_RATE}")
            takes.append(Take(name, weight, halve(samples), JUMP_RATE))
        out.append(Line(clip, 0, takes))
    return out


def _sound(voice: Voice, name: str, *, whole: bool = False) -> tuple[list[int], int]:
    sound = voice.play(name, whole=whole)
    logger.info("  %-12s %.2f s%s", name, sound.seconds, ", with its swish" if whole else "")
    return sound.samples, sound.rate


def picks(archive: bfsar.Archive, sound: int, slot: int) -> list[tuple[int, int]]:
    """
    What the game may play for a record naming `sound`, made by an actor in run `slot`.

    Returns:
        (sound number, weight) for each candidate.
    """
    first = archive.sound(sound)["user"]
    if bfsar.user_field(first, "kind"):
        sound += slot * max(1, bfsar.user_field(first, "variants"))
    count = bfsar.user_field(archive.sound(sound)["user"], "variants")
    if count < 2:
        return [(sound, 16)]
    return [(sound + k, bfsar.user_field(archive.sound(sound + k)["user"], "weight") or 16) for k in range(count)]


def table_parts(table: bytes) -> tuple[int, bytes, int, list[bytes]]:
    """
    Take the sound table apart.

    Returns:
        How many records it has, the pattern's record, the largest key id, and every record's name.

    Raises:
        BuildError: If the table is not laid out as expected.
    """
    match = re.match(rb"# table version\r\n\d+\r\n# record num\r\n(\d+)\r\n\{\r\n", table)
    if not match or not table.endswith(b"\t}\r\n}\r\n"):
        raise BuildError("the sound table does not begin and end as expected")
    names = re.findall(rb'# command name\r\n\t\t\t"([^"]*)"', table)
    if len(names) != int(match.group(1)):
        raise BuildError(f"the sound table says {int(match.group(1))} records and holds {len(names)}")
    at = table.index(b'"' + PATTERN.encode() + b'"')
    start = table.rindex(b"\t# (OBJ)", 0, at)
    end = table.index(b"\t# (OBJ)", at)
    keys = [int(k) for k in re.findall(rb"# key id\r\n\t\t\t(\d+)", table)]
    return int(match.group(1)), table[start:end], max(keys), names


def _swap_once(text: bytes, old: bytes, new: bytes) -> bytes:
    if text.count(old) != 1:
        raise BuildError(f"the pattern record does not hold {old!r} exactly once")
    return text.replace(old, new)


def _set_field(record: bytes, field: bytes, value: bytes, old: bytes = rb"\d+") -> bytes:
    """A record with the number on the line after `# <field>` replaced (or whatever there matches `old`)."""
    return re.sub(rb"(# " + field + rb"\r\n\t\t\t)" + old, lambda match: match.group(1) + value, record, count=1)


def add_voice(archive: bytes, group: bytes, table: bytes, said: list[Line]) -> tuple[bytes, bytes, bytes]:
    """
    Add his voice to the game's three files: a family of sounds and a record for each line.

    Returns:
        The new sound archive, group file and sound table.

    Raises:
        BuildError: If the game's files are not put together the way this expects.
    """
    arc, grp = bfsar.Archive(archive), bfsar.Group(group)
    count, record, top_key, names = table_parts(table)
    if not said or len({line.name for line in said}) != len(said):
        raise BuildError("each line needs a clip and frame of its own")

    # The pattern: the punch grunt's record, its family, and the leader's own entry in it.
    base = int(re.search(rb"# sound id\r\n\t\t\t(0x[0-9a-fA-F]+)", record).group(1), 16) & 0xFFFFFF
    if bfsar.user_field(arc.sound(base)["user"], "kind") != KIND:
        raise BuildError(f"the punch grunt (sound {base:#x}) does not choose by who makes it the way expected")
    slots = bfsar.TYPE_SLOTS[KIND]
    pattern = picks(arc, base, slots[LEADER])[0][0]
    like = arc.sound(pattern)
    template = arc.raw(bfsar.SOUNDS, pattern)
    if like["kind"] != bfsar.WAVE_SOUND:
        raise BuildError("the leader's punch grunt is not a wave sound")
    volume = like["volume"]

    # New files: a wave archive holding his sounds, each once however many lines use it, and one
    # short silence; and the sounds that play them.
    files, archives, sounds = arc.count(bfsar.FILES), arc.count(bfsar.WAVE_ARCHIVES), arc.count(bfsar.SOUNDS)
    order: list[str] = []
    made: dict[str, tuple[list[int], int]] = {}
    for line in said:
        for take in line.takes:
            if take.name not in made:
                made[take.name] = (list(take.samples), take.rate)
                order.append(take.name)
            elif made[take.name] != (list(take.samples), take.rate):
                raise BuildError(f"two different sounds are both called {take.name}")
    waves = [bfsar.make_wave(*made[name]) for name in order]
    waves.append(bfsar.make_wave([0] * SILENCE, SILENCE_RATE))
    silence = len(order)
    wave_archive = bfsar.make_wave_archive(waves)
    wave_sounds = bfsar.make_wave_sounds([(WAVE_ARCHIVE_ITEM | archives, i) for i in range(len(waves))])
    new_group = grp.append([(files, wave_archive), (files + 1, wave_sounds)])

    # New entries in the archive. A file entry and a wave archive entry are copies of the pattern's
    # own (both say only "this is in the main group"); a sound entry is the pattern's with new values.
    holder = arc.wave_archive(0)
    if (
        arc.file(like["file"]) != arc.file(holder)
        or arc.file(holder)["groups"] != MAIN_GROUP
        or len(arc.raw(bfsar.WAVE_ARCHIVES, 0)) != 12
    ):
        raise BuildError("the files of the punch grunt are not recorded the way expected")
    file_entry = arc.raw(bfsar.FILES, like["file"])
    archive_entry = struct.pack("<I", files) + arc.raw(bfsar.WAVE_ARCHIVES, 0)[4:]
    plain = bfsar.with_fields(
        like["user"],
        volume_spread=0,
        pitch_spread=0,
        variants=0,
        weight=1,
        spaced=0,
        gap=0,
        gap_spread=0,
        kind=0,
    )
    entries, bases = [], []
    for line in said:
        several = len(line.takes) if len(line.takes) > 1 else 0
        bases.append(sounds + len(entries))
        for slot in range(max(slots.values()) + 1):
            for k, take in enumerate(line.takes):
                if slot == slots[LEADER]:
                    user = bfsar.with_fields(plain, weight=take.weight)
                    if k == 0:
                        user = bfsar.with_fields(
                            user,
                            variants=several,
                            spaced=1 if line.gap or line.gap_spread else 0,
                            gap=line.gap,
                            gap_spread=line.gap_spread,
                        )
                    fields = {"index": order.index(take.name), "volume": volume, "user": user}
                else:
                    fields = {"index": silence, "volume": 0, "priority": 0, "user": plain}
                if slot == 0 and k == 0:  # the family's first sound says how it is laid out
                    fields["user"] = bfsar.with_fields(fields["user"], kind=KIND, variants=several)
                entries.append(bfsar.wave_sound_entry(template, file=files + 1, **fields))
    new_archive = arc.grow(
        {
            bfsar.SOUNDS: entries,
            bfsar.FILES: [file_entry, file_entry],
            bfsar.WAVE_ARCHIVES: [archive_entry],
        },
    )

    # The new records of the table: the pattern's, each for another clip and frame and its own family.
    tag = re.search(rb'# anim tag\r\n\t\t\t"([^"]*)"', record).group(1)
    new_table = table[:-3].replace(
        b"# record num\r\n%d\r\n" % count,
        b"# record num\r\n%d\r\n" % (count + len(said)),
        1,
    )
    for i, line in enumerate(said):
        clip = line.clip.encode()
        if line.name.encode() in names:
            raise BuildError(f"the sound table already has a record named {line.name}")
        new = _swap_once(record, b'"' + PATTERN.encode() + b'"', b'"' + line.name.encode() + b'"')
        new = _swap_once(new, b"(STATE) " + tag + b" ", b"(STATE) " + clip + b" ")
        new = _swap_once(new, b'# anim tag\r\n\t\t\t"' + tag + b'"', b'# anim tag\r\n\t\t\t"' + clip + b'"')
        sound = b"0x%08x # the voice of leader %s" % (SOUND_ITEM | bases[i], LEADER.encode())
        new = _set_field(new, b"sound id", sound, old=rb"[^\r]*")
        new = _set_field(new, b"key id", b"%d" % (top_key + 1 + i))
        new = _set_field(new, b"frame", b"%d" % line.frame)
        new_table += new
    new_table += b"}\r\n"

    logger.info(
        "  %d sounds and a silence added as files %d and %d; sounds %#x to %#x play them",
        len(order),
        files,
        files + 1,
        sounds,
        sounds + len(entries) - 1,
    )
    return new_archive, new_group, new_table


def build(rom: Rom, romfs: RomFs, mod: Mod) -> None:
    """Write the sound archive, the group file and the command table into the mod folder."""
    logger.info("voice:")
    archive, group, commands = romfs.read(ARCHIVE), romfs.read(GROUP), romfs.read(COMMANDS)
    table = sarc.Sarc(yaz0.decompress(commands)).files()[TABLE]
    new_archive, new_group, new_table = add_voice(archive, group, table, lines(Voice(rom)))
    mod.write(ARCHIVE, new_archive)
    mod.write(GROUP, new_group)
    mod.write(COMMANDS, repack(commands, {TABLE: new_table}))
