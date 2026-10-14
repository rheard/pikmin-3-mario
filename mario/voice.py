"""
Mario's voice: what he says when he throws a Pikmin, punches, kicks, jumps and lands, as he falls
asleep when he is left standing, and wherever the game has Alph speak.

What is written
---------------
* `audio/audiores/CarrotAudio.bfsar`: the game's sound archive, with entries for his sounds, and
  Alph's own voice sent to them.
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
  the group file, and entries for them added to the sound archive. None of that touches a stock
  sound.
* **How loud.** As loud as the leaders' punch grunt, which is a shout. What he says while he falls
  asleep, left standing, is as quiet as what the leaders themselves say then.
* **In place of Alph's.** The rest of a leader's voice has no clip of his own to hang a record on:
  the game plays it by name from its code, or from records on clips the Pikmin share, and Alph's
  run of each family is recordings of Alph. Those sounds are the one thing of the game's own that
  is changed here: each is sent to one of Mario's recordings (`OWN`), and keeps everything else
  about how it is played, its loudness included. Only recordings of Alph are changed, and none
  that another leader is heard to make, so nobody else's voice is.
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
# Left standing, as his own game has him say it while he falls asleep: "I'm-a tired" as his
# scratching begins, his yawn as his yawn begins, and a snore in and a snore out on frames 2 and 20
# of each round of his sleeping sitting. Each is a clip, the frame of it, the sound, and the game's
# own sound it is as loud as: a leader's yawn, or the other thing a leader left standing says, which
# are both much quieter than a shout. The snores are as loud as his other lines; they are faint
# recordings.
IDLE = (
    ("m_scratch", 50, "ima_tired", "WSD_Player{}Vo_LookFor"),
    ("m_yawn", 0, "yawning", "WSD_Player{}Vo_FreeAkubi"),
    ("m_sit", 66, "snoring1", None),
    ("m_sit", 84, "snoring2", None),
)
# The rest of the leader's voice. The game plays most of what a leader says by name from its code,
# or from a record of its own on a clip the leaders share with the Pikmin, and each of those ends at
# a sound of the archive that is a recording of Alph. Each of these is made to play a recording of
# Mario's instead: (when it is heard, ((the archive's name for the sound, his sound), ...)).
#
# Which of his: what he says in his own game for the nearest thing. "Okey-dokey" where he is
# chosen; "whoa" where he is startled; his cry when something hits him, "oof", "mamma mia" for a
# blast and "d'oh" when he is squashed; his three cries from a triple jump when he is thrown; small
# grunts where the leaders make small noises; "yah", "wah" and "hoo", his jump's three, wherever a
# leader says something as he jumps or dodges; "here we go", his cry from a cannon, down a slide
# and up a geyser; "hello" as he waves; his long falling cry into a pit; and his yawn.
#
# Left as they are, of the recordings the archive names for Alph:
#   * the cutscenes', which play beside a model of Alph that this mod does not replace;
#   * the search of a leader left standing (`WSD_PlayerAVo_LookFor`), which is not his alone: the
#     Pikmin's fretting is a family with no runs for the leaders, and Louie's and Olimar's places in
#     it fall on these (see `bfsar.TYPE_SLOTS`). Changing them would change how those two fret;
#   * the punch grunt, which he no longer makes: his punches and kick have a voice of their own;
#   * the ones that are silent in the game's own files, and the ones nothing plays for him.
# Four of those below can be reached the same way by another leader, through families made for
# things only a Pikmin does (climbing a stalk, throwing what it carries), and are changed all the
# same: the three of the clip `drop`, and the third of the pit.
OWN = (
    # (the second is the same recording again, which no record of the sound table plays)
    (
        "when the player changes to him",
        (("WSD_Player{}Vo_Change", "okey_dokey"), ("WSD_Player{}Vo_Change_Multi", "okey_dokey")),
    ),
    ("as he notices something and joins a party", (("WSD_PikminVo_JoinPartyPlayer{}", "whoa"),)),
    (
        "when he is hit, caught by a blast or squashed",
        (
            ("WSD_PikminVo_DamagePlayer{}", "attacked"),
            ("WSD_PikminVo_DamagePlayer{}2", "ooof"),
            ("WSD_PikminVo_BombDamagePlayer{}", "mama_mia"),
            ("WSD_PikminVo_BombDamagePlayer{}2", "ooof"),
            ("WSD_Player{}Vo_Squash", "doh"),
            ("WSD_Player{}Vo_Squash2", "ooof"),
            ("WSD_Player{}Vo_Squash3", "attacked"),
        ),
    ),
    (
        "when another leader throws him",
        (
            ("WSD_PikminVo_ThrownPlayer{}", "yahoo"),
            ("WSD_PikminVo_ThrownPlayer{}2", "waha"),
            ("WSD_PikminVo_ThrownPlayer{}3", "yippee"),
        ),
    ),
    (
        "when he is dismissed from a party",
        (
            ("WSD_PikminVo_BeFreePlayer{}", "uh2"),
            ("WSD_PikminVo_BeFreePlayer{}2", "hrmm"),
            ("WSD_PikminVo_BeFreePlayer{}3", "uh"),
        ),
    ),
    # (the leader being played makes the dodge's own sounds and says nothing)
    (
        "when a party he is in dodges",
        (
            ("WSD_PikminVo_StepOutPlayer{}", "jump_hoo"),
            ("WSD_PikminVo_StepOut2Player{}", "jump_yah"),
            ("WSD_PikminVo_StepOut3Player{}", "jump_wah"),
        ),
    ),
    (
        "as he sets off down a slide",
        (
            ("WSD_Player{}Vo_SliderStart", "here_we_go"),
            ("WSD_Player{}Vo_SliderStart2", "yippee"),
            ("WSD_Player{}Vo_SliderStart3", "waha"),
        ),
    ),
    (
        "on a trampoline, and carried up by a geyser",
        (
            ("WSD_Player{}Vo_Trampoline", "hoohoo"),
            ("WSD_Player{}Vo_Trampoline2", "yahoo"),
            ("WSD_Player{}Vo_RideOnGeyser", "here_we_go"),
            ("WSD_Player{}Vo_RideOnGeyser2", "waha"),
        ),
    ),
    (
        "as he jumps: the clips `drop_lotus`, `jump` and `drop`",
        (
            ("WSD_PikminVo_JumpToIndianLotusPlayer{}", "jump_yah"),
            ("WSD_PikminVo_JumpToIndianLotusPlayer{}2", "jump_wah"),
            ("WSD_PikminVo_JumpToIndianLotusPlayer{}3", "jump_hoo"),
            ("WSD_PikminVo_JumpPlayer{}1", "jump_yah"),
            ("WSD_PikminVo_JumpPlayer{}2", "jump_wah"),
            ("WSD_PikminVo_JumpPlayer{}3", "jump_hoo"),
            ("WSD_Player{}Vo_Drop", "jump_yah"),
            ("WSD_Player{}Vo_Drop2", "jump_wah"),
            ("WSD_Player{}Vo_Drop3", "jump_hoo"),
        ),
    ),
    (
        "when he jumps for joy, and with the ultra-spicy spray",
        (
            ("WSD_PikminVo_GladJumpPlayer{}", "haha"),
            ("WSD_PikminVo_GladJumpPlayer{}2", "yippee"),
            ("WSD_PikminVo_GekikaraJumpPlayer{}", "yahoo"),
        ),
    ),
    (
        "as he waves",
        (
            ("WSD_PikminVo_WaveOnesHandPlayer{}", "hello"),
            ("WSD_PikminVo_WaveOnesHandPlayer{}2", "hello"),
            ("WSD_PikminVo_WaveOnesHandPlayer{}3", "hello"),
        ),
    ),
    (
        "as he pumps his fist",
        (
            ("WSD_Player{}Vo_Gattsu", "jump_hoo"),
            ("WSD_Player{}Vo_Gattsu2", "jump_yah"),
            ("WSD_Player{}Vo_Gattsu3", "jump_wah"),
        ),
    ),
    (
        "as he falls into a pit",
        (
            ("WSD_PikminVo_FallIntoHellPlayer{}", "waaaooow"),
            ("WSD_PikminVo_FallIntoHellPlayer{}2", "waaaooow"),
            ("WSD_PikminVo_FallIntoHellPlayer{}3", "waaaooow"),
        ),
    ),
    (
        "caught in a net",
        (("WSD_Player{}Vo_Net", "mama_mia"), ("WSD_Player{}Vo_Net2", "whoa"), ("WSD_Player{}Vo_Net3", "uh")),
    ),
    # (the second is spelled so in the game's archive)
    (
        "knocked back, and as a bubble he is caught in bursts",
        (
            ("WSD_PikminVo_ReboundPlayer{}", "uh"),
            ("WSD_PikminrVo_ReboundPlaye{}2", "ooof"),
            ("WSD_PikminVo_ReboundPlayer{}3", "uh2"),
            ("WSD_PikminVo_BubbleBreakPlayer{}", "whoa"),
            ("WSD_PikminVo_BubbleBreakPlayer{}2", "uh"),
        ),
    ),
    (
        "as he gets up after being knocked down",
        (
            ("WSD_PikminVo_GetUpFromElectricShockPlayer{}", "hrmm"),
            ("WSD_PikminVo_GetUpFromElectricShockPlayer{}2", "uh2"),
            ("WSD_PikminVo_GetUpFromElectricShockPlayer{}3", "uh"),
        ),
    ),
    # (the first of the game's three for this is silent, and the one it picks most often: left as it is)
    (
        "running about in a panic",
        (("WSD_PikminVo_RunAwayPlayer{}2", "mama_mia"), ("WSD_PikminVo_RunAwayPlayer{}3", "whoa")),
    ),
    (
        "with the leaders' own yawn, should anything still have him play it",
        (("WSD_Player{}Vo_FreeAkubi", "yawning"), ("WSD_Player{}Vo_FreeAkubi2", "yawning")),
    ),
)
# The jumps' sounds are five seconds in all, and every sound of the game's is loaded into one pool
# of memory at start-up whose spare room is not known. They, and everything added since, are kept
# at half the rate the first ones are made at, which halves their size: his samples are recorded at
# 13 to 19 kHz, so next to nothing that was in them is above what 16 kHz holds.
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
    more at random. `like` names one of the game's own sounds that it is to be as loud as; without
    it, it is as loud as the leaders' punch grunt.
    """

    def __init__(
        self,
        clip: str,
        frame: int,
        takes: list[Take],
        gap: int = 0,
        gap_spread: int = 0,
        like: str | None = None,
    ):
        self.clip, self.frame, self.takes, self.gap, self.gap_spread = clip, frame, list(takes), gap, gap_spread
        self.like = like
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
        A line for the throw, then one for each move, then one for each jump, then those of what
        he says left standing.
    """
    clip, frame, said = THROW
    takes = [Take(name, weight, *_sound(voice, name)) for name, weight in said]
    out = [Line(clip, frame, takes, THROW_GAP, THROW_GAP_SPREAD)]
    for clip, name in MOVES:
        out.append(Line(clip, 0, [Take(name + "_whole", 1, *_sound(voice, name, whole=True))]))
    for clip, said in JUMPS:
        out.append(Line(clip, 0, [kept(voice, name, weight) for name, weight in said]))
    for clip, frame, name, like in IDLE:
        out.append(Line(clip, frame, [kept(voice, name)], like=like and like.format(LEADER)))
    return out


def _sound(voice: Voice, name: str, *, whole: bool = False) -> tuple[list[int], int]:
    sound = voice.play(name, whole=whole)
    logger.info("  %-12s %.2f s%s", name, sound.seconds, ", with its swish" if whole else "")
    return sound.samples, sound.rate


def kept(voice: Voice, name: str, weight: int = 1) -> Take:
    """
    One of his sounds at half the rate it is made at.

    Returns:
        The take, under the sound's own name.

    Raises:
        BuildError: If the sound is not made at the rate `halve` expects.
    """
    samples, rate = _sound(voice, name)
    if rate != 2 * JUMP_RATE:
        raise BuildError(f"{name} is made at {rate} Hz, not twice {JUMP_RATE}")
    return Take(name, weight, halve(samples), JUMP_RATE)


def own_voice(voice: Voice) -> list[tuple[str, Take]]:
    """
    What he says in place of the leader's own voice (`OWN`), rendered from the ROM.

    Returns:
        The archive's name for each of the game's sounds, and the take it is to play.
    """
    made: dict[str, Take] = {}
    out = []
    for _, sounds in OWN:
        for name, said in sounds:
            if said not in made:
                made[said] = kept(voice, said)
            out.append((name.format(LEADER), made[said]))
    return out


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


def stock_sound(names: dict[str, int], name: str) -> int:
    """
    One of the game's own sounds, by the name the sound archive gives it.

    Returns:
        The sound's number.

    Raises:
        BuildError: If the archive names no sound so.
    """
    item = names.get(name, 0)
    if item >> 24 != SOUND_ITEM >> 24:
        raise BuildError(f"the sound archive has no sound named {name}")
    return item & 0xFFFFFF


def _swap_once(text: bytes, old: bytes, new: bytes) -> bytes:
    if text.count(old) != 1:
        raise BuildError(f"the pattern record does not hold {old!r} exactly once")
    return text.replace(old, new)


def _set_field(record: bytes, field: bytes, value: bytes, old: bytes = rb"\d+") -> bytes:
    """A record with the number on the line after `# <field>` replaced (or whatever there matches `old`)."""
    return re.sub(rb"(# " + field + rb"\r\n\t\t\t)" + old, lambda match: match.group(1) + value, record, count=1)


def add_voice(
    archive: bytes,
    group: bytes,
    table: bytes,
    said: list[Line],
    own: list[tuple[str, Take]] | None = None,
) -> tuple[bytes, bytes, bytes]:
    """
    Add his voice to the game's three files: a family of sounds and a record for each line.

    `own` is the game's own sounds that are to play a take of his instead, by the names the
    archive gives them. Such a sound keeps its place and everything that says how it is played
    (how loud, on which player, how it is picked among its fellows); only the recording changes.

    Returns:
        The new sound archive, group file and sound table.

    Raises:
        BuildError: If the game's files are not put together the way this expects.
    """
    arc, grp = bfsar.Archive(archive), bfsar.Group(group)
    count, record, top_key, names = table_parts(table)
    own = own or []
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
    for _, take in own:
        if made.setdefault(take.name, (list(take.samples), take.rate)) != (list(take.samples), take.rate):
            raise BuildError(f"two different sounds are both called {take.name}")
        if take.name not in order:
            order.append(take.name)
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
    named = bfsar.labels(archive)
    entries, bases = [], []
    for line in said:
        several = len(line.takes) if len(line.takes) > 1 else 0
        loud = arc.sound(stock_sound(named, line.like))["volume"] if line.like else volume
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
                    fields = {"index": order.index(take.name), "volume": loud, "user": user}
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

    # The game's own sounds that are to play his: each one's entry, where it now is, sent to the new
    # file and the wave in it. The small changes of loudness and pitch the game makes at random to its
    # own voice lines are taken off, as his other lines have none.
    grown, patched, done = bfsar.Archive(new_archive), bytearray(new_archive), set()
    for name, take in own:
        index = stock_sound(named, name)
        was = arc.sound(index)
        if was["kind"] != bfsar.WAVE_SOUND or index in done:
            raise BuildError(f"{name} is not a recording, or is asked for twice")
        done.add(index)
        at, size = grown.entry(bfsar.SOUNDS, index)
        try:
            patched[at : at + size] = bfsar.wave_sound_entry(
                arc.raw(bfsar.SOUNDS, index),
                file=files + 1,
                index=order.index(take.name),
                user=bfsar.with_fields(was["user"], volume_spread=0, pitch_spread=0),
            )
        except ValueError as error:
            raise BuildError(f"{name}: {error}") from error
    new_archive = bytes(patched)

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
    if own:
        logger.info("  %d of the leader's own sounds play his in place of theirs", len(own))
    return new_archive, new_group, new_table


def build(rom: Rom, romfs: RomFs, mod: Mod) -> None:
    """Write the sound archive, the group file and the command table into the mod folder."""
    logger.info("voice:")
    archive, group, commands = romfs.read(ARCHIVE), romfs.read(GROUP), romfs.read(COMMANDS)
    table = sarc.Sarc(yaz0.decompress(commands)).files()[TABLE]
    voice = Voice(rom)
    new_archive, new_group, new_table = add_voice(archive, group, table, lines(voice), own_voice(voice))
    mod.write(ARCHIVE, new_archive)
    mod.write(GROUP, new_group)
    mod.write(COMMANDS, repack(commands, {TABLE: new_table}))
