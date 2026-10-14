"""
Read Mario's voice out of a Super Mario 64 ROM.

Nothing from the ROM is kept anywhere: your own copy is read each time.

The game keeps its sounds in three files inside the ROM: the instrument banks (which sample an
instrument plays, and how it is tuned), the samples themselves (ADPCM, nine bytes for sixteen
samples) and the sequences. Sound effects are sequence 0: a small program with one channel for each
family of sounds, and Mario's voice is channel 2. A sound number picks an entry of that channel's
table; the entry chooses a bank and an instrument and plays one note. `Voice(path)` follows exactly
that path. `sound(n)` is the decoded sample, untouched; `play(n)` is the sound as the game makes it,
at 32 kHz: the note's pitch (most samples are played a few semitones off the rate they were recorded
at), any slide in pitch while it plays (the jump's "yah" rises a semitone, the punch's "hoo" falls
four), the instrument's envelope, and the cut the game makes when the note ends before the sample
does. The three files are found by their headers, not by address, so other releases should work,
though only the US one has been read.

An entry may start something else after the voice: the punches and the kick add the swish of the
arm or leg, an instrument from another bank, one tick later. `play(n, whole=True)` follows the
entry through its wait and mixes that in; without it only the voice is given.

A sound may also be several notes in one breath ("ya-hoo"): the sample is not started again for the
second note, it goes on at a new pitch. `play` follows the one way of sliding between such notes
that these sounds use and refuses any other.

What is left out: sounds that loop, and notes that are let go before their time is up.

The formats follow the public documentation of the game.
"""

from __future__ import annotations

import math
import struct

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from mario.errors import BuildError
from mario.sm64.rom import Rom

if TYPE_CHECKING:
    from pathlib import Path

OUTPUT_RATE = 32000  # the game mixes at 32 kHz; a note at pitch 39 with tuning 1.0 plays at this rate
UPDATES = 240.0  # times a second the game moves pitch and loudness: four for each of 60 frames
TEMPO_DIVISOR = 14360  # the game's tempo counter gains tempo*48 each update and ticks when it passes this
VOICE_CHANNEL = 2  # the sound-effect sequence's channel for Mario's voice

# Sound numbers in the voice channel worth a name here: what he says, not when he says it.
SOUNDS = {
    "jump_yah": 0x00, "jump_wah": 0x01, "jump_hoo": 0x02, "hoohoo": 0x03, "yahoo": 0x04, "wah2": 0x07,
    "yawning": 0x0D, "snoring1": 0x0E, "snoring2": 0x0F,
    "haha": 0x11, "punch_yah": 0x1E, "punch_hoo": 0x1F, "punch_wah": 0x24, "waha": 0x2E, "yippee": 0x2F,
    "ima_tired": 0x37,
}  # fmt: skip

# Bytes of operand for the commands of a script that take a fixed number: a channel's (those above
# 0xC0 by the command, those below by its high four bits), the sequence's own, and a layer's.
_CHANNEL = {
    0xFC: 2, 0xF8: 1, 0xF7: 0, 0xF6: 0, 0xFB: 2, 0xFA: 2, 0xF9: 2, 0xF5: 2, 0xF3: 0, 0xF2: 1, 0xF1: 0,
    0xC2: 2, 0xC5: 0, 0xC1: 1, 0xC3: 0, 0xC4: 0, 0xDF: 1, 0xE0: 1, 0xDE: 2, 0xD3: 1, 0xDD: 1, 0xDC: 1,
    0xDB: 1, 0xDA: 2, 0xD9: 1, 0xD8: 1, 0xD7: 1, 0xE2: 3, 0xE1: 3, 0xE3: 1, 0xD6: 1, 0xD4: 1, 0xC6: 1,
    0xC7: 3, 0xC8: 1, 0xC9: 1, 0xCC: 1, 0xCA: 1, 0xCB: 2, 0xD0: 1, 0xD1: 1, 0xD2: 1, 0xE4: 0, 0xFE: 0,
}  # fmt: skip
_CHANNEL_LOW = {
    0x00: 0, 0x10: 2, 0x20: 0, 0x30: 1, 0x40: 1, 0x50: 0, 0x60: 1, 0x70: 0, 0x80: 0, 0x90: 2, 0xA0: 0,
    0xB0: 0,
}  # fmt: skip
_SEQUENCE = {
    0xFC: 2, 0xF8: 1, 0xF7: 0, 0xFB: 2, 0xFA: 2, 0xF9: 2, 0xF5: 2, 0xF2: 1, 0xF1: 0, 0xDF: 1, 0xDE: 1,
    0xDD: 1, 0xDC: 1, 0xDB: 1, 0xDA: 1, 0xD7: 2, 0xD6: 2, 0xD5: 1, 0xD4: 0, 0xD3: 1, 0xD2: 2, 0xD1: 2,
    0xD0: 1, 0xCC: 1, 0xC9: 1, 0xC8: 1, 0xFE: 0,
}  # fmt: skip
_LAYER = {
    0xFC: 2, 0xF8: 1, 0xF7: 0, 0xFB: 2, 0xC1: 1, 0xCA: 1, 0xC2: 1, 0xC9: 1, 0xC4: 0, 0xC5: 0, 0xC6: 1,
    0xC8: 0,
}  # fmt: skip


def _var(d: bytes, p: int) -> tuple[int, int]:
    """A length in a script, and where the script goes on: one byte, or two when the first has its top bit set."""
    v = d[p]
    if v & 0x80:
        return (v & 0x7F) << 8 | d[p + 1], p + 2
    return v, p + 1


def _pitch(note: int) -> float:
    """How fast a note plays its sample, against pitch 39: a semitone is a twelfth of an octave."""
    return 2.0 ** ((note - 39) / 12.0)


@dataclass
class Note:
    """One note of a layer."""

    pitch: int
    velocity: int
    ticks: int  # how long it lasts
    gate: int  # how early it is let go
    slide: tuple[int, int, int] | None  # a slide in pitch: (how, the other pitch, how fast)
    legato: bool  # run together with the notes around it


@dataclass
class Instrument:
    """The sample an instrument plays at some pitch: where it is, how it is tuned and how its loudness moves."""

    address: int
    size: int
    book: tuple[int, int, tuple[int, ...]]  # the ADPCM codebook: order, predictors, their numbers
    loops: bool
    tuning: float
    release: int
    envelope: list[tuple[int, int]]  # (updates, level) steps


@dataclass
class Extra:
    """Something a sound's entry starts besides the voice (a punch's swish): which layer, when, and what it plays."""

    layer: int
    bank: int
    instrument: int
    tick: int
    note: Note
    seconds: float


@dataclass
class Sound:
    """
    One voice sound: `samples` (signed 16-bit, mono) at `rate` Hz.

    `number` is its place in the voice channel's table, and `name` what it was asked for by if that
    was a name. `Voice.sound` adds what the entry plays (`bank`, `instrument`, `notes`, how many
    `layers` it starts, whether it goes on to something `later`) and the instrument as read
    (`instrument_data`). `Voice.play` adds `source` (the untouched sample it began from), `cut`
    (the note ended before the sample did) and `extra` (what else the entry started, when asked
    for the whole).
    """

    samples: list[int]
    rate: int
    number: int
    name: str | None = None
    bank: int | None = None
    instrument: int | None = None
    notes: list[Note] = field(default_factory=list)
    layers: int = 0
    later: bool = False
    instrument_data: Instrument | None = None
    source: Sound | None = None
    cut: bool = False
    extra: list[Extra] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        """How long it lasts."""
        return len(self.samples) / float(self.rate)


@dataclass
class _Channel:
    """What a channel's script has set by the point it was run to."""

    bank: int
    instrument: int = 0
    transpose: int = 0
    table: int | None = None  # where its table of sounds is in the sequence
    large: bool = False  # its layers write their notes in the long form
    layers: dict[int, int] = field(default_factory=dict)  # {layer: where its script is}
    starts: list[tuple[int, int, int, int, int]] = field(default_factory=list)  # each layer started, in order
    clock: int = 0  # ticks waited so far, when waits are followed
    later: bool = False  # it stopped at a wait or a test with more to come


def _transpose(note: Note, by: int) -> None:
    """Move a note, and the pitch its slide names, by the channel's transposition."""
    note.pitch += by
    if note.slide:
        note.slide = (note.slide[0], note.slide[1] + by, note.slide[2])


def _sample_pitch(note: Note) -> int:
    """The pitch an instrument's sample is picked by: with a slide, the higher of its two pitches."""
    return max(note.pitch, note.slide[1]) if note.slide else note.pitch


class Voice:
    """Mario's voice: the sounds of channel 2 of the ROM's sound-effect sequence."""

    def __init__(self, path: str | Path | Rom) -> None:
        self.rom = rom = path if isinstance(path, Rom) else Rom(path)
        d = rom.d
        self.ctl, banks = self._seq_file(1)
        self.tbl, tables = self._seq_file(2, len(banks), after=self.ctl)
        self.seq, seqs = self._seq_file(3, after=self.tbl)
        self.banks, self.tables, self.seqs = banks, tables, seqs
        # The list of banks each sequence may use follows the sequences: one 16-bit offset per sequence.
        sets = self.seq + max(o + n for o, n in seqs)
        sets = (sets + 15) & ~15
        if struct.unpack_from(">H", d, sets)[0] != 2 * len(seqs):
            raise BuildError(f"the bank lists were not found after the sequences (at {sets:#x})")
        self.sets = sets
        self.base = self.seq + self.seqs[0][0]  # where sequence 0, the sound effects, starts
        self.tempo = 120  # beats a minute; a beat is 48 ticks
        self._find_channel(VOICE_CHANNEL)

    def _seq_file(self, revision: int, count: int | None = None, after: int = 0) -> tuple[int, list[tuple[int, int]]]:
        """Find one of the three audio files by its header: (address, [(offset, length)])."""
        d = self.rom.d
        hits = []
        for p in range((after + 15) & ~15, len(d) - 16, 16):
            if d[p] or d[p + 1] != revision or d[p + 2]:
                continue
            n = d[p + 3]
            if not n or (count is not None and n != count):
                continue
            first = (4 + 8 * n + 15) & ~15
            if struct.unpack_from(">I", d, p + 4)[0] != first:
                continue
            entries = [struct.unpack_from(">II", d, p + 4 + 8 * i) for i in range(n)]
            if any(o % 16 or o < first or not size or p + o + size > len(d) for o, size in entries):
                continue  # (banks may share samples, so offsets can repeat)
            hits.append((p, entries))
        if len(hits) != 1:
            raise BuildError(f"expected one audio file of kind {revision} in this ROM, found {len(hits)}")
        return hits[0]

    # -- the sound-effect sequence ---------------------------------------------------------------

    def _bank_of(self, sequence: int, index: int) -> int:
        """The bank `chan_setbank index` selects: the lists are stored last first."""
        d = self.rom.d
        p = self.sets + struct.unpack_from(">H", d, self.sets + 2 * sequence)[0]
        count = d[p]
        if index >= count:
            raise BuildError(f"sequence {sequence} has {count} banks, not {index + 1}")
        return d[p + count - index]

    def _find_channel(self, number: int) -> None:
        """Find a channel of sequence 0 and run its opening, for the table of sounds and its defaults."""
        d, base = self.rom.d, self.base
        p, start = base, None
        for _ in range(200):
            cmd = d[p]
            if cmd in {0xFF, 0xFD}:
                break
            if cmd >= 0xC0:
                if cmd not in _SEQUENCE:
                    raise BuildError(f"sequence command {cmd:#04x} at {p - base:#x} is not handled")
                if cmd == 0xFB and struct.unpack_from(">H", d, p + 1)[0] <= p - base:
                    break
                if cmd == 0xDD:
                    self.tempo = d[p + 1]
                p += 1 + _SEQUENCE[cmd]
            elif cmd & 0xF0 == 0x90:
                if cmd & 0xF == number:
                    start = struct.unpack_from(">H", d, p + 1)[0]
                p += 3
            else:
                p += 1
        if start is None:
            raise BuildError(f"channel {number} of the sound-effect sequence was not found")
        state = _Channel(bank=self._bank_of(0, 0))
        self._run_channel(start, state, opening=True)
        if state.table is None:
            raise BuildError(f"channel {number} never sets its table of sounds")
        self.defaults = state
        self.table = state.table
        # The table has no length of its own: it runs up to the first routine one of its entries names.
        first = min(struct.unpack_from(">H", d, base + self.table + 2 * i)[0] for i in range(16))
        self.count = (first - self.table) // 2
        if not 16 <= self.count <= 256:
            raise BuildError(f"channel {number}'s table of sounds has an unlikely length ({self.count})")

    def _run_channel(self, at: int, state: _Channel, *, opening: bool = False, follow: bool = False) -> None:
        """
        Run channel commands from `at` until the routine ends (or, in the opening, until it loops).

        With `follow`, waits are not the end: time is counted in ticks and the routine is run on
        through them, so that `state.starts` lists every layer it starts and when, as (layer,
        address, bank, instrument, tick). That needs the few commands the waiting routines are
        made of: a value the script keeps, tests on it, counted loops, and a write of the value
        into the script itself (the punches set the length of their wait that way).

        Raises:
            BuildError: If it meets a command it does not know, or never ends.
        """
        d, base = self.rom.d, self.base
        p = base + at
        stack, loops, written = [], [], {}
        value = 0
        for _ in range(400):
            cmd = d[p]
            if cmd == 0xFF:
                if not stack:
                    return
                p = stack.pop()
                continue
            if cmd == 0xFD:
                if follow:
                    ticks, p = _var(d, p + 1)
                    state.clock += ticks
                    continue
                if not opening:
                    state.later = True
                    return
                p = _var(d, p + 1)[1]
                continue
            if cmd > 0xC0:
                if cmd not in _CHANNEL:
                    raise BuildError(f"channel command {cmd:#04x} at {p - base:#x} is not handled")
                arg = bytes(written.get(q - base, d[q]) for q in range(p + 1, p + 1 + _CHANNEL[cmd]))
                nxt = p + 1 + _CHANNEL[cmd]
                if follow and cmd in {0xCC, 0xC7, 0xF8, 0xF7, 0xFE, 0xF5, 0xFA, 0xF9}:
                    if cmd == 0xCC:  # keep a value
                        value = arg[0]
                    elif cmd == 0xC7:  # write the value, plus this, into the script
                        written[struct.unpack(">H", arg[1:])[0]] = (value + arg[0]) & 0xFF
                    elif cmd == 0xF8:  # do what follows this many times
                        loops.append([arg[0], nxt])
                    elif cmd == 0xF7:
                        loops[-1][0] = (loops[-1][0] - 1) & 0xFF
                        if loops[-1][0]:
                            nxt = loops[-1][1]
                        else:
                            loops.pop()
                    elif cmd == 0xFE:  # wait one tick
                        state.clock += 1
                    elif (
                        (cmd == 0xF5 and value < 0x80)
                        or (cmd == 0xFA and value == 0)
                        or (cmd == 0xF9 and value >= 0x80)
                    ):  # a jump if the value is not negative, zero, negative
                        nxt = base + struct.unpack(">H", arg)[0]
                    p = nxt
                    continue
                if cmd == 0xC6:
                    state.bank = self._bank_of(0, arg[0])
                elif cmd == 0xC1:
                    state.instrument = arg[0]
                elif cmd == 0xDB:
                    state.transpose = struct.unpack("b", arg)[0]
                elif cmd == 0xC2:
                    state.table = struct.unpack(">H", arg)[0]
                elif cmd == 0xC4:
                    state.large = True
                elif cmd == 0xC3:
                    state.large = False
                elif cmd == 0xFC:
                    stack.append(nxt)
                    nxt = base + struct.unpack(">H", arg)[0]
                elif cmd == 0xFB:
                    if opening:
                        return  # the opening ends by jumping into the shared loop
                    nxt = base + struct.unpack(">H", arg)[0]
                elif cmd in {0xFA, 0xF9, 0xF5, 0xF8, 0xF7, 0xF6, 0xE4, 0xC5, 0xF3, 0xFE} and not opening:
                    # A wait or a test: whatever follows happens later (the punches start their swish
                    # a few frames after the voice). What has been set up by now is what plays first.
                    state.later = True
                    return
                p = nxt
            else:
                low = cmd & 0xF0
                if low not in _CHANNEL_LOW:
                    raise BuildError(f"channel command {cmd:#04x} at {p - base:#x} is not handled")
                if low == 0x90:  # start a layer
                    layer = cmd & 0xF
                    state.layers[layer] = struct.unpack_from(">H", d, p + 1)[0]
                    state.starts.append((layer, state.layers[layer], state.bank, state.instrument, state.clock))
                elif low == 0x80:
                    value = 0xFF  # "what has the game asked of this channel since?": nothing (-1)
                p += 1 + _CHANNEL_LOW[low]
        raise BuildError(f"the channel routine at {at:#x} does not end")

    def _run_layer(self, at: int, state: _Channel) -> list[Note]:
        """Run a layer to its end, for the notes it plays. The layer may choose the instrument in `state`."""
        d, base = self.rom.d, self.base
        p = base + at
        stack, notes = [], []
        transpose, slide, ticks, legato = 0, None, 0, False
        for _ in range(200):
            cmd = d[p]
            if cmd == 0xFF:
                if not stack:
                    return notes
                p = stack.pop()
            elif cmd == 0xC0:
                p = _var(d, p + 1)[1]
            elif cmd < 0xC0:
                if not state.large:
                    raise BuildError(f"the layer at {at:#x} uses short notes, which are not handled")
                pitch, kind = cmd & 0x3F, cmd >> 6
                p += 1
                if kind != 2:  # kinds 0 and 1 carry their length; 2 reuses the last
                    ticks, p = _var(d, p)
                velocity = d[p]
                p += 1
                gate = 0
                if kind != 1:  # kinds 0 and 2 also say how early to let go
                    gate = d[p]
                    p += 1
                notes.append(Note(pitch + transpose, velocity, ticks, gate, slide, legato))
            elif cmd == 0xC7:  # a slide in pitch: how, towards which note, how fast
                mode, target = d[p + 1], d[p + 2] + transpose
                p += 3
                if mode & 0x80:
                    time, p = d[p], p + 1
                else:
                    time, p = _var(d, p)
                slide = (mode, target, time)
            elif cmd == 0xC3:
                p = _var(d, p + 1)[1]
            elif 0xD0 <= cmd <= 0xEF:
                p += 1
            elif cmd in _LAYER:
                arg = d[p + 1 : p + 1 + _LAYER[cmd]]
                nxt = p + 1 + _LAYER[cmd]
                if cmd == 0xC6:
                    state.instrument = arg[0]
                elif cmd == 0xC2:
                    transpose = struct.unpack("b", arg)[0]
                elif cmd == 0xC8:
                    slide = None
                elif cmd == 0xC4:
                    legato = True
                elif cmd == 0xC5:
                    legato = False
                elif cmd == 0xFC:
                    stack.append(nxt)
                    nxt = base + struct.unpack(">H", arg)[0]
                elif cmd == 0xFB:
                    nxt = base + struct.unpack(">H", arg)[0]
                p = nxt
            else:
                raise BuildError(f"layer command {cmd:#04x} at {p - base:#x} is not handled")
        raise BuildError(f"the layer at {at:#x} does not end")

    def describe(self, number: int) -> dict | None:
        """
        What a sound number plays, without decoding it.

        `notes` is what its first layer plays; `later` says the entry goes on to start something else
        after a wait (the punches add a swish from another bank), which only `play(whole=True)` follows.

        Returns:
            Its `number`, `bank`, `instrument`, `notes`, count of `layers` and `later`, or None if the
            entry plays nothing.
        """
        d = self.rom.d
        entry = struct.unpack_from(">H", d, self.base + self.table + 2 * number)[0]
        state = replace(self.defaults, layers={}, starts=[], clock=0)
        self._run_channel(entry, state)
        if not state.layers:
            return None
        notes = self._run_layer(state.layers[min(state.layers)], state)
        if not notes:
            return None
        for note in notes:
            _transpose(note, state.transpose)
        return {
            "number": number,
            "bank": state.bank,
            "instrument": state.instrument,
            "notes": notes,
            "layers": len(state.layers),
            "later": state.later,
        }

    # -- banks and samples -----------------------------------------------------------------------

    def _instrument(self, bank: int, number: int, pitch: int) -> Instrument:
        """An instrument's sample for a pitch: where it is, how it is tuned, how its loudness moves."""
        d = self.rom.d
        head = self.ctl + self.banks[bank][0]
        count = struct.unpack_from(">I", d, head)[0]
        if number >= count:
            raise BuildError(f"bank {bank} has {count} instruments, not {number + 1}")
        body = head + 16
        ptr = struct.unpack_from(">I", d, body + 4 + 4 * number)[0]
        if not ptr:
            raise BuildError(f"bank {bank} has no instrument {number}")
        inst = body + ptr
        low, high = d[inst + 1], d[inst + 2]
        which = 0 if pitch < low else 1 if pitch <= high else 2
        sample_ptr, tuning = struct.unpack_from(">If", d, inst + 8 + 8 * which)
        if not sample_ptr:
            raise BuildError(f"instrument {number} of bank {bank} has no sample for pitch {pitch}")
        sample = body + sample_ptr
        addr, loop_ptr, book_ptr, size = struct.unpack_from(">IIII", d, sample + 4)
        loop = struct.unpack_from(">IIi", d, body + loop_ptr)
        book = body + book_ptr
        order, predictors = struct.unpack_from(">ii", d, book)
        if not (0 < order <= 8 and 0 < predictors <= 16):
            raise BuildError(f"the codebook at {book:#x} is not one (order {order}, {predictors} predictors)")
        table = struct.unpack_from(f">{8 * order * predictors}h", d, book + 8)
        # The envelope: (updates, level) steps, ended by a negative count (hold, stop, or go back).
        env = body + struct.unpack_from(">I", d, inst + 4)[0]
        envelope = []
        for i in range(16):
            delay, level = struct.unpack_from(">hh", d, env + 4 * i)
            if delay <= 0:
                break
            envelope.append((delay, level))
        return Instrument(
            address=self.tbl + self.tables[bank][0] + addr,
            size=size,
            book=(order, predictors, table),
            loops=loop[2] != 0,
            tuning=tuning,
            release=d[inst + 3],
            envelope=envelope,
        )

    def sound(self, which: str | int) -> Sound:
        """
        The sample a voice sound starts with, decoded and otherwise untouched: a name from `SOUNDS`, or a number.

        It is given at the rate its first note plays it at. `play` is the sound as the game makes it.

        Returns:
            The sample, with what the sound's entry plays.

        Raises:
            BuildError: If the entry plays nothing.
        """
        number = SOUNDS[which] if isinstance(which, str) else which
        what = self.describe(number)
        if what is None:
            raise BuildError(f"voice sound {number:#x} plays nothing")
        note = what["notes"][0]
        inst = self._instrument(what["bank"], what["instrument"], _sample_pitch(note))
        samples = decode(self.rom.d, inst.address, inst.size, inst.book)
        rate = OUTPUT_RATE * inst.tuning * _pitch(note.pitch)
        name = which if isinstance(which, str) else None
        return Sound(samples=samples, rate=round(rate), name=name, instrument_data=inst, **what)

    def play(self, which: str | int, *, whole: bool = False) -> Sound:
        """
        A voice sound as the game plays it, at 32 kHz: a name from `SOUNDS`, or a number.

        That is the pitch of its note and any slide in it, its loudness, and the cut the game makes
        when the note ends before the sample does.

        `whole` adds what else the sound's entry starts, at the tick it starts it: for the punches
        and the kick that is the swish of the arm or leg, an instrument from another bank a tick
        after the voice. Without it only the voice is given.

        Returns:
            The sound, with `cut` set if the note ended before the sample did.

        Raises:
            BuildError: If the sound is played in a way that is not handled.
        """
        s = self.sound(which)
        label = s.name or f"{s.number:#x}"
        if len(s.notes) != 1:
            # Several notes in one breath ("ya-hoo"): the layer is set to run its notes together, so
            # the game does not start the sample again for the second; it goes on, at a new pitch.
            # That holds only while each note picks the same sample, and the one way of sliding
            # between them this follows is the one these sounds use: each note slides to its own
            # pitch from the one before it (the first from the pitch the slide names).
            same = all(n.legato and n.slide and n.slide[0] & 0x7F == 5 and not n.gate for n in s.notes)
            high = []
            if same:
                befores = [s.notes[0].slide[1], *(n.pitch for n in s.notes[:-1])]
                high = [max(n.pitch, before) for n, before in zip(s.notes, befores, strict=True)]
            address = s.instrument_data.address
            if not same or any(self._instrument(s.bank, s.instrument, p).address != address for p in high):
                raise BuildError(f"{label} plays {len(s.notes)} notes in one breath in a way that is not handled")
        out, cut = self._render(s.instrument_data, s.notes[0], s.samples, label, s.notes[1:])
        extra = []
        if whole:
            d = self.rom.d
            entry = struct.unpack_from(">H", d, self.base + self.table + 2 * s.number)[0]
            state = replace(self.defaults, layers={}, starts=[], clock=0)
            self._run_channel(entry, state, follow=True)
            if state.later:
                raise BuildError(f"{label} goes on in a way that is not followed")
            voice = min(layer for layer, _, _, _, _ in state.starts)
            for layer, at, bank, instrument, tick in state.starts:
                if layer == voice:
                    continue
                own = replace(state, instrument=instrument)
                notes = self._run_layer(at, own)
                if len(notes) != 1:
                    raise BuildError(f"{label}: layer {layer} plays {len(notes)} notes, which is not handled")
                note = notes[0]
                _transpose(note, state.transpose)
                inst = self._instrument(bank, own.instrument, _sample_pitch(note))  # (a layer may choose its own)
                src = decode(d, inst.address, inst.size, inst.book)
                more, _ = self._render(inst, note, src, f"{label} layer {layer}")
                start = round(tick / (self.tempo * 48.0 / TEMPO_DIVISOR) * OUTPUT_RATE / UPDATES)
                out += [0] * (start + len(more) - len(out))
                for i, v in enumerate(more):
                    out[start + i] += v
                extra.append(Extra(layer, bank, own.instrument, tick, note, len(more) / float(OUTPUT_RATE)))
            # The game mixes its voices with room to spare; here two at full strength can pass the
            # top of the scale, so the whole is turned down just enough to fit, the balance kept.
            peak = max(abs(v) for v in out)
            if peak > 32767:
                out = [round(v * 32767.0 / peak) for v in out]
        return Sound(samples=out, rate=OUTPUT_RATE, name=s.name, number=s.number, source=s, cut=cut, extra=extra)

    def _slide_speed(self, slide: tuple[int, int, int], ticks: int) -> float:
        """How fast a slide runs, in 127ths of its way each update, for a note `ticks` long."""
        mode, _, time = slide
        if not mode & 0x80:
            return 127.0 / time
        return 32512.0 * self.tempo * 48.0 / (ticks * float(TEMPO_DIVISOR) * time)

    def _render(
        self,
        inst: Instrument,
        note: Note,
        src: list[int],
        label: str,
        more: list[Note] | tuple[Note, ...] = (),
    ) -> tuple[list[int], bool]:
        """
        One note of one instrument at 32 kHz.

        `more` is the notes that follow it in the same breath, each sliding to its pitch from the one
        before (`play` has checked that): the sample plays on through them.

        Returns:
            The samples, and whether the note ended before the sample did.

        Raises:
            BuildError: If the sample loops, or the note is let go early or slides some other way.
        """
        if inst.loops:
            raise BuildError(f"{label} loops, which is not handled")
        if note.gate:
            raise BuildError(f"{label} lets its note go early, which is not handled")
        per_update = OUTPUT_RATE / UPDATES
        ticks_per_update = self.tempo * 48.0 / TEMPO_DIVISOR
        # The pitch: steady, or sliding from one note to another over a share of the note's length.
        start = end = _pitch(note.pitch)
        speed = 0.0
        if note.slide:
            mode, target, _ = note.slide
            other = _pitch(target)
            if mode & 0x7F in {1, 3, 5}:  # from the named note down or up to the played one
                start = other
            elif mode & 0x7F in {2, 4}:  # from the played note to the named one
                end = other
            else:
                raise BuildError(f"{label} slides in a way ({mode:#x}) that is not handled")
            speed = self._slide_speed(note.slide, note.ticks)
        extent = end / start - 1.0
        # The notes after it: (the update each begins on, where its pitch starts, how far it goes, how fast).
        turns, ticks, before = [], note.ticks, note
        for later in more:
            a, b = _pitch(before.pitch), _pitch(later.pitch)
            speed_then = self._slide_speed(later.slide, later.ticks)
            turns.append((math.ceil(ticks / ticks_per_update), a, b / a - 1.0, speed_then))
            ticks, before = ticks + later.ticks, later
        since = 0
        # Loudness, update by update: the instrument's envelope, then the fall once the note is let
        # go, which is when the last note of the breath ends.
        release_at = math.ceil(ticks / ticks_per_update)
        levels, level, update = [], 0.0, 0
        steps = list(inst.envelope)
        step_left, velocity = 0, 0.0
        while True:
            if update >= release_at:
                level -= inst.release * 24.0
                if level < 100.0:
                    levels.append(0.0)
                    break
            else:
                if step_left <= 0 and steps:
                    step_left, target = steps.pop(0)
                    velocity = (target - level) / step_left
                if step_left > 0:
                    level += velocity
                    step_left -= 1
            levels.append(level / 32767.0)
            update += 1
        gain = (note.velocity / 127.0) ** 2
        out, pos, n = [], 0.0, 0
        last = len(src) - 1
        while pos < last:
            at = n / per_update
            k = int(at)
            if k >= len(levels):
                break  # the note was let go and has faded out
            previous = levels[k - 1] if k else levels[0]
            amp = previous + (levels[k] - previous) * (at - k)
            while turns and k >= turns[0][0]:  # the next note of the breath begins: its own slide, from the start
                since, start, extent, speed = turns.pop(0)
            bend = min(127, int((k - since + 1) * speed))
            step = inst.tuning * start * (1.0 + extent * (2.0 ** (bend / 127.0) - 1.0))
            # The sample between two of its points: a curve through the four around it.
            i = int(pos)
            t = pos - i
            p0, p1 = src[i - 1] if i else src[0], src[i]
            p2 = src[i + 1] if i + 1 <= last else 0
            p3 = src[i + 2] if i + 2 <= last else 0
            value = p1 + 0.5 * t * (
                p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0))
            )
            out.append(max(-32768, min(32767, round(value * amp * gain))))
            pos += step
            n += 1
        return out, pos < last


def decode(d: bytes, addr: int, size: int, book: tuple[int, int, tuple[int, ...]]) -> list[int]:
    """ADPCM to 16-bit samples: each nine-byte frame is a header and sixteen four-bit residuals."""
    order, predictors, flat = book
    # Each predictor is `order` rows of eight; the decoder wants, for each of the eight outputs of a
    # half frame, the weights of the `order` previous samples and of the residuals before it.
    tables = []
    for i in range(predictors):
        rows = [[0] * (order + 8) for _ in range(8)]
        for j in range(order):
            for k in range(8):
                rows[k][j] = flat[(i * order + j) * 8 + k]
        for k in range(1, 8):
            rows[k][order] = rows[k - 1][order - 1]
        rows[0][order] = 2048
        for k in range(1, 8):
            for j in range(k, 8):
                rows[j][k + order] = rows[j - k][order]
        tables.append(rows)
    out = []
    state = [0] * 16
    for f in range(size // 9):
        p = addr + 9 * f
        head = d[p]
        scale, index = 1 << (head >> 4), head & 0xF
        if index >= predictors:
            raise BuildError(f"frame {f} names predictor {index} of {predictors}")
        rows = tables[index]
        resid = [(nib - 16 if nib >= 8 else nib) * scale for b in d[p + 1 : p + 9] for nib in (b >> 4, b & 0xF)]
        new = [0] * 16
        for half in range(2):
            vec = (state[16 - order :] if half == 0 else new[8 - order : 8]) + resid[half * 8 : half * 8 + 8]
            for k in range(8):
                row = rows[k]
                total = 0
                for j in range(order + 8):
                    total += row[j] * vec[j]
                new[half * 8 + k] = total >> 11  # floor, as the game's decoder does
        state = [max(-32768, min(32767, v)) for v in new]
        out += state
    return out
