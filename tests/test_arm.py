"""The assembler, against words from an independent assembler (Keystone)."""

import pytest

from mario import arm

HERE = 0x1000
KNOWN = [
    ("mov r0, #1", 0xE3A00001),
    ("mov r4, r0", 0xE1A04000),
    ("mvn r3, #0", 0xE3E03000),
    ("movne r0, #1", 0x13A00001),
    ("movs r1, r2", 0xE1B01002),
    ("mov r0, #0x3f800000", 0xE3A005FE),
    ("mov r3, #0x60000001", 0xE3A03216),
    ("cmp r0, #0", 0xE3500000),
    ("cmn r0, #1", 0xE3700001),
    ("tst r5, #0x800", 0xE3150B02),
    ("cmpne r1, #0", 0x13510000),
    ("add r1, r0, r1", 0xE0801001),
    ("add r5, sp, #0x68", 0xE28D5068),
    ("addcc r1, r1, r4, lsl #10", 0x30811504),
    ("sub r1, r0, #0x3a", 0xE240103A),
    ("subs r1, r1, #1", 0xE2511001),
    ("subcc r1, r1, r4, lsl #4", 0x30411204),
    ("and r0, r0, #0xff000000", 0xE20004FF),
    ("orr r0, r0, r1", 0xE1800001),
    ("orrs r1, r1, r2", 0xE1911002),
    ("orrmi r0, r0, #0x400", 0x43800B01),
    ("eor r1, r2, r1", 0xE0221001),
    ("bic r5, r5, #0x80000000", 0xE3C55102),
    ("bics r0, r0, #0x80000000", 0xE3D00102),
    ("movw r1, #0x828", 0xE3001828),
    ("ldr r0, [r7, #0x81c]", 0xE597081C),
    ("ldr r0, [r0]", 0xE5900000),
    ("ldr r0, [pc, r0]", 0xE79F0000),
    ("str r2, [r1, #0x120]", 0xE5812120),
    ("ldr r3, [r3, #4]", 0xE5933004),
    ("ldr r1, [r0, #-8]", 0xE5101008),
    ("ldrb r0, [r6, #1]", 0xE5D60001),
    ("strb r0, [r5]", 0xE5C50000),
    ("strbne r0, [r6, #1]", 0x15C60001),
    ("strbpl r1, [r5, #6]", 0x55C51006),
    ("ldrbne r1, [r1, #0xa]", 0x15D1100A),
    ("strh r1, [r6]", 0xE1C610B0),
    ("ldrh r0, [r6, #4]", 0xE1D600B4),
    ("strh r4, [r9, #0xe8]", 0xE1C94EB8),
    ("push {r4, r5, r6, lr}", 0xE92D4070),
    ("pop {r4, r5, r6, pc}", 0xE8BD8070),
    ("popne {r5, pc}", 0x18BD8020),
    ("push {r0, r3, r7, lr}", 0xE92D4089),
    ("bx lr", 0xE12FFF1E),
    ("bxne lr", 0x112FFF1E),
    ("blx r12", 0xE12FFF3C),
    ("bxeq r2", 0x012FFF12),
    ("vmov s18, r12", 0xEE09CA10),
    ("vmov s1, r0", 0xEE000A90),
    ("vmov s31, r3", 0xEE0F3A90),
    ("b 0x2000", 0xEA0003FE),
    ("bl 0x800", 0xEBFFFDFE),
    ("bne 0x1040", 0x1A00000E),
    ("blne 0x3000", 0x1B0007FE),
    ("beq 0x1000", 0x0AFFFFFE),
]
REFUSED = [
    "mov r0, #0x101",  # not eight bits turned by an even amount
    "ldr r0, [r1, #0x1000]",
    "strh r0, [r1, #0x100]",
    "b 0x1002",  # off a word
    "tst r3, r2, lsl r1",  # a shift by a register
    "ldr r0, [r1], #4",  # writeback
    "frob r0",
    "mov r16, #1",
    "vmov s32, r0",
    "mov r0, #nobody",
    'mov r0, #__import__("os")',
    "push r4",
]


def one(text: str) -> int:
    """The word for one instruction placed at `HERE`."""
    return arm.assemble(f".org {HERE:#x}\n{text}")[0][HERE]


@pytest.mark.parametrize(("text", "word"), KNOWN)
def test_known_words(text: str, word: int):
    """Every form the assembler knows gives the word Keystone gives."""
    assert one(text) == word


@pytest.mark.parametrize("text", REFUSED)
def test_refused(text: str):
    """What cannot be encoded is refused, never guessed at."""
    with pytest.raises(arm.AsmError):
        one(text)


def test_labels_and_symbols():
    """Labels, symbols and expressions meet in the second pass, wherever they are defined."""
    words, labels = arm.assemble(
        """
        .org START
        first:  ldr   r0, value         ; pc-relative, forwards
                cmp   r0, #LIMIT - 1
                bne   first             ; backwards
                bl    far
        value:  .word far - (first + 8)
        """,
        {"START": 0x2000, "LIMIT": 9, "far": 0x4000},
    )
    assert labels == {"first": 0x2000, "value": 0x2010}
    assert words == {
        0x2000: 0xE59F0008,
        0x2004: 0xE3500008,
        0x2008: 0x1AFFFFFC,
        0x200C: 0xEB0007FB,
        0x2010: 0x4000 - 0x2008,
    }


@pytest.mark.parametrize(
    ("text", "symbols"),
    [
        (".org 0\na: mov r0, r0\na: mov r0, r0", {}),  # a label defined twice
        (".org 0\na: mov r0, r0", {"a": 4}),  # a label that shadows a symbol
        ("a: mov r0, r0", {}),  # nothing says where it goes
        (".org 2\nmov r0, r0", {}),  # off a word
        (".org 0\nmov r0, r0\n.org 0\nmov r1, r1", {}),  # two words in one place
    ],
)
def test_layout_is_checked(text: str, symbols: dict[str, int]):
    """Where things go is checked before anything is encoded."""
    with pytest.raises(arm.AsmError):
        arm.assemble(text, symbols)


def test_errors_name_the_line():
    """An error says which line it is about."""
    with pytest.raises(arm.AsmError, match="line 3"):
        arm.assemble(".org 0\nmov r0, r0\nfrob r1")
