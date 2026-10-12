"""
A small assembler for 32-bit ARM (A32): just enough for the code this mod adds to the game.

It knows only what that code uses, and refuses everything else::

    mov mvn cmp cmn tst add sub and orr eor bic     register or #immediate; `s` and a condition
    movw rd, #imm16
    ldr str ldrb strb   rt, [rn] | [rn, #off] | [rn, rm] | label (pc-relative)
    strh ldrh           rt, [rn, #off]
    push {..}   pop {..}
    b bl                label or address
    bx blx              register
    vmov sn, rt         a core register's bits into a single-precision register
    .word expression    a 32-bit value
    .org expression     what follows is placed there
    name:               a label

Registers are r0-r12, sp, lr and pc. A condition goes on the end of the mnemonic (`bne`, `movcc`,
`strbne`). Numbers are decimal or 0x hex and `;` starts a comment. Wherever a number is expected an
expression may be written, using labels and the symbols handed to `assemble`.

Branches and pc-relative loads are relative, so the words are right wherever the game is loaded.
"""

from __future__ import annotations

import ast
import operator
import re

from mario.errors import BuildError

CONDITIONS = {
    "eq": 0, "ne": 1, "cs": 2, "hs": 2, "cc": 3, "lo": 3, "mi": 4, "pl": 5, "vs": 6, "vc": 7,
    "hi": 8, "ls": 9, "ge": 10, "lt": 11, "gt": 12, "le": 13, "al": 14,
}  # fmt: skip
ALWAYS = CONDITIONS["al"]
REGISTERS = {**{f"r{n}": n for n in range(16)}, "ip": 12, "sp": 13, "lr": 14, "pc": 15}
DATA = {
    "and": 0, "eor": 1, "sub": 2, "add": 4, "tst": 8, "cmp": 10, "cmn": 11, "orr": 12, "mov": 13, "bic": 14, "mvn": 15,
}  # fmt: skip
COMPARE = ("tst", "cmp", "cmn")
MEMORY = {"ldr": (1, 0), "str": (0, 0), "ldrb": (1, 1), "strb": (0, 1)}  # mnemonic: (load, byte)
HALF = {"strh": 0, "ldrh": 1}
SHIFTS = ("lsl", "lsr", "asr", "ror")
OTHERS = ("movw", "vmov", "push", "pop", "bx", "blx", "bl", "b")
MNEMONICS = sorted([*DATA, *MEMORY, *HALF, *OTHERS], key=len, reverse=True)

LABEL = re.compile(r"([A-Za-z_]\w*):\s*")
SHIFT = re.compile(rf"({'|'.join(SHIFTS)})\s+#(.+)")
SINGLE = re.compile(r"s([12]?\d|3[01])")
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}


class AsmError(BuildError):
    """Raised for text this assembler cannot turn into words."""


def encode_immediate(value: int) -> int | None:
    """The 12-bit field of an A32 immediate (eight bits turned right by an even amount), or None if it is not one."""
    value &= 0xFFFFFFFF
    for turn in range(16):
        turned = ((value << (2 * turn)) | (value >> (32 - 2 * turn))) & 0xFFFFFFFF
        if turned < 256:
            return turn << 8 | turned
    return None


def evaluate(expression: str, names: dict[str, int]) -> int:
    """
    Work out an integer expression of numbers, names, `+ - * << >> & |` and brackets.

    Returns:
        Its value.

    Raises:
        AsmError: If it uses anything else, or a name nobody defined.
    """

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        raise AsmError(f"cannot work out {expression!r}")

    try:
        return walk(ast.parse(expression.strip(), mode="eval").body)
    except SyntaxError as error:
        raise AsmError(f"cannot work out {expression!r}") from error


def assemble(text: str, symbols: dict[str, int] | None = None) -> tuple[dict[int, int], dict[str, int]]:
    """
    Assemble `text`, whose expressions may use `symbols`.

    Returns:
        The words by address, and the address of every label.

    Raises:
        AsmError: Naming the line, for anything that cannot be encoded.
    """
    symbols = dict(symbols or {})
    lines, labels = _lay_out(text, symbols)
    names = {**symbols, **labels}
    words = {}
    for number, here, line in lines:
        if here in words:
            raise AsmError(f"line {number}: two things are placed at {here:#x}")
        try:
            words[here] = _encode(line, here, names)
        except AsmError as error:
            raise AsmError(f"line {number} ({line}): {error}") from error
        except IndexError as error:
            raise AsmError(f"line {number} ({line}): an operand is missing") from error
    return words, labels


def _lay_out(text: str, symbols: dict[str, int]) -> tuple[list[tuple[int, int, str]], dict[str, int]]:
    """The first pass: (line number, address, instruction) for each word, and where every label is."""
    lines, labels, here = [], {}, None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";")[0].strip()
        while match := LABEL.match(line):
            name = match.group(1)
            if name in labels or name in symbols:
                raise AsmError(f"line {number}: {name} is defined twice")
            if here is None:
                raise AsmError(f"line {number}: a label before any .org")
            labels[name] = here
            line = line[match.end() :]
        if not line:
            continue
        if line.startswith(".org"):
            here = evaluate(line[4:], symbols)
            if here % 4:
                raise AsmError(f"line {number}: .org off a word boundary")
            continue
        if here is None:
            raise AsmError(f"line {number}: code before any .org")
        lines.append((number, here, line))
        here += 4
    return lines, labels


def _encode(line: str, here: int, names: dict[str, int]) -> int:
    """The word for one instruction or `.word` placed at `here`."""
    if line.startswith(".word"):
        return evaluate(line[5:], names) & 0xFFFFFFFF
    mnemonic, _, rest = line.partition(" ")
    base, flags, condition = _split(mnemonic.lower())
    operands = _operands(rest)
    word = condition << 28
    if base in {"b", "bl"}:
        return word | _branch(base, evaluate(operands[0], names), here)
    if base in {"bx", "blx"}:
        return word | 0x012FFF10 | (0x20 if base == "blx" else 0) | _register(operands[0])
    if base in {"push", "pop"}:
        return word | (0x092D0000 if base == "push" else 0x08BD0000) | _register_list(base, operands)
    if base == "movw":
        value = _immediate(operands[1], names)
        if not 0 <= value < 0x10000:
            raise AsmError("movw takes 0 to 0xffff")
        return word | 0x03000000 | (value >> 12) << 16 | _register(operands[0]) << 12 | value & 0xFFF
    if base == "vmov":
        single = _single(operands[0])
        return word | 0x0E000A10 | (single >> 1) << 16 | _register(operands[1]) << 12 | (single & 1) << 7
    if base in DATA:
        return word | _data(base, operands, names, flags=flags)
    return word | _memory(base, operands, here, names)


def _split(mnemonic: str) -> tuple[str, bool, int]:
    """(base, sets flags, condition) from e.g. `movne`, `adds`, `strbne`, `bls`."""
    for base in MNEMONICS:
        if not mnemonic.startswith(base):
            continue
        rest = mnemonic[len(base) :]
        flags = False
        if base in DATA and base not in COMPARE and rest.startswith("s") and (rest == "s" or rest[1:] in CONDITIONS):
            flags, rest = True, rest[1:]
        if not rest:
            return base, flags, ALWAYS
        if rest in CONDITIONS:
            return base, flags, CONDITIONS[rest]
    raise AsmError(f"unknown instruction {mnemonic!r}")


def _operands(text: str) -> list[str]:
    """Split on the commas that are not inside brackets or braces."""
    operands, depth, current = [], 0, ""
    for character in text:
        depth += (character in "[{") - (character in "]}")
        if character == "," and not depth:
            operands.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        operands.append(current.strip())
    return operands


def _register(name: str) -> int:
    if name.lower() not in REGISTERS:
        raise AsmError(f"{name!r} is not a register")
    return REGISTERS[name.lower()]


def _single(name: str) -> int:
    """The number of a single-precision register, `s0` to `s31`."""
    if not SINGLE.fullmatch(name.lower()):
        raise AsmError(f"{name!r} is not a single-precision register")
    return int(name[1:])


def _register_list(base: str, operands: list[str]) -> int:
    if not (operands and operands[0].startswith("{") and operands[0].endswith("}")):
        raise AsmError(f"{base} takes a list of registers in braces")
    mask = 0
    for name in operands[0][1:-1].split(","):
        mask |= 1 << _register(name.strip())
    return mask


def _immediate(text: str, names: dict[str, int]) -> int:
    if not text.startswith("#"):
        raise AsmError(f"expected #number, found {text!r}")
    return evaluate(text[1:], names)


def _branch(base: str, target: int, here: int) -> int:
    offset = target - (here + 8)
    if offset % 4 or not -(1 << 25) <= offset < (1 << 25):
        raise AsmError(f"branch to {target:#x} is out of reach or off a word")
    return (0x0B if base == "bl" else 0x0A) << 24 | (offset >> 2) & 0xFFFFFF


def _data(base: str, operands: list[str], names: dict[str, int], *, flags: bool) -> int:
    """A data-processing instruction, less its condition."""
    if base in COMPARE:
        rd, rn, last, flags = 0, _register(operands[0]), operands[1:], True
    elif base in {"mov", "mvn"}:
        rd, rn, last = _register(operands[0]), 0, operands[1:]
    else:
        rd, rn, last = _register(operands[0]), _register(operands[1]), operands[2:]
    word = DATA[base] << 21 | flags << 20 | rn << 16 | rd << 12
    if last[0].startswith("#"):
        field = encode_immediate(_immediate(last[0], names))
        if field is None:
            raise AsmError(f"{last[0]} is not a value an instruction can hold")
        return word | 1 << 25 | field
    word |= _register(last[0])
    if len(last) > 1:
        match = SHIFT.fullmatch(last[1].lower())
        if not match:
            raise AsmError(f"shift {last[1]!r} is not handled")
        amount = evaluate(match.group(2), names)
        if not 0 < amount < 32:
            raise AsmError(f"shift by {amount} is not handled")
        word |= amount << 7 | SHIFTS.index(match.group(1)) << 5
    return word


def _memory(base: str, operands: list[str], here: int, names: dict[str, int]) -> int:
    """A load or store, less its condition."""
    if len(operands) != 2:
        raise AsmError(f"{base} takes a register and an address")
    rt, address = _register(operands[0]), operands[1]
    index = None
    if not address.startswith("["):  # a label: pc-relative
        rn, offset = REGISTERS["pc"], evaluate(address, names) - (here + 8)
    else:
        if not address.endswith("]"):
            raise AsmError(f"writeback addressing is not handled: {address!r}")
        inner = _operands(address[1:-1])
        rn, offset = _register(inner[0]), 0
        if len(inner) == 2 and inner[1].startswith("#"):
            offset = _immediate(inner[1], names)
        elif len(inner) == 2:
            index = _register(inner[1])
        elif len(inner) != 1:
            raise AsmError(f"address {address!r} is not handled")
    up, size = offset >= 0, abs(offset)
    if base in MEMORY:
        load, byte = MEMORY[base]
        word = 1 << 26 | 1 << 24 | up << 23 | byte << 22 | load << 20 | rn << 16 | rt << 12
        if index is not None:
            return word | 1 << 25 | index
        if size > 0xFFF:
            raise AsmError(f"offset {offset} does not fit")
        return word | size
    if index is not None or size > 0xFF:
        raise AsmError(f"{base} takes an offset of 0 to 255 here")
    return 1 << 24 | up << 23 | 1 << 22 | HALF[base] << 20 | rn << 16 | rt << 12 | (size >> 4) << 8 | 0xB0 | size & 0xF
