"""The command line: `python -m mario <Super Mario 64 ROM> <Pikmin 3 Deluxe RomFS>`."""

from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

import mario

OUT = Path("Mario")  # the mod folder written when none is named


def main() -> None:
    """Build the mod from the files named on the command line."""
    parser = argparse.ArgumentParser(prog="python -m mario", description=mario.__doc__.strip().splitlines()[0])
    parser.add_argument("rom", type=Path, help="your Super Mario 64 ROM (.z64, .n64 or .v64)")
    parser.add_argument("romfs", type=Path, help="the RomFS extracted from your copy of Pikmin 3 Deluxe")
    parser.add_argument("-o", "--out", type=Path, default=OUT, help=f"the mod folder to write (default: {OUT})")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        mario.build(args.rom, args.romfs, args.out)
    except (OSError, ValueError) as error:  # a file that is missing, or is not what it should be
        sys.exit(f"The mod could not be built: {error}")


if __name__ == "__main__":
    main()
