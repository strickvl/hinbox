"""Fail if any git-tracked Python file is longer than MAX_LINES.

Run via `just check-file-length` (also part of `just ci`).
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

MAX_LINES = 1000


def tracked_python_files() -> List[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"], capture_output=True, text=True, check=True
    ).stdout
    return [Path(p) for p in out.splitlines()]


def main() -> int:
    too_long: List[Tuple[Path, int]] = []
    for path in tracked_python_files():
        count = len(path.read_text(encoding="utf-8").splitlines())
        if count > MAX_LINES:
            too_long.append((path, count))

    for path, count in sorted(too_long, key=lambda item: -item[1]):
        print(f"{path}: {count} lines (limit {MAX_LINES})")
    if too_long:
        print(f"\n{len(too_long)} file(s) over {MAX_LINES} lines. Split them up.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
