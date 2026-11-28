"""Give each registered Hinbox evaluator its description in Kitaru.

Each evaluator script in ``scripts/kitaru_evaluators/`` holds ``NAME`` and
``DESCRIPTION``: what each of its numbers means, shown on the evaluator's page
and beside an experiment's results in the Kitaru dashboard. ``kitaru evaluator
register`` takes ``--description`` only when it creates an evaluator and the
CLI can't change it afterwards, so this sets it through the API on every
evaluator of that name the logged-in workspace has; one not registered there
is skipped. Run it after registering an evaluator or editing a description::

    just kitaru-describe            # set them
    just kitaru-describe --dry-run  # list what would change
"""

import argparse
import ast
import asyncio
import sys
from pathlib import Path
from typing import Dict, List

FOLDER = Path(__file__).resolve().parent / "kitaru_evaluators"


def descriptions(folder: Path = FOLDER) -> Dict[str, str]:
    """``NAME`` -> ``DESCRIPTION`` of every evaluator script, read without
    importing it (the scripts import Kitaru's worker package)."""
    out: Dict[str, str] = {}
    for path in sorted(folder.glob("*.py")):
        found: Dict[str, str] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("NAME", "DESCRIPTION")
                and isinstance(node.value, ast.Constant)
            ):
                found[node.targets[0].id] = str(node.value.value).strip()
        if "NAME" in found and found.get("DESCRIPTION"):
            out[found["NAME"]] = found["DESCRIPTION"]
    return out


async def describe(wanted: Dict[str, str], dry_run: bool = False) -> List[str]:
    from kitaru.api_models.v1.evaluator import EvaluatorUpdateRequest
    from kitaru.client import KitaruAPIClient

    lines: List[str] = []
    async with KitaruAPIClient() as client:
        seen = set()
        async for ev in client.evaluators.iter():
            text = wanted.get(ev.name)
            if text is None:
                continue
            seen.add(ev.name)
            if (ev.description or "").strip() == text:
                lines.append(f"{ev.name} ({ev.id}): already current")
                continue
            if not dry_run:
                await client.evaluators.update(
                    ev.id, EvaluatorUpdateRequest(description=text)
                )
            lines.append(f"{ev.name} ({ev.id}): {'would set' if dry_run else 'set'}")
        lines += [
            f"{n}: not registered here, skipped" for n in sorted(set(wanted) - seen)
        ]
    return lines


def main(argv: List[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    for line in asyncio.run(describe(descriptions(), a.dry_run)):
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
