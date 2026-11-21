"""Research schema migration discovery."""

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import List


class InvalidMigrationSet(RuntimeError):
    """Raised when packaged migrations cannot define one ordered history."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path


def migrations() -> List[Migration]:
    root = files("src.research.migrations")
    found: List[Migration] = []
    for item in root.iterdir():
        if item.name.endswith(".sql"):
            prefix, name = item.name[:-4].split("_", 1)
            found.append(Migration(int(prefix), name, Path(str(item))))
    ordered = sorted(found, key=lambda migration: migration.version)
    versions = [migration.version for migration in ordered]
    expected = list(range(1, len(ordered) + 1))
    if versions != expected:
        raise InvalidMigrationSet(
            f"migration versions must be unique and contiguous: expected {expected}, got {versions}"
        )
    return ordered


def latest_schema_version() -> int:
    available = migrations()
    return available[-1].version if available else 0
