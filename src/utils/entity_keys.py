"""The key the reader gives each entity row, shared with the pipeline so a
correction made in the reader names the same entity in a run.

People are keyed by name; the other tables by name (or title) plus a short
hash of the type (or start date), since two places can share a name.
"""

import hashlib
import re
from enum import Enum
from typing import Any, Callable, Dict

from src.constants import HASH_TRUNCATE_LENGTH


def _hashed_key(name_field: str, other_field: str) -> Callable[[Dict[str, Any]], str]:
    def key(row: Dict[str, Any]) -> str:
        name = row.get(name_field) or ""
        other = row.get(other_field) or ""
        h = hashlib.md5(f"{name}##{other}".encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
        return f"{name} ({h})"

    return key


def make_person_key(person: Dict[str, Any]) -> str:
    return person.get("name", "") or ""


make_event_key = _hashed_key("title", "start_date")
make_location_key = _hashed_key("name", "type")
make_org_key = _hashed_key("name", "type")

KEY_FUNCS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "people": make_person_key,
    "organizations": make_org_key,
    "locations": make_location_key,
    "events": make_event_key,
}
NAME_FIELDS: Dict[str, str] = {
    "people": "name",
    "organizations": "name",
    "locations": "name",
    "events": "title",
}


# A kind ("military_base") reaches the merger in three spellings: an Enum from a
# fresh extraction, "LocationType.MILITARY_BASE" from the extraction cache (its
# JSON writes enums with str()), and "military_base" from a saved table. Keys
# built from different spellings never meet, so the same place was created
# twice (12 Nov full run: two "Guantánamo North" military_base rows).
_ENUM_REPR = re.compile(r"^(?:Person|Organization|Location|Event)(?:Type|Tag)\.(\w+)$")
KIND_FIELDS = ("type", "event_type")


def plain_kind(value: Any) -> Any:
    """The configured string for an entity kind, whatever spelling it came in."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        m = _ENUM_REPR.match(value)
        return m.group(1).lower() if m else value
    return value


def plain_kinds(entity: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of ``entity`` with its kind fields as plain strings."""
    if not any(f in entity for f in KIND_FIELDS):
        return entity
    return {**entity, **{f: plain_kind(entity[f]) for f in KIND_FIELDS if f in entity}}
