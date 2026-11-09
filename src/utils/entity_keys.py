"""The key the reader gives each entity row, shared with the pipeline so a
correction made in the reader names the same entity in a run.

People are keyed by name; the other tables by name (or title) plus a short
hash of the type (or start date), since two places can share a name.
"""

import hashlib
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
