"""Record a merged entity's other name on the entity it merged into."""

from typing import Any, Dict, Optional, Tuple, Union

from src.logging_config import log


def _part(key: Union[str, Tuple], index: int) -> str:
    if isinstance(key, tuple):
        return key[index] if len(key) > index else ""
    return str(key) if index == 0 else ""


def add_alternative_name(
    entity_type: str,
    alternative_field: str,
    existing_entity: Dict[str, Any],
    new_key: Union[str, Tuple],
    source_entity: Optional[Dict[str, Any]] = None,
) -> bool:
    """Add ``new_key`` to the entity's alternative names; True if it was new.

    Shapes: people a list of names; organizations/locations
    ``{"name", "type"}``; events ``{"title", "start_date", "event_type"}``.
    """
    names = existing_entity.setdefault(alternative_field, [])
    if entity_type == "people":
        entry: Any = new_key
        label = f"name: '{new_key}'"
    elif entity_type == "events":
        entry = {
            "title": _part(new_key, 0),
            "start_date": _part(new_key, 1),
            "event_type": (source_entity or {}).get("event_type", ""),
        }
        label = f"event: '{entry['title']}'"
    else:  # organizations, locations
        entry = {"name": _part(new_key, 0), "type": _part(new_key, 1)}
        label = f"{entity_type[:-1]}: '{entry['name']}'"
    if entry in names:
        return False
    names.append(entry)
    log(f"Added alternative {label}", level="processing")
    return True
