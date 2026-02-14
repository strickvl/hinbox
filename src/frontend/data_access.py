"""Data access layer for loading and indexing entity data from Parquet files.

This module provides functions to load entity data from domain-specific Parquet files,
create searchable indexes, and generate unique keys for entities. It supports both
domain-specific and legacy hardcoded data paths with fallback mechanisms.
"""

import hashlib
import os
from typing import Any, Dict, List

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.constants import HASH_TRUNCATE_LENGTH


def load_parquet(path: str) -> List[Dict[str, Any]]:
    """Load a Parquet file and return a list of dictionaries.

    Reads a Parquet file using PyArrow and converts it to a list of dictionaries
    for easy manipulation in the frontend. Returns empty list if file doesn't exist.

    Args:
        path: Absolute or relative path to the Parquet file

    Returns:
        List of dictionaries representing the Parquet file rows

    Note:
        Returns empty list rather than raising exception for missing files.
    """
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    return table.to_pylist()


def get_domain_data(domain: str = "guantanamo") -> Dict[str, List[Any]]:
    """Load data for a specific domain from its configured entity files.

    Loads all entity types (people, events, locations, organizations) for the
    specified domain using the domain's configuration. Falls back to legacy
    hardcoded paths if domain configuration fails.

    Args:
        domain: Domain name to load data for

    Returns:
        Dictionary with keys for each entity type containing lists of entity records

    Note:
        Falls back to "data/entities" directory if domain configuration fails.
        All entity types are included even if their files don't exist (empty lists).
    """
    try:
        config = DomainConfig(domain)
        data_dir = config.get_output_dir()

        people_file = os.path.join(data_dir, "people.parquet")
        events_file = os.path.join(data_dir, "events.parquet")
        locations_file = os.path.join(data_dir, "locations.parquet")
        orgs_file = os.path.join(data_dir, "organizations.parquet")

        return {
            "people": load_parquet(people_file),
            "events": load_parquet(events_file),
            "locations": load_parquet(locations_file),
            "organizations": load_parquet(orgs_file),
        }
    except Exception:
        # Fallback to legacy hardcoded paths
        data_dir = "data/entities"
        people_file = os.path.join(data_dir, "people.parquet")
        events_file = os.path.join(data_dir, "events.parquet")
        locations_file = os.path.join(data_dir, "locations.parquet")
        orgs_file = os.path.join(data_dir, "organizations.parquet")

        return {
            "people": load_parquet(people_file),
            "events": load_parquet(events_file),
            "locations": load_parquet(locations_file),
            "organizations": load_parquet(orgs_file),
        }


# Default data load for backward compatibility
_default_data = get_domain_data()
people_data = _default_data["people"]
events_data = _default_data["events"]
locations_data = _default_data["locations"]
orgs_data = _default_data["organizations"]


def make_person_key(person: Dict[str, Any]) -> str:
    """Generate a unique key for a person entity.

    Uses the person's name directly as the key since person names are
    expected to be unique within the dataset.

    Args:
        person: Person entity dictionary with 'name' field

    Returns:
        Person's name as the unique key

    Note:
        Returns empty string if 'name' field is missing.
    """
    return person.get("name", "")


def make_event_key(event: Dict[str, Any]) -> str:
    """Generate a unique key for an event entity with hash disambiguation.

    Creates a key combining the event title with a short hash for uniqueness,
    based on title and start_date to handle events with duplicate titles.

    Args:
        event: Event entity dictionary with 'title' and 'start_date' fields

    Returns:
        Formatted key as "Title (hash)" where hash is MD5 truncated to configured length

    Note:
        Hash is generated from "title##start_date" to ensure uniqueness.
    """
    title = event.get("title", "")
    start = event.get("start_date", "")
    combined = f"{title}##{start}"
    h = hashlib.md5(combined.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
    return f"{title} ({h})"


def make_location_key(loc: Dict[str, Any]) -> str:
    """Generate a unique key for a location entity with hash disambiguation.

    Creates a key combining the location name with a short hash for uniqueness,
    based on name and type to handle locations with duplicate names.

    Args:
        loc: Location entity dictionary with 'name' and 'type' fields

    Returns:
        Formatted key as "Name (hash)" where hash is MD5 truncated to configured length

    Note:
        Hash is generated from "name##type" to ensure uniqueness across types.
    """
    nm = loc.get("name", "")
    t = loc.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
    return f"{nm} ({h})"


def make_org_key(org: Dict[str, Any]) -> str:
    """Generate a unique key for an organization entity with hash disambiguation.

    Creates a key combining the organization name with a short hash for uniqueness,
    based on name and type to handle organizations with duplicate names.

    Args:
        org: Organization entity dictionary with 'name' and 'type' fields

    Returns:
        Formatted key as "Name (hash)" where hash is MD5 truncated to configured length

    Note:
        Hash is generated from "name##type" to ensure uniqueness across types.
    """
    nm = org.get("name", "")
    t = org.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
    return f"{nm} ({h})"


def build_indexes(domain_data: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    """Build searchable indexes from domain entity data.

    Creates dictionary indexes for all entity types using their respective
    key generation functions, enabling fast lookups by entity key.

    Args:
        domain_data: Dictionary containing lists of entities for each type

    Returns:
        Dictionary of indexes, each mapping entity keys to entity records

    Note:
        Each entity type uses its specific key generation function for consistent
        key formatting across the application.
    """
    return {
        "people": {make_person_key(p): p for p in domain_data["people"]},
        "events": {make_event_key(e): e for e in domain_data["events"]},
        "locations": {make_location_key(loc): loc for loc in domain_data["locations"]},
        "organizations": {make_org_key(o): o for o in domain_data["organizations"]},
    }


# Default indexes for backward compatibility
_default_indexes = build_indexes(_default_data)
people_index = _default_indexes["people"]
events_index = _default_indexes["events"]
locations_index = _default_indexes["locations"]
orgs_index = _default_indexes["organizations"]
