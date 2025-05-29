import hashlib
import os
from typing import Any, Dict, List

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.constants import HASH_TRUNCATE_LENGTH


def load_parquet(path: str):
    """Load a Parquet file and return a list of dictionaries."""
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    return table.to_pylist()


def get_domain_data(domain: str = "guantanamo") -> Dict[str, List[Any]]:
    """Load data for a specific domain."""
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


def make_person_key(person: dict) -> str:
    """Use the person's name as the key"""
    return person.get("name", "")


def make_event_key(event: dict) -> str:
    """Use a short hash for uniqueness, based on title##start_date."""
    title = event.get("title", "")
    start = event.get("start_date", "")
    combined = f"{title}##{start}"
    h = hashlib.md5(combined.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
    return f"{title} ({h})"


def make_location_key(loc: dict) -> str:
    nm = loc.get("name", "")
    t = loc.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
    return f"{nm} ({h})"


def make_org_key(org: dict) -> str:
    nm = org.get("name", "")
    t = org.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:HASH_TRUNCATE_LENGTH]
    return f"{nm} ({h})"


def build_indexes(domain_data: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
    """Build indexes from domain data."""
    return {
        "people": {make_person_key(p): p for p in domain_data["people"]},
        "events": {make_event_key(e): e for e in domain_data["events"]},
        "locations": {make_location_key(l): l for l in domain_data["locations"]},
        "organizations": {make_org_key(o): o for o in domain_data["organizations"]},
    }


# Default indexes for backward compatibility
_default_indexes = build_indexes(_default_data)
people_index = _default_indexes["people"]
events_index = _default_indexes["events"]
locations_index = _default_indexes["locations"]
orgs_index = _default_indexes["organizations"]
