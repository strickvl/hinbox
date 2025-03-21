import hashlib
import os

import pyarrow.parquet as pq

DATA_DIR = "data/entities"

PEOPLE_FILE = os.path.join(DATA_DIR, "people.parquet")
EVENTS_FILE = os.path.join(DATA_DIR, "events.parquet")
LOCATIONS_FILE = os.path.join(DATA_DIR, "locations.parquet")
ORGS_FILE = os.path.join(DATA_DIR, "organizations.parquet")


def load_parquet(path: str):
    """Load a Parquet file and return a list of dictionaries."""
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    return table.to_pylist()


# In-memory loaded data
people_data = load_parquet(PEOPLE_FILE)
events_data = load_parquet(EVENTS_FILE)
locations_data = load_parquet(LOCATIONS_FILE)
orgs_data = load_parquet(ORGS_FILE)


def make_person_key(person: dict) -> str:
    """Use the person's name as the key"""
    return person.get("name", "")


def make_event_key(event: dict) -> str:
    """Use a short hash for uniqueness, based on title##start_date."""
    title = event.get("title", "")
    start = event.get("start_date", "")
    combined = f"{title}##{start}"
    h = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"{title} ({h})"


def make_location_key(loc: dict) -> str:
    nm = loc.get("name", "")
    t = loc.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"{nm} ({h})"


def make_org_key(org: dict) -> str:
    nm = org.get("name", "")
    t = org.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"{nm} ({h})"


people_index = {make_person_key(p): p for p in people_data}
events_index = {make_event_key(e): e for e in events_data}
locations_index = {make_location_key(l): l for l in locations_data}
orgs_index = {make_org_key(o): o for o in orgs_data}
