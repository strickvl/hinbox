"""Domain-aware path tests for entity Parquet IO.

These tests verify that:
- File operations (write/read/exists) use the provided base_dir without relying on global paths.
- Process helpers in src.process_and_extract operate on the passed base_dir for both reading and writing.

Rationale:
We keep payloads minimal to avoid schema drift and ensure that sanitize_for_parquet
does not encounter complex types. Keys follow the production conventions:
- people: name (str)
- events: (title, start_date)
- locations: (name, type)
- organizations: (name, type)
"""

from src.process_and_extract import (
    load_existing_entities,
    write_entities_to_files,
)
from src.utils.file_ops import (
    entity_exists,
    read_entities_from_file,
    write_entity_to_file,
)


def test_file_ops_people(tmp_path):
    """Round-trip people IO using file_ops with explicit base_dir."""
    base_dir = tmp_path / "entities"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Minimal person payload; "name" is the key field for people
    person = {"name": "Test Person"}

    # Write and verify file creation
    write_entity_to_file("people", person["name"], person, str(base_dir))
    assert (base_dir / "people.parquet").exists()

    # Read back and verify content
    people = read_entities_from_file("people", str(base_dir))
    assert isinstance(people, list)
    assert len(people) == 1
    assert people[0].get("name") == "Test Person"

    # Existence checks (positive and negative)
    assert entity_exists("people", "Test Person", str(base_dir)) is True
    assert entity_exists("people", "Nonexistent", str(base_dir)) is False


def test_file_ops_events(tmp_path):
    """Round-trip events IO using file_ops with explicit base_dir and tuple key."""
    base_dir = tmp_path / "entities"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Minimal event payload; events key is (title, start_date)
    event = {"title": "Hearing", "start_date": "2025-01-01"}
    key = (event["title"], event["start_date"])

    # Write and verify file creation
    write_entity_to_file("events", key, event, str(base_dir))
    assert (base_dir / "events.parquet").exists()

    # Read back and verify content
    events = read_entities_from_file("events", str(base_dir))
    assert isinstance(events, list)
    assert len(events) == 1
    row = events[0]
    assert row.get("title") == "Hearing"
    assert row.get("start_date", "") == "2025-01-01"

    # Existence checks for exact and non-matching keys
    assert entity_exists("events", key, str(base_dir)) is True
    assert entity_exists("events", ("Hearing", "2024-01-01"), str(base_dir)) is False


def test_load_existing_entities_roundtrip(tmp_path):
    """Seed all entity types via file_ops and verify load_existing_entities reads them from base_dir."""
    base_dir = tmp_path / "entities"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Seed minimal entities with correct keys
    write_entity_to_file("people", "Alice", {"name": "Alice"}, str(base_dir))
    write_entity_to_file(
        "events",
        ("EventA", "2025-02-01"),
        {"title": "EventA", "start_date": "2025-02-01"},
        str(base_dir),
    )
    write_entity_to_file(
        "locations",
        ("Camp", "Facility"),
        {"name": "Camp", "type": "Facility"},
        str(base_dir),
    )
    write_entity_to_file(
        "organizations",
        ("ACME", "Company"),
        {"name": "ACME", "type": "Company"},
        str(base_dir),
    )

    # Load using process helper and verify keys are present
    entities = load_existing_entities(str(base_dir))

    assert "Alice" in entities["people"]
    assert ("EventA", "2025-02-01") in entities["events"]
    assert ("Camp", "Facility") in entities["locations"]
    assert ("ACME", "Company") in entities["organizations"]


def test_write_entities_to_files(tmp_path):
    """End-to-end: write all entity types using write_entities_to_files and verify Parquet outputs under base_dir."""
    base_dir = tmp_path / "entities"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Prepare minimal in-memory entities dict keyed by type-specific keys
    entities = {
        "people": {
            "John": {"name": "John"},
        },
        "events": {
            ("E1", "2025-03-01"): {"title": "E1", "start_date": "2025-03-01"},
        },
        "locations": {
            ("Site", "Building"): {"name": "Site", "type": "Building"},
        },
        "organizations": {
            ("Org", "Company"): {"name": "Org", "type": "Company"},
        },
    }

    # Write using process helper
    write_entities_to_files(entities, str(base_dir))

    # Ensure output files exist
    for name in ("people", "events", "locations", "organizations"):
        assert (base_dir / f"{name}.parquet").exists()

    # Verify counts via file_ops readers
    assert len(read_entities_from_file("people", str(base_dir))) == 1
    assert len(read_entities_from_file("events", str(base_dir))) == 1
    assert len(read_entities_from_file("locations", str(base_dir))) == 1
    assert len(read_entities_from_file("organizations", str(base_dir))) == 1
