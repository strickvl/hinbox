"""
Utility functions for the hinbox project.
"""

import os
from typing import Any, Dict

import pyarrow as pa
import pyarrow.parquet as pq

from src.constants import (
    EVENTS_OUTPUT_PATH,
    LOCATIONS_OUTPUT_PATH,
    ORGANIZATIONS_OUTPUT_PATH,
    PEOPLE_OUTPUT_PATH,
)


def write_entity_to_file(
    entity_type: str, entity_key: Any, entity_data: Dict[str, Any]
):
    """
    Write a single entity to its respective Parquet file. This function now uses
    an append/update approach for simplicity:
      - reads existing data from the file
      - updates or appends the entity
      - writes it all back

    Args:
        entity_type: "people", "events", "locations", or "organizations"
        entity_key: Key to identify entity (name for people, tuple for others)
        entity_data: Entity data to write
    """
    if entity_type == "people":
        output_path = PEOPLE_OUTPUT_PATH
    elif entity_type == "events":
        output_path = EVENTS_OUTPUT_PATH
    elif entity_type == "locations":
        output_path = LOCATIONS_OUTPUT_PATH
    elif entity_type == "organizations":
        output_path = ORGANIZATIONS_OUTPUT_PATH
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

    entities = []
    entity_found = False

    if os.path.exists(output_path):
        table = pq.read_table(output_path)
        entities = table.to_pylist()

        for i, entity in enumerate(entities):
            if entity_type == "people" and entity.get("name") == entity_key:
                entities[i] = entity_data  # Update existing entity
                entity_found = True
                break
            elif (
                entity_type == "events"
                and (entity.get("title"), entity.get("start_date", "")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break
            elif (
                entity_type in ["locations", "organizations"]
                and (entity.get("name"), entity.get("type", "")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break

    if not entity_found:
        entities.append(entity_data)  # Append new entity

    # Sort entities appropriately
    if entity_type == "people":
        entities.sort(key=lambda x: x["name"])
    elif entity_type == "events":
        entities.sort(key=lambda x: (x.get("start_date", ""), x.get("title", "")))
    elif entity_type in ["locations", "organizations"]:
        entities.sort(key=lambda x: (x.get("name", ""), x.get("type", "")))

    # Convert to PyArrow table and write to Parquet
    table = pa.Table.from_pylist(entities)
    pq.write_table(table, output_path)
