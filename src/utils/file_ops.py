"""File operations for entity data management.

This module is now domain-agnostic. Callers must provide a `base_dir`
that points to the current domain's output directory (typically from
`DomainConfig(domain).get_output_dir()`).
"""

import copy
import os
import tempfile
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from src.logging_config import get_logger

# Get logger for this module
logger = get_logger("utils.file_ops")


def sanitize_for_parquet(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively remove or transform fields that are incompatible with Arrow from
    an entity dictionary. Specifically, we remove/serialize any LLM response objects
    in reflection_history and ensure consistent types across all fields.

    Args:
        entity: Entity dictionary to sanitize

    Returns:
        Sanitized dictionary safe for Parquet storage
    """
    # Create a deep copy so we don't mutate the original
    sanitized = copy.deepcopy(entity)

    # If there's a reflection_history field, strip or convert the "response" object
    if "reflection_history" in sanitized and isinstance(
        sanitized["reflection_history"], list
    ):
        for iteration in sanitized["reflection_history"]:
            # Convert iteration to a dict if needed
            if isinstance(iteration, dict):
                # If there's a 'response' field that's not arrow-compatible, replace it or string-ify it
                if "response" in iteration:
                    # We'll store a string representation or just clear it out
                    iteration["response"] = str(iteration["response"])

    # Ensure consistent types for common fields that might have mixed types
    # Convert None values to empty strings for string fields
    string_fields = [
        "name",
        "title",
        "type",
        "text",
        "description",
        "reason",
        "profile_text",
        "location",
    ]
    for field in string_fields:
        if field in sanitized and sanitized[field] is None:
            sanitized[field] = ""

    # Ensure list fields are always lists
    list_fields = ["tags", "sources", "aliases", "article_ids", "issues"]
    for field in list_fields:
        if field in sanitized:
            if sanitized[field] is None:
                sanitized[field] = []
            elif not isinstance(sanitized[field], list):
                # Convert single values to list
                sanitized[field] = [sanitized[field]]

            # Convert enum values in lists to strings (for tags)
            if field == "tags" and sanitized[field]:
                converted_tags = []
                for tag in sanitized[field]:
                    if hasattr(tag, "value"):
                        converted_tags.append(tag.value)
                    else:
                        converted_tags.append(str(tag))
                sanitized[field] = converted_tags

    # Ensure numeric fields are properly typed
    numeric_fields = ["confidence", "similarity_score"]
    for field in numeric_fields:
        if field in sanitized and sanitized[field] is not None:
            try:
                sanitized[field] = float(sanitized[field])
            except (ValueError, TypeError):
                sanitized[field] = 0.0

    # Handle enum fields - convert enum to string
    enum_fields = ["event_type", "type"]
    for field in enum_fields:
        if field in sanitized and sanitized[field] is not None:
            # If it's an enum or has a value attribute, extract the string value
            if hasattr(sanitized[field], "value"):
                sanitized[field] = sanitized[field].value
            else:
                sanitized[field] = str(sanitized[field])

    # Handle datetime fields - convert to string
    datetime_fields = ["start_date", "end_date"]
    for field in datetime_fields:
        if field in sanitized and sanitized[field] is not None:
            # If it's a datetime object, convert to ISO string
            if hasattr(sanitized[field], "isoformat"):
                sanitized[field] = sanitized[field].isoformat()
            else:
                sanitized[field] = str(sanitized[field])

    # Handle nested dictionaries recursively
    for key, value in sanitized.items():
        if isinstance(value, dict):
            sanitized[key] = sanitize_for_parquet(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Sanitize list of dictionaries
            sanitized[key] = [
                sanitize_for_parquet(item) if isinstance(item, dict) else item
                for item in value
            ]

    # Return the sanitized dict
    return sanitized


def get_entity_output_path(entity_type: str, base_dir: str) -> str:
    """
    Get the output path for a given entity type under `base_dir`.

    Args:
        entity_type: Type of entity ("people", "events", "locations", "organizations")
        base_dir: Directory where entity parquet files for this domain are stored

    Returns:
        Path to the output file

    Raises:
        ValueError: If entity_type is not recognized
    """
    filenames = {
        "people": "people.parquet",
        "events": "events.parquet",
        "locations": "locations.parquet",
        "organizations": "organizations.parquet",
    }

    if entity_type not in filenames:
        raise ValueError(f"Unknown entity type: {entity_type}")

    return os.path.join(base_dir, filenames[entity_type])


def write_entity_to_file(
    entity_type: str, entity_key: Any, entity_data: Dict[str, Any], base_dir: str
):
    """
    Write a single entity to its respective Parquet file. This function uses
    an append/update approach for simplicity:
      - reads existing data from the file
      - updates or appends the entity
      - writes it all back

    Args:
        entity_type: "people", "events", "locations", or "organizations"
        entity_key: Key to identify entity (name for people, tuple for others)
        entity_data: Entity data to write
        base_dir: Domain-specific output directory
    """
    output_path = get_entity_output_path(entity_type, base_dir)

    # Sanitize incoming entity data to avoid storing objects that PyArrow can't handle
    entity_data = sanitize_for_parquet(entity_data)

    entities = []
    entity_found = False

    if os.path.exists(output_path):
        table = pq.read_table(output_path)
        entities = table.to_pylist()

        # Sanitize existing entities from file to ensure consistency
        sanitized_entities = []
        for existing_entity in entities:
            sanitized_entity = sanitize_for_parquet(existing_entity)
            sanitized_entities.append(sanitized_entity)
        entities = sanitized_entities

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
                entity_type == "locations"
                and (entity.get("name"), entity.get("type")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break
            elif (
                entity_type == "organizations"
                and (entity.get("name"), entity.get("type")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break

    if not entity_found:
        entities.append(entity_data)

    # Write the updated list back to the file
    if entities:
        try:
            table = pa.Table.from_pylist(entities)
            pq.write_table(table, output_path)

            action = "Updated" if entity_found else "Added"
            # Format entity_key appropriately for display
            if entity_type == "events" and isinstance(entity_key, tuple):
                display_key = entity_key[0]  # Just the event title
            elif entity_type in ["locations", "organizations"] and isinstance(
                entity_key, tuple
            ):
                display_key = entity_key[0]  # Just the name
            else:
                display_key = entity_key
            logger.debug(f"{action} {entity_type} entity: {display_key}")
        except Exception as e:
            # If we still get an error, try to identify the problematic field
            logger.error(f"Error writing {entity_type} to parquet: {e}")

            # Debug: Check for fields with inconsistent types
            if entities:
                field_types = {}
                for entity in entities:
                    for key, value in entity.items():
                        if key not in field_types:
                            field_types[key] = set()
                        field_types[key].add(type(value).__name__)

                # Log fields with multiple types
                for field, types in field_types.items():
                    if len(types) > 1:
                        logger.error(f"Field '{field}' has multiple types: {types}")
                        # Log some examples
                        examples = []
                        for entity in entities[:3]:
                            if field in entity:
                                examples.append(
                                    f"{type(entity[field]).__name__}: {repr(entity[field])[:50]}"
                                )
                        logger.error(f"  Examples: {examples}")
            raise
    else:
        logger.warning(f"No entities to write for {entity_type}")


def write_entities_table(
    entity_type: str, entities: List[Dict[str, Any]], base_dir: str
) -> None:
    """Write all entities of a given type to Parquet in a single atomic operation.

    Instead of per-entity read-modify-write (O(N^2) I/O), this writes the
    complete entity list once.  The write goes to a temporary file first and
    is then atomically renamed into place, so readers never see a half-written
    file.

    Args:
        entity_type: "people", "events", "locations", or "organizations"
        entities: Full list of entity dicts for this type (already in-memory)
        base_dir: Domain-specific output directory
    """
    if not entities:
        logger.debug(f"No {entity_type} entities to write — skipping")
        return

    output_path = get_entity_output_path(entity_type, base_dir)

    # Sanitize all entities for Parquet compatibility
    sanitized = [sanitize_for_parquet(e) for e in entities]

    try:
        # Write to a temp file in the same directory (same filesystem) so
        # os.replace is an atomic rename, not a cross-device copy.
        dir_name = os.path.dirname(output_path)
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=dir_name)
        os.close(fd)  # mkstemp opens the fd; we only need the path

        table = pa.Table.from_pylist(sanitized)
        pq.write_table(table, tmp_path)

        # Atomic replace — overwrites any existing file safely.
        os.replace(tmp_path, output_path)

        logger.info(f"Wrote {len(sanitized)} {entity_type} entities to {output_path}")
    except Exception:
        # Clean up the temp file if the rename didn't happen
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_entities_from_file(entity_type: str, base_dir: str) -> List[Dict[str, Any]]:
    """
    Read all entities of a given type from their Parquet file.

    Args:
        entity_type: Type of entity to read
        base_dir: Domain-specific output directory

    Returns:
        List of entity dictionaries, empty list if file doesn't exist
    """
    output_path = get_entity_output_path(entity_type, base_dir)

    if not os.path.exists(output_path):
        logger.info(f"No existing file found at {output_path}")
        return []

    try:
        table = pq.read_table(output_path)
        entities = table.to_pylist()
        logger.info(f"Read {len(entities)} {entity_type} entities from {output_path}")
        return entities
    except Exception as e:
        logger.error(f"Error reading {entity_type} from {output_path}: {e}")
        return []


def entity_exists(entity_type: str, entity_key: Any, base_dir: str) -> bool:
    """
    Check if an entity already exists in the file.

    Args:
        entity_type: Type of entity
        entity_key: Key to identify the entity
        base_dir: Domain-specific output directory

    Returns:
        True if entity exists, False otherwise
    """
    entities = read_entities_from_file(entity_type, base_dir)

    for entity in entities:
        if entity_type == "people" and entity.get("name") == entity_key:
            return True
        elif (
            entity_type == "events"
            and (entity.get("title"), entity.get("start_date", "")) == entity_key
        ):
            return True
        elif (
            entity_type == "locations"
            and (entity.get("name"), entity.get("type")) == entity_key
        ):
            return True
        elif (
            entity_type == "organizations"
            and (entity.get("name"), entity.get("type")) == entity_key
        ):
            return True

    return False
