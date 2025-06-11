"""Helper functions for filtering and rendering entity lists.

This module provides generic utility functions for filtering entities based on
search criteria and rendering them as FastHTML components. Functions are designed
to work with different entity types (locations, organizations) that share common
filtering patterns.
"""

from typing import Any, Dict, List, Optional, Tuple

from fasthtml.common import A, Div, Li, Span, Ul

from .utils import encode_key


def filter_simple_entities(
    entity_index: Dict[str, Dict[str, Any]],
    q: str = "",
    selected_types: Optional[List[str]] = None,
    type_field: str = "type",
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generic filtering for simple entities (locations, organizations).

    Filters entities based on name search and type selection criteria.
    Designed for entities that have 'name' and configurable type fields.

    Args:
        entity_index: Dictionary mapping entity keys to entity records
        q: Search query string to match against entity names (case-insensitive)
        selected_types: List of entity types to include in results
        type_field: Name of the field containing the entity type

    Returns:
        List of (entity_key, entity_record) tuples matching the filter criteria

    Note:
        Empty query matches all entities. Empty type selection matches all types.
        Name search is case-insensitive substring matching.
    """
    selected_types = selected_types or []

    filtered = []
    for k, entity in entity_index.items():
        # Search filter
        if q and q not in entity.get("name", "").strip().lower():
            continue

        # Type filter
        entity_type = entity.get(type_field, "").strip().lower()
        if selected_types and entity_type not in [t.lower() for t in selected_types]:
            continue

        filtered.append((k, entity))

    return filtered


def render_simple_entity_list(
    filtered_entities: List[Tuple[str, Dict[str, Any]]],
    entity_type_name: str,
    route_prefix: str,
) -> Div:
    """Generic rendering for simple entity lists.

    Creates a standard entity list display with count, entity links, and type badges.
    Returns an empty state message if no entities match the filters.

    Args:
        filtered_entities: List of (entity_key, entity_record) tuples to display
        entity_type_name: Display name for the entity type (e.g., "Locations")
        route_prefix: URL prefix for entity detail links (e.g., "locations")

    Returns:
        Div component containing count and entity list, or empty state message

    Note:
        Each entity is rendered as a link with an optional type badge.
        Entity keys are URL-encoded for safe use in links.
    """
    if not filtered_entities:
        return Div(
            f"No {entity_type_name.lower()} match your filters. Try adjusting your criteria.",
            cls="empty-state",
        )

    items = []
    for k, entity in filtered_entities:
        type_badge = ""
        if entity.get("type"):
            type_badge = Span(entity.get("type"), cls="tag")
        link = A(entity["name"], href=f"/{route_prefix}/{encode_key(k)}")
        items.append(Li(link, " ", type_badge))

    return Div(
        Div(
            f"{len(filtered_entities)} {entity_type_name.lower()} found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*items),
        id="results",
    )
