from fasthtml.common import A, Div, Li, Span, Ul

from .utils import encode_key


def filter_simple_entities(entity_index, q="", selected_types=None, type_field="type"):
    """Generic filtering for simple entities (locations, organizations)."""
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


def render_simple_entity_list(filtered_entities, entity_type_name, route_prefix):
    """Generic rendering for simple entity lists."""
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
