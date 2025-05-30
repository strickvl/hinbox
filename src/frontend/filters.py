import logging

from fasthtml.common import *

log = logging.getLogger(__name__)

from .components import (
    ApplyFiltersButton,
    ClearFiltersButton,
    DateRangeInputs,
    FilterForm,
    SearchInput,
    TypeChipsSection,
)
from .utils import random_pastel_color


def chip_checkbox(name, value, checked, target_route):
    """Create a styled chip checkbox with FastHTML components."""
    return Label(
        Input(
            type="checkbox",
            name=name,
            value=value,
            checked="checked" if checked else None,
            style="display:none;",
            onchange="this.parentElement.classList.toggle('selected', this.checked);",
            hx_trigger="change",
            hx_get=target_route,
            hx_target=".content-area",
            hx_swap="innerHTML",
            hx_include="[name]",
        ),
        value.capitalize(),
        cls=f"filter-chip{' selected' if checked else ''}",
        style=f"background-color: {random_pastel_color(value)};",
    )


def people_filter_panel(q="", selected_types=None, selected_tags=None):
    """Create a more idiomatic filter panel using reusable components."""
    from .data_access import people_index

    selected_types = selected_types or []
    selected_tags = selected_tags or []

    # Gather possible filter values
    types = sorted(
        {
            p.get("type", "").strip()
            for p in people_index.values()
            if p.get("type", "").strip()
        }
    )
    tags = sorted(
        {
            tag.strip()
            for p in people_index.values()
            for tag in (p.get("profile") or {}).get("tags", [])
            if tag.strip()
        }
    )

    # Prepare chip data for reusable components
    type_chips_data = (
        [(t, t, t.lower() in selected_types) for t in types] if types else []
    )

    tag_chips_data = (
        [(tag, tag, tag.lower() in selected_tags) for tag in tags] if tags else []
    )

    return Div(
        ClearFiltersButton("/people"),
        FilterForm(
            "/people",
            H3("People Filters"),
            # Search input (no auto-trigger)
            SearchInput("q", q, "Search by name..."),
            # Type chips (keep immediate behavior)
            TypeChipsSection("Person Types", type_chips_data, "/people", "type"),
            # Tag chips (keep immediate behavior)
            TypeChipsSection("Tags", tag_chips_data, "/people", "tag"),
            # Apply Filters button
            ApplyFiltersButton("/people"),
        ),
    )


def events_filter_panel(q="", selected_types=None, start_date="", end_date=""):
    """Events filter panel with date range using reusable components."""
    from .data_access import events_index

    selected_types = selected_types or []
    types = sorted(
        {
            e.get("event_type", "").strip()
            for e in events_index.values()
            if e.get("event_type", "").strip()
        }
    )

    # Prepare chip data for reusable components
    type_chips_data = (
        [(t, t, t.lower() in selected_types) for t in types] if types else []
    )

    return Div(
        ClearFiltersButton("/events"),
        FilterForm(
            "/events",
            H3("Event Filters"),
            # Search input (no auto-trigger)
            SearchInput("q", q, "Search by title..."),
            # Type chips (keep immediate behavior)
            TypeChipsSection("Event Types", type_chips_data, "/events", "etype"),
            # Date range (no auto-trigger)
            DateRangeInputs(start_date, end_date),
            # Apply Filters button
            ApplyFiltersButton("/events"),
        ),
    )


def locations_filter_panel(q="", selected_types=None):
    """Locations filter panel with type filtering."""
    from .data_access import locations_index

    selected_types = selected_types or []
    types = sorted(
        {
            loc.get("type", "").strip()
            for loc in locations_index.values()
            if loc.get("type", "").strip()
        }
    )

    return Div(
        Button(
            "Clear Filters",
            cls="contrast outline",
            onclick="window.location.href='/locations'",
            style="width:100%; margin-bottom:15px; font-weight:bold;",
        ),
        Form(
            H3("Location Filters"),
            # Search input (no auto-trigger)
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by name...",
                    style="width:100%; margin-top:5px;",
                ),
                cls="search-box",
            ),
            # Type chips (keep immediate behavior)
            Div(
                H4("Location Types"),
                *[
                    chip_checkbox(
                        "loc_type", t, t.lower() in selected_types, "/locations"
                    )
                    for t in types
                ],
                style="margin-bottom:15px;",
            )
            if types
            else "",
            # Apply Filters button
            Button(
                "Apply Filters",
                type="submit",
                cls="primary",
                style="width:100%; margin-top:15px; font-weight:bold;",
            ),
            method="get",
            action="/locations",
            hx_get="/locations",
            hx_target=".content-area",
            hx_swap="innerHTML",
        ),
    )


def organizations_filter_panel(q="", selected_types=None):
    """Organizations filter panel with type filtering."""
    from .data_access import orgs_index

    selected_types = selected_types or []
    types = sorted(
        {
            org.get("type", "").strip()
            for org in orgs_index.values()
            if org.get("type", "").strip()
        }
    )

    return Div(
        Button(
            "Clear Filters",
            cls="contrast outline",
            onclick="window.location.href='/organizations'",
            style="width:100%; margin-bottom:15px; font-weight:bold;",
        ),
        Form(
            H3("Organization Filters"),
            # Search input (no auto-trigger)
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by name...",
                    style="width:100%; margin-top:5px;",
                ),
                cls="search-box",
            ),
            # Type chips (keep immediate behavior)
            Div(
                H4("Organization Types"),
                *[
                    chip_checkbox(
                        "org_type", t, t.lower() in selected_types, "/organizations"
                    )
                    for t in types
                ],
                style="margin-bottom:15px;",
            )
            if types
            else "",
            # Apply Filters button
            Button(
                "Apply Filters",
                type="submit",
                cls="primary",
                style="width:100%; margin-top:15px; font-weight:bold;",
            ),
            method="get",
            action="/organizations",
            hx_get="/organizations",
            hx_target=".content-area",
            hx_swap="innerHTML",
        ),
    )
