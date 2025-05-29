import logging

from fasthtml.common import *

log = logging.getLogger(__name__)

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
    """Create a more idiomatic filter panel using FastHTML patterns."""
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

    return Div(
        Button(
            "Clear Filters",
            cls="contrast outline",
            onclick="window.location.href='/people'",
            style="width:100%; margin-bottom:15px; font-weight:bold;",
        ),
        Form(
            H3("People Filters"),
            # Live search
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by name...",
                    style="width:100%; margin-top:5px;",
                    hx_get="/people",
                    hx_trigger="keyup changed delay:500ms",
                    hx_target=".content-area",
                    hx_include="[name]",
                ),
                cls="search-box",
            ),
            # Type chips
            Div(
                H4("Person Types"),
                *[
                    chip_checkbox("type", t, t.lower() in selected_types, "/people")
                    for t in types
                ],
                style="margin-bottom:15px;",
            )
            if types
            else "",
            # Tag chips
            Div(
                H4("Tags"),
                *[
                    chip_checkbox("tag", tag, tag.lower() in selected_tags, "/people")
                    for tag in tags
                ],
                style="margin-bottom:15px;",
            )
            if tags
            else "",
            method="get",
            action="/people",
        ),
    )


def events_filter_panel(q="", selected_types=None, start_date="", end_date=""):
    """Events filter panel with date range."""
    from .data_access import events_index

    selected_types = selected_types or []
    types = sorted(
        {
            e.get("event_type", "").strip()
            for e in events_index.values()
            if e.get("event_type", "").strip()
        }
    )

    return Div(
        Button(
            "Clear Filters",
            cls="contrast outline",
            onclick="window.location.href='/events'",
            style="width:100%; margin-bottom:15px; font-weight:bold;",
        ),
        Form(
            H3("Event Filters"),
            # Live search
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by title...",
                    style="width:100%; margin-top:5px;",
                    hx_get="/events",
                    hx_trigger="keyup changed delay:500ms",
                    hx_target=".content-area",
                    hx_include="[name]",
                ),
                cls="search-box",
            ),
            # Type chips
            Div(
                H4("Event Types"),
                *[
                    chip_checkbox("etype", t, t.lower() in selected_types, "/events")
                    for t in types
                ],
                style="margin-bottom:15px;",
            )
            if types
            else "",
            # Date range
            Div(
                H4("Date Range"),
                Div(
                    Label("From:"),
                    Input(
                        type="date",
                        name="start_date",
                        value=start_date if start_date else None,
                        hx_get="/events",
                        hx_trigger="change",
                        hx_target=".content-area",
                        hx_include="[name]",
                    ),
                    style="margin-bottom:10px;",
                ),
                Div(
                    Label("To:"),
                    Input(
                        type="date",
                        name="end_date",
                        value=end_date if end_date else None,
                        hx_get="/events",
                        hx_trigger="change",
                        hx_target=".content-area",
                        hx_include="[name]",
                    ),
                    style="margin-bottom:10px;",
                ),
                cls="date-range",
            ),
            method="get",
            action="/events",
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
            # Live search
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by name...",
                    style="width:100%; margin-top:5px;",
                    hx_get="/locations",
                    hx_trigger="keyup changed delay:500ms",
                    hx_target=".content-area",
                    hx_include="[name]",
                ),
                cls="search-box",
            ),
            # Type chips
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
            method="get",
            action="/locations",
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
            # Live search
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by name...",
                    style="width:100%; margin-top:5px;",
                    hx_get="/organizations",
                    hx_trigger="keyup changed delay:500ms",
                    hx_target=".content-area",
                    hx_include="[name]",
                ),
                cls="search-box",
            ),
            # Type chips
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
            method="get",
            action="/organizations",
        ),
    )
