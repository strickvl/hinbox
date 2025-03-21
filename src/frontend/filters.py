import logging

from fasthtml.common import H3, H4, A, Button, Div, Form, Input, Label

log = logging.getLogger(__name__)

from .utils import random_pastel_color


def people_filter_panel(
    q: str = "", selected_types: list[str] = None, selected_tags: list[str] = None
):
    """
    Builds a People filter panel with checkboxes for 'type' & 'tag', plus a text search.
    Includes a 'Clear Filters' link to reset all filters. Styled more visibly.
    """
    from .data_access import people_index

    if selected_types is None:
        selected_types = []
    if selected_tags is None:
        selected_tags = []

    possible_types = set()
    possible_tags = set()

    for p in people_index.values():
        t = p.get("type", "").strip()
        if t:
            possible_types.add(t)
        p_tags = (p.get("profile") or {}).get("tags") or []
        for tg in p_tags:
            tg_str = tg.strip()
            if tg_str:
                possible_tags.add(tg_str)

    type_chips = []
    for pt in sorted(possible_types):
        selected = pt.lower() in selected_types
        chip_label = Label(
            Input(
                type="checkbox",
                name="type",
                value=pt,
                checked="checked" if selected else None,
                style="display:none;",
                onchange="this.parentElement.classList.toggle('selected', this.checked);",
                hx_trigger="change",
                hx_get="/people",
                hx_target=".content-area",
                hx_swap="innerHTML",
                hx_include="[name='type'], [name='tag']",
            ),
            pt.capitalize(),
            cls=f"filter-chip{' selected' if selected else ''}",
            style=f"background-color: {random_pastel_color(pt)};",
        )
        type_chips.append(chip_label)

    tag_chips = []
    for tg in sorted(possible_tags):
        selected_t = tg.lower() in selected_tags
        chip_label = Label(
            Input(
                type="checkbox",
                name="tag",
                value=tg,
                checked="checked" if selected_t else None,
                style="display:none;",
                onchange="this.parentElement.classList.toggle('selected', this.checked);",
                hx_trigger="change",
                hx_get="/people",
                hx_target=".content-area",
                hx_swap="innerHTML",
                hx_include="[name='type'], [name='tag']",
            ),
            tg.capitalize(),
            cls=f"filter-chip{' selected' if selected_t else ''}",
            style=f"background-color: {random_pastel_color(tg)};",
        )
        tag_chips.append(chip_label)

    return Div(
        A(
            "Clear Filters",
            href="/people",
            cls="button secondary",
            style="display:block; margin-bottom:15px; background-color:#e35a5a; color:white; font-weight:bold;",
        ),
        Form(
            H3("People Filters"),
            Div(H4("Person Types"), *type_chips, style="margin-bottom:15px;")
            if type_chips
            else "",
            Div(H4("Tags"), *tag_chips, style="margin-bottom:15px;")
            if tag_chips
            else "",
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
            Button("Apply Filters", type="submit", cls="primary"),
            method="get",
            action="/people",
        ),
    )


def events_filter_panel(
    q: str = "",
    selected_types: list[str] = None,
    start_date: str = "",
    end_date: str = "",
):
    """
    Builds an Events filter panel with checkboxes for 'event_type', optional date range, text search,
    and a clearly visible 'Clear Filters' link at the top.
    """
    from .data_access import events_index

    if selected_types is None:
        selected_types = []

    possible_types = set()
    for e in events_index.values():
        t = e.get("event_type", "").strip()
        if t:
            possible_types.add(t)

    chips = []
    for et in sorted(possible_types):
        selected = et.lower() in selected_types
        chip_label = Label(
            Input(
                type="checkbox",
                name="etype",
                value=et,
                checked="checked" if selected else None,
                style="display:none;",
                onchange="this.parentElement.classList.toggle('selected', this.checked);",
                hx_trigger="change",
                hx_get="/events",
                hx_target=".content-area",
                hx_swap="innerHTML",
                hx_include="[name='etype']",
            ),
            et.capitalize(),
            cls=f"filter-chip{' selected' if selected else ''}",
            style=f"background-color: {random_pastel_color(et)};",
        )
        chips.append(chip_label)

    return Div(
        A(
            "Clear Filters",
            href="/events",
            cls="button secondary",
            style="display:block; margin-bottom:15px; background-color:#e35a5a; color:white; font-weight:bold;",
        ),
        Form(
            H3("Event Filters"),
            Div(H4("Event Types"), *chips, style="margin-bottom:15px;")
            if chips
            else "",
            Div(
                H4("Date Range"),
                Div(
                    Label("From:"),
                    Input(
                        type="date",
                        name="start_date",
                        value=start_date if start_date else None,
                    ),
                    style="margin-bottom:10px;",
                ),
                Div(
                    Label("To:"),
                    Input(
                        type="date",
                        name="end_date",
                        value=end_date if end_date else None,
                    ),
                    style="margin-bottom:10px;",
                ),
                cls="date-range",
            ),
            Div(
                Label("Search: "),
                Input(
                    type="text",
                    name="q",
                    value=q,
                    placeholder="Search by title...",
                    style="width:100%; margin-top:5px;",
                ),
                cls="search-box",
            ),
            Button("Apply Filters", type="submit", cls="primary"),
            method="get",
            action="/events",
        ),
    )


def locations_filter_panel(q: str = "", selected_types: list[str] = None):
    """
    Builds a Locations filter panel with checkboxes for 'type', plus optional text search,
    and a very visible 'Clear Filters' link at the top.
    """
    from .data_access import locations_index

    if selected_types is None:
        selected_types = []

    possible_types = set()
    for loc in locations_index.values():
        t = loc.get("type", "").strip()
        if t:
            possible_types.add(t)

    chips = []
    for lt in sorted(possible_types):
        selected = lt.lower() in selected_types
        chip_label = Label(
            Input(
                type="checkbox",
                name="loc_type",
                value=lt,
                checked="checked" if selected else None,
                style="display:none;",
                onchange="this.parentElement.classList.toggle('selected', this.checked);",
                hx_trigger="change",
                hx_get="/locations",
                hx_target=".content-area",
                hx_swap="innerHTML",
                hx_include="[name='loc_type']",
            ),
            lt.capitalize(),
            cls=f"filter-chip{' selected' if selected else ''}",
            style=f"background-color: {random_pastel_color(lt)};",
        )
        chips.append(chip_label)

    return Div(
        A(
            "Clear Filters",
            href="/locations",
            cls="button secondary",
            style="display:block; margin-bottom:15px; background-color:#e35a5a; color:white; font-weight:bold;",
        ),
        Form(
            H3("Location Filters"),
            Div(H4("Location Types"), *chips, style="margin-bottom:15px;")
            if chips
            else "",
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
            Button("Apply Filters", type="submit", cls="primary"),
            method="get",
            action="/locations",
        ),
    )


def organizations_filter_panel(q: str = "", selected_types: list[str] = None):
    """
    Builds an Organizations filter panel with checkboxes for 'type', plus optional text search.
    Also includes a big red 'Clear Filters' link at the top to reset all filters.
    """
    log.warning(
        "filters.py organizations_filter_panel CALLED with q=%s; selected_types=%s",
        q,
        selected_types,
    )
    from .data_access import orgs_index

    if selected_types is None:
        selected_types = []

    possible_types = set()
    for org in orgs_index.values():
        t = org.get("type", "").strip()
        if t:
            possible_types.add(t)

    chips = []
    for ot in sorted(possible_types):
        selected = ot.lower() in selected_types
        chip_label = Label(
            Input(
                type="checkbox",
                name="org_type",
                value=ot,
                checked="checked" if selected else None,
                style="display:none;",
                onchange="this.parentElement.classList.toggle('selected', this.checked);",
                hx_trigger="change",
                hx_get="/organizations",
                hx_target=".content-area",
                hx_swap="innerHTML",
                hx_include="[name='org_type']",
            ),
            ot.capitalize(),
            cls=f"filter-chip{' selected' if selected else ''}",
            style=f"background-color: {random_pastel_color(ot)};",
        )
        chips.append(chip_label)

    return Div(
        A(
            "Clear Filters",
            href="/organizations",
            cls="button secondary",
            style="display:block; margin-bottom:15px; background-color:#e35a5a; color:white; font-weight:bold;",
        ),
        Form(
            H3("Organization Filters"),
            Div(H4("Organization Types"), *chips, style="margin-bottom:15px;")
            if chips
            else "",
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
            Button("Apply Filters", type="submit", cls="primary"),
            method="get",
            action="/organizations",
        ),
    )
