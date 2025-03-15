import hashlib
import os
from urllib.parse import quote, unquote

import markdown
import pyarrow.parquet as pq
from fasthtml.common import *

# ----- Constants -----
DATA_DIR = "data/entities"

# Filenames for each entity type (using Parquet)
PEOPLE_FILE = os.path.join(DATA_DIR, "people.parquet")
EVENTS_FILE = os.path.join(DATA_DIR, "events.parquet")
LOCATIONS_FILE = os.path.join(DATA_DIR, "locations.parquet")
ORGS_FILE = os.path.join(DATA_DIR, "organizations.parquet")


# ----- Data Loading -----
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


# ----- Text Processing -----
def transform_profile_text(text, articles):
    """Replace footnote references with links to articles."""
    import re

    # Build a map from article_id to article_url
    article_map = {}
    for a in articles:
        aid = a.get("article_id")
        article_map[aid] = a.get("article_url", "#")

    # Replace footnotes in the form ^[...] with links to the article URL,
    # replacing the raw UUID with a sequential number for unobtrusive display.
    pattern = r"\^\[([0-9a-fA-F-]+)\]"

    marker_map = {}
    marker_counter = [1]  # Using a list to allow modification in nested function

    def replacer(match):
        ref = match.group(1)
        url = article_map.get(ref, "#")
        if ref not in marker_map:
            marker_map[ref] = str(marker_counter[0])
            marker_counter[0] += 1
        marker = marker_map[ref]
        return f'<sup><a href="{url}" target="_blank">{marker}</a></sup>'

    return re.sub(pattern, replacer, text)


# ----- Entity Key Generation -----
# For each entity, we generate a "key" for referencing in routes
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


# Build indexes/dicts for easy retrieval
people_index = {make_person_key(p): p for p in people_data}
events_index = {make_event_key(e): e for e in events_data}
locations_index = {make_location_key(l): l for l in locations_data}
orgs_index = {make_org_key(o): o for o in orgs_data}

# ----- FastHTML Setup -----
app, rt = fast_app()

# ----- Global Styles -----
STYLES = Style("""
    :root {
        --primary: #004080;
        --primary-light: #3374a5;
        --secondary: #6c757d;
        --background: #f8f9fa;
        --sidebar: #f0f2f5;
        --text: #333;
        --text-light: #6c757d;
        --border: #dee2e6;
        --card: #fff;
        --highlight: #e8f4f8;
        --danger: #dc3545;
        --success: #28a745;
        --warning: #ffc107;
        --info: #17a2b8;
    }
    body {
        background-color: var(--background);
        color: var(--text);
        font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    nav {
        border-radius: 8px;
        background-color: var(--primary);
        padding: 12px 20px;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    nav a {
        color: white !important;
        margin-right: 20px;
        font-weight: 600;
        text-decoration: none;
        transition: opacity 0.2s;
        padding: 8px 12px;
        border-radius: 4px;
    }
    nav a:hover {
        background-color: var(--primary-light);
        opacity: 0.9;
    }
    nav button {
        background-color: white !important;
        color: var(--primary) !important;
    }
    .filter-panel {
        background-color: var(--sidebar);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .filter-panel h3 {
        margin-top: 0;
        color: var(--primary);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .filter-panel h4 {
        color: var(--text);
        margin: 15px 0 10px 0;
        font-size: 1rem;
    }
    .filter-panel label {
        display: inline-flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .filter-panel input[type="checkbox"] {
        margin-right: 8px;
    }
    .filter-panel button {
        width: 100%;
        margin-top: 15px;
    }
    .content-area {
        background-color: var(--card);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .content-area h2 {
        color: var(--primary);
        margin-top: 0;
        border-bottom: 2px solid var(--border);
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .content-area ul {
        padding-left: 20px;
    }
    .content-area li {
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 4px;
        transition: background-color 0.2s;
    }
    .content-area li:hover {
        background-color: var(--highlight);
    }
    .content-area a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 500;
    }
    .content-area a:hover {
        text-decoration: underline;
    }
    .entity-detail h2 {
        color: var(--primary);
        margin-bottom: 15px;
    }
    .entity-detail h3 {
        color: var(--primary);
        margin: 25px 0 10px 0;
        border-top: 1px solid var(--border);
        padding-top: 15px;
    }
    .entity-detail p {
        margin-bottom: 15px;
        line-height: 1.5;
    }
    .tag {
        display: inline-block;
        background-color: var(--primary-light);
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    .search-box {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .search-box input {
        width: 100%;
        padding: 8px 12px;
        border-radius: 4px;
        border: 1px solid var(--border);
    }
    .date-range {
        display: flex;
        flex-direction: column;
        gap: 10px;
        margin-bottom: 15px;
    }
    .empty-state {
        text-align: center;
        padding: 30px;
        color: var(--text-light);
    }
    .article-list {
        margin-top: 15px;
    }
    .article-list li {
        padding: 10px;
        border-bottom: 1px solid var(--border);
    }
    .article-list li:last-child {
        border-bottom: none;
    }
""")


# ----- Utility Functions -----
def encode_key(k: str) -> str:
    """Encode the entity key so it can be used in a URL."""
    return quote(k, safe="")


def decode_key(k: str) -> str:
    """Decode the entity key from a URL."""
    return unquote(k)


def nav_bar():
    """Returns an FT component with a top nav bar."""
    return Nav(
        A("Home", href="/", hx_boost="true"),
        A("People", href="/people", hx_boost="true"),
        A("Events", href="/events", hx_boost="true"),
        A("Locations", href="/locations", hx_boost="true"),
        A("Organizations", href="/organizations", hx_boost="true"),
        Button(
            "About",
            cls="secondary",
            style="margin-left:auto;",
            onclick="alert('Guantánamo Entities Browser helps researchers explore entities mentioned in documents related to the Guantánamo Bay detention camp.');",
        ),
        style="display:flex; gap:1em; margin-bottom:1em; align-items:center;",
    )


def main_layout(page_title: str, filter_panel, content):
    """
    Create a 3-section layout:
    1) Nav bar
    2) Left filter area
    3) Main content
    """
    return Titled(
        page_title,
        Container(
            STYLES,
            nav_bar(),
            # We'll wrap the filter panel and the main content in a side-by-side arrangement
            Div(
                Div(
                    filter_panel,
                    cls="filter-panel",
                    style="flex:0 0 220px;",
                ),
                Div(content, cls="content-area", style="flex:1; margin-left:20px;"),
                style="display:flex; gap:20px;",
            ),
        ),
    )


def format_article_list(articles):
    """Create a consistent formatting for article lists."""
    if not articles:
        return Div("No articles associated with this entity.", cls="empty-state")

    art_list = []
    for art in articles:
        art_list.append(
            Li(
                f"{art.get('article_title', 'Untitled')} ",
                A("(View Source)", href=art.get("article_url", "#"), target="_blank"),
                style="display:flex; justify-content:space-between; align-items:center;",
            )
        )

    return Ul(*art_list, cls="article-list")


# ----- Filter Panels -----
def people_filter_panel(
    q: str = "", selected_types: list[str] = None, selected_tags: list[str] = None
):
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
        p_tags = p.get("profile", {}).get("tags", [])
        for tg in p_tags:
            tg_str = tg.strip()
            if tg_str:
                possible_tags.add(tg_str)

    type_checks = []
    for pt in sorted(possible_types):
        type_checks.append(
            Div(
                Input(
                    type="checkbox",
                    name="type",
                    value=pt,
                    checked="checked" if pt.lower() in selected_types else None,
                ),
                Label(pt),
                style="margin-bottom:8px;",
            )
        )

    tag_checks = []
    for tg in sorted(possible_tags):
        tag_checks.append(
            Div(
                Input(
                    type="checkbox",
                    name="tag",
                    value=tg,
                    checked="checked" if tg.lower() in selected_tags else None,
                ),
                Label(tg),
                style="margin-bottom:8px;",
            )
        )

    return Form(
        H3("People Filters"),
        Div(H4("Person Types"), *type_checks) if type_checks else "",
        Div(H4("Tags"), *tag_checks) if tag_checks else "",
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
    )


def events_filter_panel(
    q: str = "",
    selected_types: list[str] = None,
    start_date: str = "",
    end_date: str = "",
):
    if selected_types is None:
        selected_types = []

    possible_types = set()
    for e in events_index.values():
        t = e.get("event_type", "").strip()
        if t:
            possible_types.add(t)

    checks = []
    for et in sorted(possible_types):
        checks.append(
            Div(
                Input(
                    type="checkbox",
                    name="etype",
                    value=et,
                    checked="checked" if et.lower() in selected_types else None,
                ),
                Label(et),
                style="margin-bottom:8px;",
            )
        )

    return Form(
        H3("Event Filters"),
        Div(H4("Event Types"), *checks) if checks else "",
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
                    type="date", name="end_date", value=end_date if end_date else None
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
    )


def locations_filter_panel(q: str = "", selected_types: list[str] = None):
    if selected_types is None:
        selected_types = []

    possible_types = set()
    for loc in locations_index.values():
        t = loc.get("type", "").strip()
        if t:
            possible_types.add(t)

    checks = []
    for lt in sorted(possible_types):
        checks.append(
            Div(
                Input(
                    type="checkbox",
                    name="loc_type",
                    value=lt,
                    checked="checked" if lt.lower() in selected_types else None,
                ),
                Label(lt),
                style="margin-bottom:8px;",
            )
        )

    return Form(
        H3("Location Filters"),
        Div(H4("Location Types"), *checks) if checks else "",
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
    )


def organizations_filter_panel(q: str = "", selected_types: list[str] = None):
    if selected_types is None:
        selected_types = []

    possible_types = set()
    for org in orgs_index.values():
        t = org.get("type", "").strip()
        if t:
            possible_types.add(t)

    checks = []
    for ot in sorted(possible_types):
        checks.append(
            Div(
                Input(
                    type="checkbox",
                    name="org_type",
                    value=ot,
                    checked="checked" if ot.lower() in selected_types else None,
                ),
                Label(ot),
                style="margin-bottom:8px;",
            )
        )

    return Form(
        H3("Organization Filters"),
        Div(H4("Organization Types"), *checks) if checks else "",
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
    )


# ----- Homepage -----
@rt("/")
def get_home():
    """Home route: show quick welcome and links to each entity type."""
    return Titled(
        "GTMO Browse - Home",
        Container(
            STYLES,
            nav_bar(),
            Div(
                H1(
                    "Guantánamo Entities Browser",
                    style="color:var(--primary); margin-bottom:30px; text-align:center;",
                ),
                Div(
                    Div(
                        H2("Browse Entities"),
                        P(
                            "Explore entities extracted from documents related to Guantánamo Bay."
                        ),
                        style="text-align:center; margin-bottom:30px;",
                    ),
                    Div(
                        Div(
                            H3("People", style="color:var(--primary);"),
                            P(
                                "Browse detainees, officials, legal representatives, and other individuals."
                            ),
                            A(
                                "View People →",
                                href="/people",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        Div(
                            H3("Events", style="color:var(--primary);"),
                            P(
                                "Timeline of hearings, transfers, policy changes, and other significant events."
                            ),
                            A(
                                "View Events →",
                                href="/events",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        Div(
                            H3("Locations", style="color:var(--primary);"),
                            P(
                                "Explore detention facilities, courtrooms, and other relevant locations."
                            ),
                            A(
                                "View Locations →",
                                href="/locations",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        Div(
                            H3("Organizations", style="color:var(--primary);"),
                            P(
                                "Agencies, military units, legal teams, and other organizations involved."
                            ),
                            A(
                                "View Organizations →",
                                href="/organizations",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        style="display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:20px;",
                    ),
                    cls="content-area",
                ),
            ),
        ),
    )


# ----- People -----
@rt("/people")
def list_people(request):
    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("type")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]

    selected_tags_raw = request.query_params.getlist("tag")
    selected_tags = [t.strip().lower() for t in selected_tags_raw if t.strip()]

    filtered_items = []
    for k, person in people_index.items():
        ptype = person.get("type", "").strip().lower()
        pname = person.get("name", "").strip().lower()
        p_tags = [
            tg.strip().lower() for tg in person.get("profile", {}).get("tags", [])
        ]

        # if there's a type filter, skip if not matching
        if selected_types and ptype not in selected_types:
            continue
        # optional search
        if q and q not in pname:
            continue
        # tag filter
        if selected_tags:
            # Must have intersection with selected_tags
            if not set(p_tags).intersection(selected_tags):
                continue

        # Display type badge for each person
        type_badge = ""
        if person.get("type"):
            type_badge = Span(person.get("type"), cls="tag")

        link = A(person["name"], href=f"/people/{encode_key(k)}")
        filtered_items.append(Li(link, " ", type_badge))

    content = Div(
        H2("People"),
        Div(
            f"{len(filtered_items)} people found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*filtered_items)
        if filtered_items
        else Div(
            "No people match your filters. Try adjusting your criteria.",
            cls="empty-state",
        ),
    )

    return main_layout(
        "GTMO Browse - People",
        people_filter_panel(
            q=q, selected_types=selected_types, selected_tags=selected_tags
        ),
        content,
    )


@rt("/people/{key:path}")
def show_person(key: str):
    """Display detail about one person."""
    actual_key = decode_key(key)
    person = people_index.get(actual_key)
    if not person:
        return Titled(
            "GTMO Browse - People - Not Found",
            Container(
                STYLES,
                nav_bar(),
                Div(
                    H2("Person not found", style="color:var(--danger);"),
                    P(f"No person found with the name: {actual_key}"),
                    A("← Back to People", href="/people", cls="primary"),
                    cls="content-area",
                    style="max-width:800px; margin:0 auto;",
                ),
            ),
        )

    name = person.get("name", "N/A")
    typ = person.get("type", "N/A")
    profile = person.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, person.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = person.get("articles", [])

    # Create tags display
    tags = profile.get("tags", [])
    tag_elements = []
    for tag in tags:
        if tag.strip():
            tag_elements.append(Span(tag, cls="tag"))

    detail_content = Div(
        H2(f"Person: {name}"),
        Div(
            Span(f"Type: ", style="font-weight:bold;"),
            Span(typ, cls="tag"),
            style="margin-bottom:15px;",
        ),
        Div(*tag_elements, style="margin-bottom:20px;") if tag_elements else "",
        H3("Profile Information"),
        Div(
            NotStr(markdown.markdown(transformed_text))
            if transformed_text
            else "No detailed profile information available for this person.",
            cls="profile-text",
        ),
        Div(
            Span("AI Confidence: ", style="font-weight:bold;"),
            Span(conf, style="font-style:italic;"),
            style="margin-top:10px; color:var(--text-light); font-size:0.9rem;",
        ),
        H3("Related Articles"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        f"GTMO Browse - People - {name}",
        Div(
            H3("Navigation"),
            A(
                "← Back to People",
                href="/people",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
    )


# ----- Events -----
@rt("/events")
def list_events(request):
    import arrow

    def parse_dt(dt):
        if not dt:
            return None
        try:
            return arrow.get(dt)
        except:
            return None

    # gather query filters
    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("etype")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]
    start_q = request.query_params.get("start_date", "")
    end_q = request.query_params.get("end_date", "")

    def rowinfo(k, ev):
        startdt = parse_dt(ev.get("start_date", ""))
        return (k, startdt, ev)

    all_rows = [rowinfo(k, e) for k, e in events_index.items()]

    # filter by type, date range, optional search on title
    filtered = []
    for k, startdt, ev in all_rows:
        evtype = ev.get("event_type", "").strip().lower()
        title_lower = ev.get("title", "").strip().lower()

        # date checks
        keep = True
        if selected_types and evtype not in selected_types:
            keep = False

        if q and q not in title_lower:
            keep = False

        if keep and start_q:
            try:
                filter_start = arrow.get(start_q)
                if startdt and startdt < filter_start:
                    keep = False
            except:
                pass

        if keep and end_q:
            try:
                filter_end = arrow.get(end_q)
                if startdt and startdt > filter_end:
                    keep = False
            except:
                pass

        if keep:
            filtered.append((k, startdt, ev))

    # Sort by date ascending
    filtered.sort(key=lambda x: (x[1].timestamp() if x[1] else 0))

    items = []
    for k, startdt, ev in filtered:
        start_str = startdt.format("YYYY-MM-DD") if startdt else "Unknown"
        event_type = ev.get("event_type", "")

        # Display type badge for each event
        type_badge = ""
        if event_type:
            type_badge = Span(event_type, cls="tag")

        link = A(ev["title"], href=f"/events/{encode_key(k)}")
        items.append(
            Li(
                Div(
                    Span(
                        start_str,
                        style="font-weight:bold; margin-right:10px; color:var(--primary);",
                    ),
                    link,
                    style="display:flex; align-items:center;",
                ),
                Div(type_badge, style="margin-top:5px;") if type_badge else "",
            )
        )

    content = Div(
        H2("Events"),
        Div(
            f"{len(filtered)} events found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*items)
        if items
        else Div(
            "No events match your filters. Try adjusting your criteria.",
            cls="empty-state",
        ),
    )

    return main_layout(
        "GTMO Browse - Events",
        events_filter_panel(q, selected_types, start_q, end_q),
        content,
    )


@rt("/events/{key:path}")
def show_event(key: str):
    actual_key = decode_key(key)
    ev = events_index.get(actual_key)
    if not ev:
        return main_layout(
            "GTMO Browse - Events - Not Found",
            Div("No filters for detail pages."),
            Div(
                H2("Event not found", style="color:var(--danger);"),
                P(f"No event found with the key: {actual_key}"),
                A("← Back to Events", href="/events", cls="primary"),
            ),
        )

    title = ev.get("title", "N/A")
    event_type = ev.get("event_type", "N/A")
    start = ev.get("start_date", "")
    end = ev.get("end_date", "")
    desc = ev.get("description", "")
    is_fuzzy = ev.get("is_fuzzy_date", False)
    profile = ev.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, ev.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = ev.get("articles", [])

    detail_content = Div(
        H2(f"Event: {title}"),
        Div(
            Span("Type: ", style="font-weight:bold;"),
            Span(event_type, cls="tag"),
            style="margin-bottom:15px;",
        ),
        Div(
            Div(
                Span("Start Date: ", style="font-weight:bold;"),
                Span(start if start else "Unknown"),
                style="margin-bottom:5px;",
            ),
            Div(
                Span("End Date: ", style="font-weight:bold;"),
                Span(end if end else "N/A"),
                style="margin-bottom:5px;",
            ),
            Div(
                Span("Date Precision: ", style="font-weight:bold;"),
                Span("Approximate" if is_fuzzy else "Exact"),
                style="font-style:italic; color:var(--text-light);",
            ),
            style="margin-bottom:20px; padding:10px; background:var(--highlight); border-radius:5px;",
        ),
        H3("Description"),
        P(desc)
        if desc
        else P(
            "No description available.",
            style="font-style:italic; color:var(--text-light);",
        ),
        H3("Profile Information"),
        Div(
            NotStr(markdown.markdown(transformed_text))
            if transformed_text
            else "No detailed profile information available for this event.",
            cls="profile-text",
        ),
        Div(
            Span("AI Confidence: ", style="font-weight:bold;"),
            Span(conf, style="font-style:italic;"),
            style="margin-top:10px; color:var(--text-light); font-size:0.9rem;",
        ),
        H3("Related Articles"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        f"GTMO Browse - Events - {title}",
        Div(
            H3("Navigation"),
            A(
                "← Back to Events",
                href="/events",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
    )


# ----- Locations -----
@rt("/locations")
def list_locations(request):
    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("loc_type")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]

    filtered_items = []
    for k, loc in locations_index.items():
        ltype = loc.get("type", "").strip().lower()
        lname = loc.get("name", "").strip().lower()

        if selected_types and ltype not in selected_types:
            continue
        if q and q not in lname:
            continue

        # Display type badge for each location
        type_badge = ""
        if loc.get("type"):
            type_badge = Span(loc.get("type"), cls="tag")

        link = A(loc["name"], href=f"/locations/{encode_key(k)}")
        filtered_items.append(Li(link, " ", type_badge))

    content = Div(
        H2("Locations"),
        Div(
            f"{len(filtered_items)} locations found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*filtered_items)
        if filtered_items
        else Div(
            "No locations match your filters. Try adjusting your criteria.",
            cls="empty-state",
        ),
    )

    return main_layout(
        "GTMO Browse - Locations",
        locations_filter_panel(q=q, selected_types=selected_types),
        content,
    )


@rt("/locations/{key:path}")
def show_location(key: str):
    actual_key = decode_key(key)
    loc = locations_index.get(actual_key)
    if not loc:
        return main_layout(
            "GTMO Browse - Locations - Not Found",
            Div("No filters for detail pages."),
            Div(
                H2("Location not found", style="color:var(--danger);"),
                P(f"No location found with the key: {actual_key}"),
                A("← Back to Locations", href="/locations", cls="primary"),
            ),
        )

    nm = loc.get("name", "N/A")
    typ = loc.get("type", "N/A")
    profile = loc.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, loc.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = loc.get("articles", [])

    detail_content = Div(
        H2(f"Location: {nm}"),
        Div(
            Span("Type: ", style="font-weight:bold;"),
            Span(typ, cls="tag"),
            style="margin-bottom:20px;",
        ),
        H3("Profile Information"),
        Div(
            NotStr(markdown.markdown(transformed_text))
            if transformed_text
            else "No detailed profile information available for this location.",
            cls="profile-text",
        ),
        Div(
            Span("AI Confidence: ", style="font-weight:bold;"),
            Span(conf, style="font-style:italic;"),
            style="margin-top:10px; color:var(--text-light); font-size:0.9rem;",
        ),
        H3("Related Articles"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        f"GTMO Browse - Locations - {nm}",
        Div(
            H3("Navigation"),
            A(
                "← Back to Locations",
                href="/locations",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
    )


# ----- Organizations -----
@rt("/organizations")
def list_orgs(request):
    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("org_type")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]

    filtered_items = []
    for k, org in orgs_index.items():
        otype = org.get("type", "").strip().lower()
        oname = org.get("name", "").strip().lower()

        if selected_types and otype not in selected_types:
            continue
        if q and q not in oname:
            continue

        # Display type badge for each organization
        type_badge = ""
        if org.get("type"):
            type_badge = Span(org.get("type"), cls="tag")

        link = A(org["name"], href=f"/organizations/{encode_key(k)}")
        filtered_items.append(Li(link, " ", type_badge))

    content = Div(
        H2("Organizations"),
        Div(
            f"{len(filtered_items)} organizations found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*filtered_items)
        if filtered_items
        else Div(
            "No organizations match your filters. Try adjusting your criteria.",
            cls="empty-state",
        ),
    )

    return main_layout(
        "GTMO Browse - Organizations",
        organizations_filter_panel(q=q, selected_types=selected_types),
        content,
    )


@rt("/organizations/{key:path}")
def show_org(key: str):
    actual_key = decode_key(key)
    org = orgs_index.get(actual_key)
    if not org:
        return main_layout(
            "GTMO Browse - Organizations - Not Found",
            Div("No filters for detail pages."),
            Div(
                H2("Organization not found", style="color:var(--danger);"),
                P(f"No organization found with the key: {actual_key}"),
                A("← Back to Organizations", href="/organizations", cls="primary"),
            ),
        )

    nm = org.get("name", "N/A")
    typ = org.get("type", "N/A")
    profile = org.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, org.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = org.get("articles", [])

    detail_content = Div(
        H2(f"Organization: {nm}"),
        Div(
            Span("Type: ", style="font-weight:bold;"),
            Span(typ, cls="tag"),
            style="margin-bottom:20px;",
        ),
        H3("Profile Information"),
        Div(
            NotStr(markdown.markdown(transformed_text))
            if transformed_text
            else "No detailed profile information available for this organization.",
            cls="profile-text",
        ),
        Div(
            Span("AI Confidence: ", style="font-weight:bold;"),
            Span(conf, style="font-style:italic;"),
            style="margin-top:10px; color:var(--text-light); font-size:0.9rem;",
        ),
        H3("Related Articles"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        f"GTMO Browse - Organizations - {nm}",
        Div(
            H3("Navigation"),
            A(
                "← Back to Organizations",
                href="/organizations",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
    )


# ----- Run -----
if __name__ == "__main__":
    serve()
