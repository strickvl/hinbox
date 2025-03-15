import hashlib
import os
from urllib.parse import quote, unquote

import markdown
import pyarrow.parquet as pq
from fasthtml.common import *

DATA_DIR = "data/entities"

# Filenames for each entity type (using Parquet)
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

def transform_profile_text(text, articles):
    import re

    # Build a map from article_id to article_url
    article_map = {}
    for a in articles:
        aid = a.get("article_id")
        article_map[aid] = a.get("article_url", "#")

    # We'll replace footnotes in the form ^[...] with links to the article URL,
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
people_index = {}
for p in people_data:
    k = make_person_key(p)
    people_index[k] = p

events_index = {}
for e in events_data:
    k = make_event_key(e)
    events_index[k] = e

locations_index = {}
for l in locations_data:
    k = make_location_key(l)
    locations_index[k] = l

orgs_index = {}
for o in orgs_data:
    k = make_org_key(o)
    orgs_index[k] = o

################################################################
#              Utility
################################################################

app, rt = fast_app()

def encode_key(k: str) -> str:
    """Encode the entity key so it can be used in a URL."""
    return quote(k, safe="")

def decode_key(k: str) -> str:
    return unquote(k)

def nav_bar():
    """Returns an FT component with a top nav bar."""
    return Nav(
        A("Home", href="/"),
        A("People", href="/people"),
        A("Events", href="/events"),
        A("Locations", href="/locations"),
        A("Organizations", href="/organizations"),
        Button("About", cls="secondary", style="margin-left:auto;", onclick="alert('Not implemented');"),
        style="display:flex; gap:1em; margin-bottom:1em; align-items:center;"
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
            nav_bar(),
            # We'll wrap the filter panel and the main content in a side-by-side arrangement
            Div(
                Div(filter_panel, style="flex:0 0 220px; border-right:1px solid #ccc; padding-right:1em;"),
                Div(content, style="flex:1; padding-left:1em;"),
                style="display:flex; gap:1em;"
            ),
        )
    )

################################################################
#              Filter Panels (PLACEHOLDERS)
################################################################

def people_filter_panel():
    # For demonstration, we create a simple checkboxes list for people types
    # We'll guess some possible types, but in reality you'd gather them from the data
    possible_types = {"Detainee", "Military Personnel", "Lawyer", "Journalist"}
    checks = []
    for pt in sorted(possible_types):
        checks.append(
            Div(
                Input(type="checkbox", name="type", value=pt),
                Label(pt)
            )
        )
    return Form(
        H3("People Filters"),
        *checks,
        Label("Search: ", Input(type="text", name="q", placeholder="Name...")),
        Button("Apply Filters", type="submit"),
        method="get",
        action="/people"  # reloading the same page with the chosen filters
    )

def events_filter_panel():
    # Filter by event type & date range placeholders
    possible_types = {"Hearing", "Transfer", "CourtSession", "Misc"}
    checks = []
    for et in sorted(possible_types):
        checks.append(
            Div(
                Input(type="checkbox", name="etype", value=et),
                Label(et)
            )
        )
    return Form(
        H3("Event Filters"),
        P("Event Type:"),
        *checks,
        P("Date Range:"),
        Label("From:", Input(type="date", name="start_date")),
        Label("To:", Input(type="date", name="end_date")),
        Button("Apply Filters", type="submit"),
        method="get",
        action="/events"
    )

def locations_filter_panel():
    # Filter by location type
    possible_types = {"Country", "Facility", "Unknown", "Base"}
    checks = []
    for lt in sorted(possible_types):
        checks.append(
            Div(
                Input(type="checkbox", name="loc_type", value=lt),
                Label(lt)
            )
        )
    return Form(
        H3("Locations Filters"),
        *checks,
        Label("Search: ", Input(type="text", name="q", placeholder="Location name...")),
        Button("Apply Filters", type="submit"),
        method="get",
        action="/locations"
    )

def organizations_filter_panel():
    # Filter by org type
    possible_types = {"NGO", "Military", "GovAgency", "Unknown"}
    checks = []
    for ot in sorted(possible_types):
        checks.append(
            Div(
                Input(type="checkbox", name="org_type", value=ot),
                Label(ot)
            )
        )
    return Form(
        H3("Organization Filters"),
        *checks,
        Label("Search: ", Input(type="text", name="q", placeholder="Organization name...")),
        Button("Apply Filters", type="submit"),
        method="get",
        action="/organizations"
    )

################################################################
#             HOME
################################################################

@rt("/")
def get_home():
    """Home route: show quick welcome and links to each entity type."""
    return Titled(
        "GTMO Browse - Home",
        nav_bar(),
        H1("Welcome to Guantánamo Entities Browser"),
        P("Use the navigation above to browse the extracted entity data.")
    )

################################################################
#              People
################################################################

@rt("/people")
def list_people(request):
    q = request.query_params.get("q", "").strip().lower()
    selected_types = request.query_params.getlist("type")
    selected_types = [t.strip().lower() for t in selected_types if t.strip()]

    filtered_items = []
    for k, person in people_index.items():
        ptype = person.get("type", "").strip().lower()
        pname = person.get("name", "").strip().lower()

        # if there's a type filter, skip if not matching
        if selected_types and ptype not in selected_types:
            continue
        # optional search
        if q and q not in pname:
            continue

        link = A(person["name"], href=f"/people/{encode_key(k)}")
        filtered_items.append(Li(link))

    content = Div(
        H2("People"),
        Ul(*filtered_items),
        style="margin-top:1em;"
    )
    return main_layout("GTMO Browse - People", people_filter_panel(), content)

@rt("/people/{key:path}")
def show_person(key: str):
    """Display detail about one person."""
    actual_key = decode_key(key)
    person = people_index.get(actual_key)
    if not person:
        return Titled(
            "GTMO Browse - People - Not Found",
            Container(
                nav_bar(),
                H2("Person not found"),
                P(f"No person found for key: {actual_key}"),
            ),
        )
    name = person.get("name", "N/A")
    typ = person.get("type", "N/A")
    profile = person.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, person.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = person.get("articles", [])
    art_list = []
    for art in articles:
        art_list.append(
            Li(
                f"Article ID: {art.get('article_id', '')}, Title: {art.get('article_title', 'N/A')} ",
                A("(link)", href=art.get("article_url", "#"), target="_blank"),
            )
        )

    detail_content = Div(
        H2(f"Person: {name}"),
        P(f"Type: {typ}"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )

    # We can wrap detail pages in the same layout.
    # We'll pass an empty filter panel or minimal placeholder for detail pages.
    return main_layout(
        f"GTMO Browse - People - {name}",
        Div("No filters for detail pages."),
        detail_content
    )

################################################################
#              Events
################################################################

@rt("/events")
@rt("/events")
def list_events(request):
    import arrow

    def parse_dt(dt):
        if not dt: return None
        try:
            return arrow.get(dt)
        except:
            return None

    # gather query filters
    q = request.query_params.get("q", "").strip().lower()
    selected_types = request.query_params.getlist("etype")
    selected_types = [t.strip().lower() for t in selected_types if t.strip()]
    start_q = request.query_params.get("start_date", "")
    end_q = request.query_params.get("end_date", "")

    def rowinfo(k, ev):
        startdt = parse_dt(ev.get("start_date", ""))
        return (k, startdt, ev)

    all_rows = [rowinfo(k, e) for k, e in events_index.items()]

    # filter by type, date range, optional search on title
    filtered = []
    for (k, startdt, ev) in all_rows:
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
    for (k, startdt, ev) in filtered:
        start_str = startdt.format("YYYY-MM-DD") if startdt else "Unknown"
        link = A(ev["title"], href=f"/events/{encode_key(k)}")
        items.append(
            Li(
                f"{start_str} — ",
                link
            )
        )

    content = Div(
        H2("Events"),
        Ul(*items),
        style="margin-top:1em;"
    )
    return main_layout("GTMO Browse - Events", events_filter_panel(), content)

@rt("/events/{key:path}")
def show_event(key: str):
    actual_key = decode_key(key)
    ev = events_index.get(actual_key)
    if not ev:
        return main_layout(
            "GTMO Browse - Events - Not Found",
            Div("No filters for detail pages."),
            Div(H2("Event not found"), P(f"No event found for key: {actual_key}"))
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
    art_list = []
    for art in articles:
        art_list.append(
            Li(
                f"Article ID: {art.get('article_id', '')}, Title: {art.get('article_title', 'N/A')} ",
                A("(link)", href=art.get("article_url", "#"), target="_blank"),
            )
        )

    detail_content = Div(
        H2(f"Event: {title}"),
        P(f"Type: {event_type}"),
        P(f"Start Date: {start}"),
        P(f"End Date: {end if end else 'None'}"),
        P(f"Is Fuzzy Date: {is_fuzzy}"),
        H3("Description"),
        P(desc),
        H3("Profile Text"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )
    return main_layout(f"GTMO Browse - Events - {title}", Div("No filters for detail pages."), detail_content)

################################################################
#              Locations
################################################################

@rt("/locations")
@rt("/locations")
def list_locations(request):

    q = request.query_params.get("q", "").strip().lower()
    selected_types = request.query_params.getlist("loc_type")
    selected_types = [t.strip().lower() for t in selected_types if t.strip()]

    filtered_items = []
    for k, loc in locations_index.items():
        ltype = loc.get("type", "").strip().lower()
        lname = loc.get("name", "").strip().lower()

        if selected_types and ltype not in selected_types:
            continue
        if q and q not in lname:
            continue

        link = A(loc["name"], href=f"/locations/{encode_key(k)}")
        filtered_items.append(Li(link))

    content = Div(
        H2("Locations"),
        Ul(*filtered_items),
        style="margin-top:1em;"
    )
    return main_layout("GTMO Browse - Locations", locations_filter_panel(), content)

@rt("/locations/{key:path}")
def show_location(key: str):
    actual_key = decode_key(key)
    loc = locations_index.get(actual_key)
    if not loc:
        return main_layout(
            "GTMO Browse - Locations - Not Found",
            Div("No filters for detail pages."),
            Div(H2("Location not found"), P(f"No location found for key: {actual_key}"))
        )
    nm = loc.get("name", "N/A")
    typ = loc.get("type", "N/A")
    profile = loc.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, loc.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = loc.get("articles", [])
    art_list = []
    for art in articles:
        art_list.append(
            Li(
                f"Article ID: {art.get('article_id', '')}, Title: {art.get('article_title', 'N/A')} ",
                A("(link)", href=art.get("article_url", "#"), target="_blank"),
            )
        )

    detail_content = Div(
        H2(f"Location: {nm}"),
        P(f"Type: {typ}"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )
    return main_layout(f"GTMO Browse - Locations - {nm}", Div("No filters for detail pages."), detail_content)

################################################################
#              Organizations
################################################################

@rt("/organizations")
@rt("/organizations")
def list_orgs(request):

    q = request.query_params.get("q", "").strip().lower()
    selected_types = request.query_params.getlist("org_type")
    selected_types = [t.strip().lower() for t in selected_types if t.strip()]

    filtered_items = []
    for k, org in orgs_index.items():
        otype = org.get("type", "").strip().lower()
        oname = org.get("name", "").strip().lower()

        if selected_types and otype not in selected_types:
            continue
        if q and q not in oname:
            continue

        link = A(org["name"], href=f"/organizations/{encode_key(k)}")
        filtered_items.append(Li(link))

    content = Div(
        H2("Organizations"),
        Ul(*filtered_items),
        style="margin-top:1em;"
    )
    return main_layout("GTMO Browse - Organizations", organizations_filter_panel(), content)

@rt("/organizations/{key:path}")
def show_org(key: str):
    actual_key = decode_key(key)
    org = orgs_index.get(actual_key)
    if not org:
        return main_layout(
            "GTMO Browse - Organizations - Not Found",
            Div("No filters for detail pages."),
            Div(H2("Organization not found"), P(f"No organization found for key: {actual_key}"))
        )
    nm = org.get("name", "N/A")
    typ = org.get("type", "N/A")
    profile = org.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, org.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = org.get("articles", [])
    art_list = []
    for art in articles:
        art_list.append(
            Li(
                f"Article ID: {art.get('article_id', '')}, Title: {art.get('article_title', 'N/A')} ",
                A("(link)", href=art.get("article_url", "#"), target="_blank"),
            )
        )

    detail_content = Div(
        H2(f"Organization: {nm}"),
        P(f"Type: {typ}"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )
    return main_layout(f"GTMO Browse - Organizations - {nm}", Div("No filters for detail pages."), detail_content)

################################################################
#              Run
################################################################

if __name__ == "__main__":
    serve()