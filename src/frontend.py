import hashlib
import os
from urllib.parse import quote, unquote

import markdown
import pyarrow.parquet as pq

# We'll use FastHTML for building our small web server
from fasthtml.common import *

DATA_DIR = "data/entities"

# Filenames for each entity type (now using Parquet)
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


# We'll store each entity in an in-memory dictionary for easy lookup by a unique key.
people_data = load_parquet(PEOPLE_FILE)
events_data = load_parquet(EVENTS_FILE)
locations_data = load_parquet(LOCATIONS_FILE)
orgs_data = load_parquet(ORGS_FILE)


# For each entity, we generate a "key" for referencing in routes
# For People, we'll just use the person's name as the key
def make_person_key(person: dict) -> str:
    return person.get("name", "")


# For Events, use "title + start_date" or just a hashed approach to ensure uniqueness
def make_event_key(event: dict) -> str:
    # If there's a valid start_date, we might incorporate it:
    # e.g. "Title##2024-07-31T00:00:00+00:00"
    # But we'll also keep a fallback for missing start_date or duplicates with a short hash
    title = event.get("title", "")
    start = event.get("start_date", "")
    combined = f"{title}##{start}"
    # We'll make a short hash for final uniqueness
    h = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"{title} ({h})"


# For Locations, use the "name + type" or just name if it's unique
def make_location_key(loc: dict) -> str:
    nm = loc.get("name", "")
    t = loc.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"{nm} ({h})"


# For Organizations, similarly
def make_org_key(org: dict) -> str:
    nm = org.get("name", "")
    t = org.get("type", "")
    combined = f"{nm}##{t}"
    h = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"{nm} ({h})"


# Now, let's create dictionaries for each set of data
people_index = {}
for p in people_data:
    k = make_person_key(p)
    # If there's a possibility of duplicates, we could add a check. But let's keep it simple:
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
#                 Build a FastHTML app
################################################################

app, rt = fast_app()


def nav_bar():
    """Returns an FT component with a nav bar linking to the 4 entity types."""
    # We'll generate a simple nav bar with anchors
    return Nav(
        A("People", href="/people"),
        A("Events", href="/events"),
        A("Locations", href="/locations"),
        A("Organizations", href="/organizations"),
        style="display:flex; gap:1em; margin-bottom:1em;",
    )


@rt("/")
def get():
    """Home route: show quick welcome and links to each entity type."""
    return Container(
        nav_bar(),
        H1("Welcome to Guantánamo Entities Browser"),
        P("Use the navigation above to browse the extracted entity data."),
    )


################################################################
#              Generic Utility: Link generation
################################################################


def encode_key(k: str) -> str:
    """Encode the entity key so it can be used in a URL."""
    return quote(k, safe="")


def decode_key(k: str) -> str:
    return unquote(k)


################################################################
#              People
################################################################


@rt("/people")
def list_people():
    """List all people entities."""
    # We'll turn each person into a link
    # e.g. /people/{encoded_key}
    items = []
    for k, person in people_index.items():
        link = A(person["name"], href=f"/people/{encode_key(k)}")
        items.append(Li(link))
    return Container(nav_bar(), H2("People"), Ul(*items))


@rt("/people/{key:path}")
def show_person(key: str):
    """Display detail about one person."""
    actual_key = decode_key(key)
    person = people_index.get(actual_key)
    if not person:
        return Container(
            nav_bar(),
            H2("Person not found"),
            P(f"No person found for key: {actual_key}"),
        )
    # Build detail info
    name = person.get("name", "N/A")
    typ = person.get("type", "N/A")
    profile = person.get("profile", {})
    text = profile.get("text", "")
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
    return Container(
        nav_bar(),
        H2(f"Person: {name}"),
        P(f"Type: {typ}"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )


################################################################
#              Events
################################################################


@rt("/events")
def list_events():
    """List all event entities."""
    items = []
    for k, e in events_index.items():
        link = A(e["title"], href=f"/events/{encode_key(k)}")
        items.append(Li(link))
    return Container(nav_bar(), H2("Events"), Ul(*items))


@rt("/events/{key:path}")
def show_event(key: str):
    actual_key = decode_key(key)
    ev = events_index.get(actual_key)
    if not ev:
        return Container(
            nav_bar(), H2("Event not found"), P(f"No event found for key: {actual_key}")
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
    return Container(
        nav_bar(),
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


################################################################
#              Locations
################################################################


@rt("/locations")
def list_locations():
    """List all location entities."""
    items = []
    for k, loc in locations_index.items():
        link = A(loc["name"], href=f"/locations/{encode_key(k)}")
        items.append(Li(link))
    return Container(nav_bar(), H2("Locations"), Ul(*items))


@rt("/locations/{key:path}")
def show_location(key: str):
    actual_key = decode_key(key)
    loc = locations_index.get(actual_key)
    if not loc:
        return Container(
            nav_bar(),
            H2("Location not found"),
            P(f"No location found for key: {actual_key}"),
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
    return Container(
        nav_bar(),
        H2(f"Location: {nm}"),
        P(f"Type: {typ}"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )


################################################################
#              Organizations
################################################################


@rt("/organizations")
def list_orgs():
    """List all organization entities."""
    items = []
    for k, org in orgs_index.items():
        link = A(org["name"], href=f"/organizations/{encode_key(k)}")
        items.append(Li(link))
    return Container(nav_bar(), H2("Organizations"), Ul(*items))


@rt("/organizations/{key:path}")
def show_org(key: str):
    actual_key = decode_key(key)
    org = orgs_index.get(actual_key)
    if not org:
        return Container(
            nav_bar(),
            H2("Organization not found"),
            P(f"No organization found for key: {actual_key}"),
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
    return Container(
        nav_bar(),
        H2(f"Organization: {nm}"),
        P(f"Type: {typ}"),
        Div(NotStr(markdown.markdown(transformed_text))),
        P(f"Confidence: {conf}"),
        H3("Articles"),
        Ul(*art_list),
    )


################################################################
#              Run
################################################################

if __name__ == "__main__":
    serve()
