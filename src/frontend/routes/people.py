import markdown
from fasthtml.common import H2, A, Container, Div, Li, NotStr, P, Span, Titled, Ul

from ..app_config import main_layout, rt
from ..data_access import people_index
from ..filters import people_filter_panel
from ..utils import decode_key, encode_key, format_article_list, transform_profile_text


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

        if selected_types:
            if not set(selected_types).issubset({ptype}):
                continue
        if q and q not in pname:
            continue
        if selected_tags:
            if not set(selected_tags).issubset(set(p_tags)):
                continue

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

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return content
    else:
        return main_layout(
            "GTMO Browse - People",
            people_filter_panel(
                q=q, selected_types=selected_types, selected_tags=selected_tags
            ),
            content,
        )


@rt("/people/{key:path}")
def show_person(key: str):
    actual_key = decode_key(key)
    person = people_index.get(actual_key)
    if not person:
        return Titled(
            "GTMO Browse - People - Not Found",
            Container(
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

    tags = profile.get("tags", [])
    tag_elements = []
    from fasthtml.common import Span

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
        H2("Profile Information", style="margin-bottom:5px; font-size:1.25rem;"),
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
        H2("Related Articles", style="margin-top:25px; font-size:1.25rem;"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        f"GTMO Browse - People - {name}",
        Div(
            H2("Navigation"),
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
