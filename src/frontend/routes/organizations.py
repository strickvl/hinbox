import markdown
from fasthtml.common import H2, A, Div, Li, NotStr, P, Span, Ul

from ..app_config import main_layout, rt
from ..data_access import orgs_index
from ..filters import organizations_filter_panel
from ..utils import decode_key, encode_key, format_article_list, transform_profile_text


@rt("/organizations")
def list_orgs(request):
    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("org_type")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]

    filtered_items = []
    for k, org in orgs_index.items():
        otype = org.get("type", "").strip().lower()
        oname = org.get("name", "").strip().lower()

        if selected_types:
            if not set(selected_types).issubset({otype}):
                continue
        if q and q not in oname:
            continue

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

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return content
    else:
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
        H2("Profile Information", style="margin-bottom:5px; font-size:1.25rem;"),
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
        H2("Related Articles", style="margin-top:25px; font-size:1.25rem;"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        f"GTMO Browse - Organizations - {nm}",
        Div(
            H2("Navigation"),
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
