import markdown
from fasthtml.common import H2, A, Div, Li, NotStr, P, Span, Ul

from src.config_loader import DomainConfig

from ..app_config import get_current_domain, main_layout, rt
from ..data_access import build_indexes, get_domain_data
from ..filters import locations_filter_panel
from ..utils import decode_key, encode_key, format_article_list, transform_profile_text


def get_page_title(page_name: str, domain: str = "guantanamo") -> str:
    """Get domain-aware page title."""
    try:
        config = DomainConfig(domain)
        domain_config = config.load_config()
        domain_name = domain_config.get("domain", domain).title()
        return f"{domain_name} Browse - {page_name}"
    except Exception:
        return f"Research Browse - {page_name}"


@rt("/locations")
def list_locations(request):
    current_domain = get_current_domain(request)

    # Load domain-specific data
    domain_data = get_domain_data(current_domain)
    if not domain_data["locations"]:
        # No data for this domain
        content = Div(
            Div(
                "No data available",
                style="margin-bottom:15px; color:var(--text-light);",
            ),
            Div(
                f"No locations data found for the '{current_domain}' domain. You may need to process articles for this domain first.",
                cls="empty-state",
            ),
        )
        return main_layout(
            get_page_title("Locations"),
            Div(),  # Empty filter panel
            content,
            page_header_title="Locations",
            current_domain=current_domain,
        )

    # Build indexes for this domain
    domain_indexes = build_indexes(domain_data)
    locations_index = domain_indexes["locations"]

    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("loc_type")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]

    filtered_items = []
    for k, loc in locations_index.items():
        ltype = loc.get("type", "").strip().lower()
        lname = loc.get("name", "").strip().lower()

        if selected_types:
            if not set(selected_types).issubset({ltype}):
                continue
        if q and q not in lname:
            continue

        type_badge = ""
        if loc.get("type"):
            type_badge = Span(loc.get("type"), cls="tag")

        link = A(loc["name"], href=f"/locations/{encode_key(k)}")
        filtered_items.append(Li(link, " ", type_badge))

    content = Div(
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

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return content
    else:
        return main_layout(
            get_page_title("Locations"),
            locations_filter_panel(q=q, selected_types=selected_types),
            content,
            page_header_title="Locations",
            current_domain=current_domain,
        )


@rt("/locations/{key:path}")
def show_location(key: str, request):
    current_domain = get_current_domain(request)

    # Load domain-specific data
    domain_data = get_domain_data(current_domain)
    if not domain_data["locations"]:
        # No data for this domain
        return main_layout(
            get_page_title("Locations - Not Found"),
            Div("No filters for detail pages."),
            Div(
                H2("No locations data", style="color:var(--danger);"),
                P(
                    f"No locations data found for the '{current_domain}' domain. You may need to process articles for this domain first."
                ),
                A("← Back to Locations", href="/locations", cls="primary"),
            ),
            current_domain=current_domain,
        )

    # Build indexes for this domain
    domain_indexes = build_indexes(domain_data)
    locations_index = domain_indexes["locations"]

    actual_key = decode_key(key)
    loc = locations_index.get(actual_key)
    if not loc:
        return main_layout(
            get_page_title("Locations - Not Found"),
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
        Div(
            Span("Type: ", style="font-weight:bold;"),
            Span(typ, cls="tag"),
            style="margin-bottom:20px;",
        ),
        H2("Profile Information", style="margin-bottom:5px; font-size:1.25rem;"),
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
        H2("Related Articles", style="margin-top:25px; font-size:1.25rem;"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        get_page_title(f"Locations - {nm}"),
        Div(
            H2("Navigation"),
            A(
                "← Back to Locations",
                href=f"/locations?domain={current_domain}",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
        page_header_title=nm,
        current_domain=current_domain,
    )
