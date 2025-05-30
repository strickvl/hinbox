import logging

import markdown
from fasthtml.common import *

from src.config_loader import DomainConfig
from src.constants import ENABLE_PROFILE_VERSIONING
from src.profiles import VersionedProfile
from src.utils.error_handler import ErrorHandler

from ..app_config import (
    get_current_domain,
    main_layout,
    rt,
)
from ..components import ProfileVersionSelector
from ..data_access import build_indexes, get_domain_data
from ..filters import people_filter_panel
from ..utils import decode_key, encode_key, format_article_list, transform_profile_text

logger = logging.getLogger(__name__)


def filter_people(people_index, q="", selected_types=None, selected_tags=None):
    """Filter people based on search criteria."""
    selected_types = selected_types or []
    selected_tags = selected_tags or []

    filtered = []
    for k, person in people_index.items():
        # Search filter
        if q and q not in person.get("name", "").strip().lower():
            continue

        # Type filter
        ptype = person.get("type", "").strip().lower()
        if selected_types and ptype not in [t.lower() for t in selected_types]:
            continue

        # Tag filter
        person_tags = [
            t.strip().lower() for t in person.get("profile", {}).get("tags", [])
        ]
        if selected_tags and not any(t.lower() in person_tags for t in selected_tags):
            continue

        filtered.append((k, person))

    return filtered


def render_people_list(filtered_people):
    """Render the people list with consistent styling."""
    if not filtered_people:
        return Div(
            "No people match your filters. Try adjusting your criteria.",
            cls="empty-state",
        )

    items = []
    for k, person in filtered_people:
        type_badge = ""
        if person.get("type"):
            type_badge = Span(person.get("type"), cls="tag")
        link = A(person["name"], href=f"/people/{encode_key(k)}")
        items.append(Li(link, " ", type_badge))

    return Div(
        Div(
            f"{len(filtered_people)} people found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*items),
        id="results",
    )


def get_page_title(page_name: str, domain: str = "guantanamo") -> str:
    """Get domain-aware page title."""
    try:
        config = DomainConfig(domain)
        domain_config = config.load_config()
        domain_name = domain_config.get("domain", domain).title()
        return f"{domain_name} Browse - {page_name}"
    except Exception:
        return f"Research Browse - {page_name}"


@rt("/people")
def list_people(request):
    error_handler = ErrorHandler("people_list", {"route": "/people"})

    try:
        current_domain = get_current_domain(request)
        domain_data = get_domain_data(current_domain)
    except Exception as e:
        error_handler.log_error(e, "error")
        return main_layout(
            "Error - People",
            Div(),
            Div(
                H2("Error Loading Data", style="color:var(--danger);"),
                P(
                    "There was an error loading the people data. Please try again later."
                ),
                A("← Back to Home", href="/", cls="primary"),
            ),
            page_header_title="Error",
        )

    if not domain_data["people"]:
        content = Div(
            Div(
                "No data available",
                style="margin-bottom:15px; color:var(--text-light);",
            ),
            Div(
                f"No people data found for the '{current_domain}' domain. You may need to process articles for this domain first.",
                cls="empty-state",
            ),
        )
        return main_layout(
            get_page_title("People"),
            Div(),  # Empty filter panel
            content,
            page_header_title="People",
            current_domain=current_domain,
        )

    try:
        # Build indexes for this domain
        domain_indexes = build_indexes(domain_data)
        people_index = domain_indexes["people"]

        # Get filters from request
        q = request.query_params.get("q", "").strip()
        selected_types_raw = request.query_params.getlist("type")
        selected_types = [t.strip() for t in selected_types_raw if t.strip()]
        selected_tags_raw = request.query_params.getlist("tag")
        selected_tags = [t.strip() for t in selected_tags_raw if t.strip()]

        # Apply filters
        filtered_people = filter_people(
            people_index, q.lower(), selected_types, selected_tags
        )

        # Render results
        content = render_people_list(filtered_people)

        # Return full page or partial based on HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            return content
        else:
            return main_layout(
                get_page_title("People"),
                people_filter_panel(
                    q=q, selected_types=selected_types, selected_tags=selected_tags
                ),
                content,
                page_header_title="People",
                current_domain=current_domain,
            )

    except Exception as e:
        error_handler.log_error(e, "error")
        error_content = Div(
            H2("Error Processing Request", style="color:var(--danger);"),
            P("There was an error processing your request. Please try again later."),
            A("← Back to People", href="/people", cls="primary"),
        )

        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            return error_content
        else:
            return main_layout(
                get_page_title("People - Error"),
                Div(),
                error_content,
                page_header_title="Error",
                current_domain=current_domain,
            )


@rt("/people/{key:path}")
def show_person(key: str, request):
    error_handler = ErrorHandler(
        "person_detail", {"route": f"/people/{key}", "key": key}
    )

    try:
        current_domain = get_current_domain(request)

        # Load domain-specific data
        domain_data = get_domain_data(current_domain)
    except Exception as e:
        error_handler.log_error(e, "error")
        # Return error page
        return main_layout(
            "Error - Person",
            Div(),
            Div(
                H2("Error Loading Data", style="color:var(--danger);"),
                P(
                    "There was an error loading the person data. Please try again later."
                ),
                A("← Back to People", href="/people", cls="primary"),
            ),
            page_header_title="Error",
        )
    if not domain_data["people"]:
        return main_layout(
            get_page_title("People - Not Found"),
            Div("No filters for detail pages."),
            Div(
                H2("No Data Available", style="color:var(--danger);"),
                P(f"No people data found for the '{current_domain}' domain."),
                A(
                    "← Back to Home",
                    href=f"/?domain={current_domain}",
                    cls="primary",
                ),
            ),
            current_domain=current_domain,
        )

    # Build indexes for this domain
    domain_indexes = build_indexes(domain_data)
    people_index = domain_indexes["people"]

    actual_key = decode_key(key)
    person = people_index.get(actual_key)
    if not person:
        return main_layout(
            get_page_title("People - Not Found"),
            Div("No filters for detail pages."),
            Div(
                H2("Person not found", style="color:var(--danger);"),
                P(f"No person found with the name: {actual_key}"),
                A(
                    "← Back to People",
                    href=f"/people?domain={current_domain}",
                    cls="primary",
                ),
            ),
            current_domain=current_domain,
        )

    name = person.get("name", "N/A")
    typ = person.get("type", "N/A")
    profile = person.get("profile", {})

    # Handle version selection for profile versioning
    requested_version = None
    version_selector = ""

    if ENABLE_PROFILE_VERSIONING and "profile_versions" in person:
        try:
            versioned_profile = VersionedProfile(**person["profile_versions"])

            # Check for version parameter
            version_param = request.query_params.get("version")
            if version_param:
                try:
                    requested_version = int(version_param)
                    if 1 <= requested_version <= versioned_profile.current_version:
                        # Use specific version
                        version_data = versioned_profile.get_version(requested_version)
                        if version_data:
                            profile = version_data.profile_data
                    else:
                        # Invalid version, redirect to current
                        from fasthtml.common import Response

                        return Response(
                            status_code=302, headers={"Location": f"/people/{key}"}
                        )
                except (ValueError, TypeError):
                    # Invalid version format, redirect to current
                    from fasthtml.common import Response

                    return Response(
                        status_code=302, headers={"Location": f"/people/{key}"}
                    )

            # Create version selector
            version_selector = ProfileVersionSelector(
                entity_name=name,
                entity_type="people",
                current_version=versioned_profile.current_version,
                total_versions=len(versioned_profile.versions),
                route_prefix="people",
                entity_key=actual_key,
                selected_version=requested_version,
            )
        except Exception as e:
            # Log error but continue with current profile
            logger.warning(f"Error loading profile versions for {name}: {e}")

    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, person.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = person.get("articles", [])

    tags = profile.get("tags", [])
    tag_elements = []
    for tag in tags:
        if tag.strip():
            tag_elements.append(Span(tag, cls="tag"))

    detail_content = Div(
        version_selector,  # Add version selector at the top
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
        format_article_list(articles, text),
        cls="entity-detail",
    )

    return main_layout(
        get_page_title(f"People - {name}"),
        Div(
            H2("Navigation"),
            A(
                "← Back to People",
                href=f"/people?domain={current_domain}",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
        page_header_title=name,
        current_domain=current_domain,
    )
