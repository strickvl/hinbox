import logging

import markdown
from fasthtml.common import H2, A, Div, NotStr, P, Response, Span

from src.config_loader import DomainConfig
from src.constants import ENABLE_PROFILE_VERSIONING
from src.engine import VersionedProfile
from src.utils.error_handler import ErrorHandler

from ..app_config import get_current_domain, main_layout, rt
from ..components import (
    AliasesDisplay,
    ConfidenceBadge,
    ProfileVersionSelector,
    TagsDisplay,
)
from ..data_access import build_indexes, get_domain_data
from ..entity_helpers import filter_simple_entities, render_simple_entity_list
from ..filters import organizations_filter_panel
from ..utils import decode_key, format_article_list, transform_profile_text

logger = logging.getLogger(__name__)


def get_page_title(page_name: str, domain: str = "guantanamo") -> str:
    """Get domain-aware page title."""
    try:
        config = DomainConfig(domain)
        domain_config = config.load_config()
        domain_name = domain_config.get("domain", domain).title()
        return f"{domain_name} Browse - {page_name}"
    except Exception:
        return f"Research Browse - {page_name}"


@rt("/organizations")
def list_orgs(request):
    error_handler = ErrorHandler("organizations_list", {"route": "/organizations"})

    try:
        current_domain = get_current_domain(request)
        domain_data = get_domain_data(current_domain)
    except Exception as e:
        error_handler.log_error(e, "error")
        return main_layout(
            "Error - Organizations",
            Div(),
            Div(
                H2("Error Loading Data", style="color:var(--danger);"),
                P(
                    "There was an error loading the organizations data. Please try again later."
                ),
                A("← Back to Home", href="/", cls="primary"),
            ),
            page_header_title="Error",
        )

    if not domain_data["organizations"]:
        # No data for this domain
        content = Div(
            Div(
                "No data available",
                style="margin-bottom:15px; color:var(--text-light);",
            ),
            Div(
                f"No organizations data found for the '{current_domain}' domain. You may need to process articles for this domain first.",
                cls="empty-state",
            ),
        )
        return main_layout(
            get_page_title("Organizations"),
            Div(),  # Empty filter panel
            content,
            page_header_title="Organizations",
            current_domain=current_domain,
        )

    try:
        # Build indexes for this domain
        domain_indexes = build_indexes(domain_data)
        orgs_index = domain_indexes["organizations"]

        # Get filters from request
        q = request.query_params.get("q", "").strip()
        selected_types_raw = request.query_params.getlist("org_type")
        selected_types = [t.strip() for t in selected_types_raw if t.strip()]

        # Apply filters using generic helper
        filtered_organizations = filter_simple_entities(
            orgs_index, q.lower(), selected_types
        )

        # Render results using generic helper
        content = render_simple_entity_list(
            filtered_organizations, "organizations", "organizations"
        )

        # Return full page or partial based on HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            return content
        else:
            return main_layout(
                get_page_title("Organizations"),
                organizations_filter_panel(q=q, selected_types=selected_types),
                content,
                page_header_title="Organizations",
                current_domain=current_domain,
            )
    except Exception as e:
        error_handler.log_error(e, "error")
        error_content = Div(
            H2("Error Processing Request", style="color:var(--danger);"),
            P("There was an error processing your request. Please try again later."),
            A("← Back to Organizations", href="/organizations", cls="primary"),
        )

        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            return error_content
        else:
            return main_layout(
                get_page_title("Organizations - Error"),
                Div(),
                error_content,
                page_header_title="Error",
                current_domain=current_domain,
            )


@rt("/organizations/{key:path}")
def show_org(key: str, request):
    current_domain = get_current_domain(request)

    # Load domain-specific data
    domain_data = get_domain_data(current_domain)
    if not domain_data["organizations"]:
        # No data for this domain
        return main_layout(
            get_page_title("Organizations - Not Found"),
            Div("No filters for detail pages."),
            Div(
                H2("No organizations data", style="color:var(--danger);"),
                P(
                    f"No organizations data found for the '{current_domain}' domain. You may need to process articles for this domain first."
                ),
                A("← Back to Organizations", href="/organizations", cls="primary"),
            ),
            current_domain=current_domain,
        )

    # Build indexes for this domain
    domain_indexes = build_indexes(domain_data)
    orgs_index = domain_indexes["organizations"]

    actual_key = decode_key(key)
    org = orgs_index.get(actual_key)
    if not org:
        return main_layout(
            get_page_title("Organizations - Not Found"),
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

    # Handle version selection for profile versioning
    requested_version = None
    version_selector = ""

    if ENABLE_PROFILE_VERSIONING and "profile_versions" in org:
        try:
            versioned_profile = VersionedProfile(**org["profile_versions"])

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
                        return Response(
                            status_code=302,
                            headers={"Location": f"/organizations/{key}"},
                        )
                except (ValueError, TypeError):
                    # Invalid version format, redirect to current
                    return Response(
                        status_code=302, headers={"Location": f"/organizations/{key}"}
                    )

            # Create version selector
            version_selector = ProfileVersionSelector(
                entity_name=nm,
                entity_type="organizations",
                current_version=versioned_profile.current_version,
                total_versions=len(versioned_profile.versions),
                route_prefix="organizations",
                entity_key=actual_key,
                selected_version=requested_version,
            )
        except Exception as e:
            # Log error but continue with current profile
            logger.warning(f"Error loading profile versions for {nm}: {e}")

    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, org.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = org.get("articles", [])

    aliases = org.get("aliases", [])

    detail_content = Div(
        version_selector,  # Add version selector at the top
        Div(
            Span("Type: ", style="font-weight:bold;"),
            Span(typ, cls="tag"),
            style="margin-bottom:15px;",
        ),
        AliasesDisplay(aliases),
        TagsDisplay(profile.get("tags", [])),
        H2("Profile Information", style="margin-bottom:5px; font-size:1.25rem;"),
        Div(
            NotStr(markdown.markdown(transformed_text))
            if transformed_text
            else "No detailed profile information available for this organization.",
            cls="profile-text",
        ),
        ConfidenceBadge(conf),
        H2("Related Articles", style="margin-top:25px; font-size:1.25rem;"),
        format_article_list(articles),
        cls="entity-detail",
    )

    return main_layout(
        get_page_title(f"Organizations - {nm}"),
        Div(
            H2("Navigation"),
            A(
                "← Back to Organizations",
                href=f"/organizations?domain={current_domain}",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
        page_header_title=nm,
        current_domain=current_domain,
    )
