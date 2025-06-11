"""FastHTML application configuration and layout components.

This module sets up the core FastHTML application instance and provides reusable
layout components including navigation, domain switching, and page layout utilities.
The frontend supports multi-domain configurations with dynamic domain switching.
"""

from typing import Any, List, Optional, Union

from fasthtml.common import (
    H1,
    A,
    Body,
    Button,
    Div,
    Head,
    Html,
    Link,
    Meta,
    Nav,
    Option,
    Select,
    Title,
    fast_app,
)

from src.config_loader import DomainConfig

# Use fast_app() for automatic setup with better defaults
app, rt = fast_app(
    static_path="src/frontend/static",
    hdrs=(Link(rel="stylesheet", href="/styles.css", type="text/css"),),
)


# Domain management
def get_current_domain(request: Any = None) -> str:
    """Get the current domain from request query parameters or return default.

    Extracts the domain from the request's query parameters if available and valid,
    otherwise returns the default domain. Validates that the requested domain exists
    in the available domain configurations.

    Args:
        request: FastHTML request object with query_params attribute

    Returns:
        Domain name string, either from request or default "guantanamo"

    Note:
        Only returns domains that exist in DomainConfig.get_available_domains().
        Invalid or missing domains fall back to the default.
    """
    if request and hasattr(request, "query_params"):
        domain = request.query_params.get("domain")
        if domain and domain in DomainConfig.get_available_domains():
            return domain
    return "guantanamo"  # Default domain


def get_domain_description(domain: Optional[str] = None) -> str:
    """Get the description of the specified domain from its configuration.

    Loads the domain configuration and extracts the description field, providing
    a fallback description if the domain config is invalid or missing.

    Args:
        domain: Domain name to get description for, defaults to "guantanamo"

    Returns:
        Domain description string from config or fallback text

    Note:
        Returns "this research domain" as fallback for any configuration errors.
    """
    if not domain:
        domain = "guantanamo"
    try:
        config = DomainConfig(domain)
        domain_config = config.load_config()
        return domain_config.get("description", "this research domain")
    except Exception:
        return "this research domain"


def domain_switcher(current_domain: str = "guantanamo") -> Union[Div, Select]:
    """Create a domain switcher UI component (dropdown or label).

    Generates either a dropdown select for multiple domains or a simple label
    for single domain setups. Handles the case where no domains are configured.

    Args:
        current_domain: Currently selected domain name

    Returns:
        Div component (empty, label, or dropdown) for domain switching

    Note:
        - Returns empty Div if no domains configured
        - Returns styled label for single domain
        - Returns Select dropdown for multiple domains with onchange handler
    """
    try:
        available_domains = DomainConfig.get_available_domains()

        # If no domains, don't show anything
        if len(available_domains) == 0:
            return Div()

        options = []
        for domain in available_domains:
            # Use domain name as-is (no title case)
            selected = domain == current_domain
            options.append(Option(domain, value=domain, selected=selected))

        # If only one domain, show as compact label (just the domain name)
        if len(available_domains) == 1:
            return Div(
                available_domains[0],
                style="font-size: 0.9rem; padding: 4px 8px; background: var(--card); border: 1px solid var(--border); border-radius: 4px; color: var(--text); white-space: nowrap;",
            )

        # Multiple domains - show compact dropdown (just domain names)
        return Select(
            *options,
            onchange="window.location.href = '/?domain=' + this.value",
            style="font-size: 0.9rem; padding: 4px 8px; border: 1px solid var(--border); border-radius: 4px; background: white;",
        )
    except Exception:
        return Div()  # Return empty div if domain detection fails


def nav_bar(current_domain: str = "guantanamo") -> Nav:
    """Create the main navigation bar with domain-aware links.

    Generates a horizontal navigation bar with links to all main entity pages
    and an About button. All links include the current domain parameter.

    Args:
        current_domain: Domain name to include in navigation links

    Returns:
        Nav component with styled navigation links and About button

    Note:
        The About button shows an alert with domain-specific description.
    """
    return Nav(
        A("Home", href=f"/?domain={current_domain}"),
        A("People", href=f"/people?domain={current_domain}"),
        A("Events", href=f"/events?domain={current_domain}"),
        A("Locations", href=f"/locations?domain={current_domain}"),
        A("Organizations", href=f"/organizations?domain={current_domain}"),
        Button(
            "About",
            cls="secondary",
            style="margin-left:auto;",
            onclick=f"alert('Entity Browser helps researchers explore entities mentioned in documents related to {get_domain_description(current_domain)}.');",
        ),
    )


def page_header(title: str, current_domain: str = "guantanamo") -> Div:
    """Create a page header with title and domain switcher.

    Creates a flexbox header with the page title on the left and domain
    switcher on the right, with consistent styling and bottom border.

    Args:
        title: Page title text to display
        current_domain: Current domain for the domain switcher

    Returns:
        Div component containing styled header with title and domain switcher
    """
    return Div(
        H1(title, style="color:var(--primary); margin:0; flex:1;"),
        Div(
            domain_switcher(current_domain),
            style="margin-left:20px;",
        ),
        style="display:flex; align-items:center; margin-bottom:20px; padding-bottom:15px; border-bottom:2px solid var(--border);",
    )


def title_with_domain_picker(
    page_title: str, current_domain: str = "guantanamo"
) -> Div:
    """Create a title bar with domain picker in top right.

    Alternative to page_header with identical functionality but different name.
    Creates a flexbox layout with title and domain switcher.

    Args:
        page_title: Title text to display
        current_domain: Current domain for the domain switcher

    Returns:
        Div component with title and domain picker layout

    Note:
        This function duplicates page_header functionality and could be consolidated.
    """
    return Div(
        H1(page_title, style="color:var(--primary); margin:0; flex:1;"),
        Div(
            domain_switcher(current_domain),
            style="margin-left:20px;",
        ),
        style="display:flex; align-items:center; margin-bottom:20px; padding-bottom:15px; border-bottom:2px solid var(--border);",
    )


def titled_with_domain_picker(
    page_title: str, current_domain: str, children: List[Any]
) -> Html:
    """Create a complete HTML page with domain picker integrated into the title area.

    Generates a full HTML document with head, meta tags, and body containing
    a title bar with domain picker and the provided child content.

    Args:
        page_title: Full page title for browser title bar
        current_domain: Current domain for the domain switcher
        children: List of FastHTML components to include in the page body

    Returns:
        Html component representing complete HTML document

    Note:
        Extracts main title from "Domain Browse - Page" format for display.
        Includes responsive viewport meta tag and CSS stylesheet link.
    """
    # Extract just the main title from "Domain Browse - Page" format
    title_parts = page_title.split(" - ")
    main_title = title_parts[-1] if len(title_parts) > 1 else page_title

    return Html(
        Head(
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Title(page_title),
            Link(rel="stylesheet", href="/styles.css", type="text/css"),
        ),
        Body(
            Div(
                # Title bar with domain picker
                Div(
                    H1(main_title, style="color:var(--primary); margin:0; flex:1;"),
                    Div(
                        domain_switcher(current_domain),
                        style="margin-left:20px;",
                    ),
                    style="display:flex; align-items:center; margin-bottom:20px; padding-bottom:15px; border-bottom:2px solid var(--border);",
                ),
                *children,
            ),
            cls="container",
        ),
    )


def main_layout(
    page_title: str,
    filter_panel: Any,
    content: Any,
    page_header_title: Optional[str] = None,
    current_domain: str = "guantanamo",
) -> Html:
    """Create the main application layout with navigation, filter panel, and content area.

    Provides the standard layout structure used across all entity pages, with
    navigation bar, left sidebar filter panel, and main content area in a flexbox layout.

    Args:
        page_title: Full page title for browser title bar
        filter_panel: FastHTML component for the left sidebar filter controls
        content: FastHTML component for the main content area
        page_header_title: Optional custom header title (currently unused)
        current_domain: Current domain for navigation and domain switching

    Returns:
        Html component with complete page layout including navigation and content areas

    Note:
        Uses a responsive flexbox layout with fixed-width filter panel (220px)
        and flexible content area. The filter panel has HTMX targets for dynamic updates.
    """
    return titled_with_domain_picker(
        page_title,
        current_domain,
        [
            nav_bar(current_domain),
            Div(
                Div(
                    filter_panel,
                    cls="filter-panel",
                    style="flex:0 0 220px;",
                ),
                Div(
                    content,
                    cls="content-area",
                    style="flex:1; margin-left:20px;",
                ),
                style="display:flex; gap:20px;",
            ),
        ],
    )
