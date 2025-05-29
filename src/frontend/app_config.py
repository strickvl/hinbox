from fasthtml.common import A, Button, Div, Link, Nav, Option, Select, fast_app

from src.config_loader import DomainConfig

# Use fast_app() for automatic setup with better defaults
app, rt = fast_app(
    static_path="src/frontend/static",
    hdrs=(Link(rel="stylesheet", href="/styles.css", type="text/css"),),
)


# Domain management
def get_current_domain(request=None):
    """Get the current domain from request or default."""
    if request and hasattr(request, "query_params"):
        domain = request.query_params.get("domain")
        if domain and domain in DomainConfig.get_available_domains():
            return domain
    return "guantanamo"  # Default domain


def get_domain_description(domain=None):
    """Get the description of the specified domain."""
    if not domain:
        domain = "guantanamo"
    try:
        config = DomainConfig(domain)
        domain_config = config.load_config()
        return domain_config.get("description", "this research domain")
    except Exception:
        return "this research domain"


def domain_switcher(current_domain="guantanamo"):
    """Create a domain switcher dropdown."""
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


def nav_bar(current_domain="guantanamo"):
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


def page_header(title: str, current_domain: str = "guantanamo"):
    """Create a page header with title and domain switcher."""
    from fasthtml.common import H1, Div

    return Div(
        H1(title, style="color:var(--primary); margin:0; flex:1;"),
        Div(
            domain_switcher(current_domain),
            style="margin-left:20px;",
        ),
        style="display:flex; align-items:center; margin-bottom:20px; padding-bottom:15px; border-bottom:2px solid var(--border);",
    )


def title_with_domain_picker(page_title: str, current_domain: str = "guantanamo"):
    """Create a title bar with domain picker in top right."""
    from fasthtml.common import H1, Div

    return Div(
        H1(page_title, style="color:var(--primary); margin:0; flex:1;"),
        Div(
            domain_switcher(current_domain),
            style="margin-left:20px;",
        ),
        style="display:flex; align-items:center; margin-bottom:20px; padding-bottom:15px; border-bottom:2px solid var(--border);",
    )


def titled_with_domain_picker(page_title: str, current_domain: str, children):
    """Create a Titled page with domain picker integrated into the title area."""
    from fasthtml.common import H1, Body, Div, Head, Html, Meta, Title

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
    filter_panel,
    content,
    page_header_title: str = None,
    current_domain: str = "guantanamo",
):
    from fasthtml.common import Div

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
