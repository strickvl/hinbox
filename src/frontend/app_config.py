import os

from fasthtml.common import A, Button, Div, Link, Nav, Option, Select, fast_app

from src.config_loader import DomainConfig

# Use absolute path to static files
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
app, rt = fast_app(static_path=static_dir)

# CSS is now served from static/styles.css
STYLES_LINK = Link(rel="stylesheet", href="/static/styles.css")

# Fallback inline styles for critical CSS
from fasthtml.common import Style

FALLBACK_STYLES = Style("""
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
    .filter-chip {
        display: inline-block;
        font-size: 0.8rem;
        padding: 3px 6px;
        border-radius: 16px;
        border: 1px solid #ddd;
        transition: transform 0.2s, color 0.2s;
        user-select: none;
        margin-right: 5px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    .filter-chip:hover {
        transform: scale(1.05);
    }
    .filter-chip.selected {
        background-color: #ffe98e !important;
        box-shadow: 0 0 0 1px var(--primary-light);
        color: #000;
        border: 1px solid var(--primary-light);
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
    .content-area a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 500;
    }
    .content-area a:hover {
        text-decoration: underline;
    }
""")


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
        style="display:flex; gap:1em; margin-bottom:1em; align-items:center;",
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
            STYLES_LINK,
            FALLBACK_STYLES,
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
                    style="flex:0 0 220px; background-color: var(--sidebar); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);",
                ),
                Div(
                    content,
                    cls="content-area",
                    style="flex:1; margin-left:20px; background-color: var(--card); border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);",
                ),
                style="display:flex; gap:20px;",
            ),
        ],
    )
