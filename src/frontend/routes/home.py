from fasthtml.common import H2, A, Div, P

from src.config_loader import DomainConfig

from ..app_config import get_current_domain, nav_bar, rt, titled_with_domain_picker


@rt("/")
def get_home(domain: str = None, request=None):
    """Home route: show quick welcome and links to each entity type."""

    # Update current domain if provided
    current_domain = domain or get_current_domain(request)

    # Get domain configuration
    try:
        config = DomainConfig(current_domain)
        domain_config = config.load_config()
        domain_title = domain_config.get("description", "Entity Browser")
    except Exception:
        domain_title = "Entity Browser"

    return titled_with_domain_picker(
        "Hinbox Local Browser",
        current_domain,
        [
            nav_bar(current_domain),
            Div(
                Div(
                    H2("Browse Entities"),
                    P(
                        f"Explore entities extracted from documents related to {domain_title.lower()}."
                    ),
                    style="text-align:center; margin-bottom:30px;",
                ),
                Div(
                    Div(
                        H2("People", style="color:var(--primary);"),
                        P("Browse individuals mentioned in the research documents."),
                        A(
                            "View People →",
                            href=f"/people?domain={current_domain}",
                            cls="primary",
                            style="display:inline-block; margin-top:10px;",
                        ),
                        style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                    ),
                    Div(
                        H2("Events", style="color:var(--primary);"),
                        P("Timeline of significant events mentioned in the documents."),
                        A(
                            "View Events →",
                            href=f"/events?domain={current_domain}",
                            cls="primary",
                            style="display:inline-block; margin-top:10px;",
                        ),
                        style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                    ),
                    Div(
                        H2("Locations", style="color:var(--primary);"),
                        P("Explore locations mentioned in the research documents."),
                        A(
                            "View Locations →",
                            href=f"/locations?domain={current_domain}",
                            cls="primary",
                            style="display:inline-block; margin-top:10px;",
                        ),
                        style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                    ),
                    Div(
                        H2("Organizations", style="color:var(--primary);"),
                        P("Organizations mentioned in the research documents."),
                        A(
                            "View Organizations →",
                            href=f"/organizations?domain={current_domain}",
                            cls="primary",
                            style="display:inline-block; margin-top:10px;",
                        ),
                        style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                    ),
                    style="display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:20px;",
                ),
                style="background-color: var(--card); border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);",
            ),
        ],
    )
