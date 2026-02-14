from fasthtml.common import *

from src.config_loader import DomainConfig

from ..app_config import get_current_domain, nav_bar, rt, titled_with_domain_picker
from ..data_access import get_domain_data


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

    # Get entity counts for the current domain
    try:
        domain_data = get_domain_data(current_domain)
        counts = {
            "people": len(domain_data["people"]),
            "events": len(domain_data["events"]),
            "locations": len(domain_data["locations"]),
            "organizations": len(domain_data["organizations"]),
        }
    except Exception:
        counts = {"people": 0, "events": 0, "locations": 0, "organizations": 0}

    cards = [
        ("People", "people", "Browse individuals mentioned in the research documents."),
        (
            "Events",
            "events",
            "Timeline of significant events mentioned in the documents.",
        ),
        (
            "Locations",
            "locations",
            "Explore locations mentioned in the research documents.",
        ),
        (
            "Organizations",
            "organizations",
            "Organizations mentioned in the research documents.",
        ),
    ]

    return titled_with_domain_picker(
        "Hinbox Local Browser",
        current_domain,
        [
            nav_bar(current_domain),
            Div(
                H2("Browse Entities"),
                P(
                    f"Explore entities extracted from documents related to {domain_title.lower()}."
                ),
                style="text-align:center; margin-bottom:30px;",
            ),
            Div(
                *[
                    Div(
                        Span(str(counts[key]), cls="home-card-count"),
                        H3(label),
                        P(desc),
                        Span(f"View {label} \u2192", cls="card-link"),
                        cls="home-card",
                        onclick=f"window.location.href='/{key}?domain={current_domain}'",
                    )
                    for label, key, desc in cards
                ],
                cls="home-grid",
            ),
        ],
    )
