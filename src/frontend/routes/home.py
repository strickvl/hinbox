from fasthtml.common import H1, H2, A, Container, Div, P, Titled

from ..app_config import STYLES, nav_bar, rt


@rt("/")
def get_home():
    """Home route: show quick welcome and links to each entity type."""
    return Titled(
        "GTMO Browse - Home",
        Container(
            STYLES,
            nav_bar(),
            Div(
                H1(
                    "Guantánamo Entities Browser",
                    style="color:var(--primary); margin-bottom:30px; text-align:center;",
                ),
                Div(
                    Div(
                        H2("Browse Entities"),
                        P(
                            "Explore entities extracted from documents related to Guantánamo Bay."
                        ),
                        style="text-align:center; margin-bottom:30px;",
                    ),
                    Div(
                        Div(
                            H2("People", style="color:var(--primary);"),
                            P(
                                "Browse detainees, officials, legal representatives, and other individuals."
                            ),
                            A(
                                "View People →",
                                href="/people",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        Div(
                            H2("Events", style="color:var(--primary);"),
                            P(
                                "Timeline of hearings, transfers, policy changes, and other significant events."
                            ),
                            A(
                                "View Events →",
                                href="/events",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        Div(
                            H2("Locations", style="color:var(--primary);"),
                            P(
                                "Explore detention facilities, courtrooms, and other relevant locations."
                            ),
                            A(
                                "View Locations →",
                                href="/locations",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        Div(
                            H2("Organizations", style="color:var(--primary);"),
                            P(
                                "Agencies, military units, legal teams, and other organizations involved."
                            ),
                            A(
                                "View Organizations →",
                                href="/organizations",
                                cls="primary",
                                style="display:inline-block; margin-top:10px;",
                            ),
                            style="background:var(--card); border-radius:8px; padding:20px; box-shadow:0 2px 4px rgba(0,0,0,0.05);",
                        ),
                        style="display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:20px;",
                    ),
                    cls="content-area",
                ),
            ),
        ),
    )
