import arrow
import markdown
from fasthtml.common import H2, A, Div, Li, NotStr, P, Span, Ul

from src.config_loader import DomainConfig
from src.utils.error_handler import ErrorHandler

from ..app_config import get_current_domain, main_layout, rt
from ..data_access import build_indexes, get_domain_data
from ..filters import events_filter_panel
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


def parse_event_date(dt):
    """Parse event date string to Arrow object."""
    if not dt:
        return None
    try:
        return arrow.get(dt)
    except:
        return None


def filter_events(events_index, q="", selected_types=None, start_date="", end_date=""):
    """Filter events with date range support."""
    selected_types = selected_types or []

    # Parse date filters
    filter_start = None
    filter_end = None
    if start_date:
        try:
            filter_start = arrow.get(start_date)
        except:
            pass
    if end_date:
        try:
            filter_end = arrow.get(end_date)
        except:
            pass

    # Create event tuples with parsed dates
    all_events = []
    for k, event in events_index.items():
        start_dt = parse_event_date(event.get("start_date", ""))
        all_events.append((k, start_dt, event))

    # Apply filters
    filtered = []
    for k, start_dt, event in all_events:
        # Search filter
        if q and q not in event.get("title", "").strip().lower():
            continue

        # Type filter
        event_type = event.get("event_type", "").strip().lower()
        if selected_types and event_type not in [t.lower() for t in selected_types]:
            continue

        # Date range filters
        if filter_start and start_dt and start_dt < filter_start:
            continue
        if filter_end and start_dt and start_dt > filter_end:
            continue

        filtered.append((k, start_dt, event))

    return filtered


def sort_events_by_date(filtered_events):
    """Sort events chronologically."""
    return sorted(filtered_events, key=lambda x: (x[1].timestamp() if x[1] else 0))


def render_events_list(filtered_events):
    """Render events list with date formatting."""
    if not filtered_events:
        return Div(
            "No events match your filters. Try adjusting your criteria.",
            cls="empty-state",
        )

    # Sort events by date
    sorted_events = sort_events_by_date(filtered_events)

    items = []
    for k, start_dt, event in sorted_events:
        start_str = start_dt.format("YYYY-MM-DD") if start_dt else "Unknown"
        event_type = event.get("event_type", "")

        type_badge = ""
        if event_type:
            type_badge = Span(event_type, cls="tag")

        link = A(event["title"], href=f"/events/{encode_key(k)}")
        items.append(
            Li(
                Div(
                    Span(
                        start_str,
                        style="font-weight:bold; margin-right:10px; color:var(--primary);",
                    ),
                    link,
                    style="display:flex; align-items:center;",
                ),
                Div(type_badge, style="margin-top:5px;") if type_badge else "",
            )
        )

    return Div(
        Div(
            f"{len(filtered_events)} events found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*items),
        id="results",
    )


@rt("/events")
def list_events(request):
    error_handler = ErrorHandler("events_list", {"route": "/events"})

    try:
        current_domain = get_current_domain(request)
        domain_data = get_domain_data(current_domain)
    except Exception as e:
        error_handler.log_error(e, "error")
        return main_layout(
            "Error - Events",
            Div(),
            Div(
                H2("Error Loading Data", style="color:var(--danger);"),
                P(
                    "There was an error loading the events data. Please try again later."
                ),
                A("← Back to Home", href="/", cls="primary"),
            ),
            page_header_title="Error",
        )

    if not domain_data["events"]:
        # No data for this domain
        content = Div(
            Div(
                "No data available",
                style="margin-bottom:15px; color:var(--text-light);",
            ),
            Div(
                f"No events data found for the '{current_domain}' domain. You may need to process articles for this domain first.",
                cls="empty-state",
            ),
        )
        return main_layout(
            get_page_title("Events"),
            Div(),  # Empty filter panel
            content,
            page_header_title="Events",
            current_domain=current_domain,
        )

    try:
        # Build indexes for this domain
        domain_indexes = build_indexes(domain_data)
        events_index = domain_indexes["events"]

        # Get filters from request
        q = request.query_params.get("q", "").strip()
        selected_types_raw = request.query_params.getlist("etype")
        selected_types = [t.strip() for t in selected_types_raw if t.strip()]
        start_date = request.query_params.get("start_date", "")
        end_date = request.query_params.get("end_date", "")

        # Apply filters using helper functions
        filtered_events = filter_events(
            events_index, q.lower(), selected_types, start_date, end_date
        )

        # Render results using helper function
        content = render_events_list(filtered_events)

        # Return full page or partial based on HTMX request
        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            return content
        else:
            return main_layout(
                get_page_title("Events"),
                events_filter_panel(q, selected_types, start_date, end_date),
                content,
                page_header_title="Events",
                current_domain=current_domain,
            )
    except Exception as e:
        error_handler.log_error(e, "error")
        error_content = Div(
            H2("Error Processing Request", style="color:var(--danger);"),
            P("There was an error processing your request. Please try again later."),
            A("← Back to Events", href="/events", cls="primary"),
        )

        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            return error_content
        else:
            return main_layout(
                get_page_title("Events - Error"),
                Div(),
                error_content,
                page_header_title="Error",
                current_domain=current_domain,
            )


@rt("/events/{key:path}")
def show_event(key: str, request):
    current_domain = get_current_domain(request)

    # Load domain-specific data
    domain_data = get_domain_data(current_domain)
    if not domain_data["events"]:
        # No data for this domain
        return main_layout(
            get_page_title("Events - Not Found"),
            Div("No filters for detail pages."),
            Div(
                H2("No events data", style="color:var(--danger);"),
                P(
                    f"No events data found for the '{current_domain}' domain. You may need to process articles for this domain first."
                ),
                A("← Back to Events", href="/events", cls="primary"),
            ),
            current_domain=current_domain,
        )

    # Build indexes for this domain
    domain_indexes = build_indexes(domain_data)
    events_index = domain_indexes["events"]

    actual_key = decode_key(key)
    ev = events_index.get(actual_key)
    if not ev:
        return main_layout(
            get_page_title("Events - Not Found"),
            Div("No filters for detail pages."),
            Div(
                H2("Event not found", style="color:var(--danger);"),
                P(f"No event found with the key: {actual_key}"),
                A("← Back to Events", href="/events", cls="primary"),
            ),
        )

    title = ev.get("title", "N/A")
    event_type = ev.get("event_type", "N/A")
    start = ev.get("start_date", "")
    end = ev.get("end_date", "")
    desc = ev.get("description", "")
    is_fuzzy = ev.get("is_fuzzy_date", False)
    profile = ev.get("profile", {})
    text = profile.get("text", "")
    transformed_text = transform_profile_text(text, ev.get("articles", []))
    conf = profile.get("confidence", "(none)")
    articles = ev.get("articles", [])

    detail_content = Div(
        Div(
            Span("Type: ", style="font-weight:bold;"),
            Span(event_type, cls="tag"),
            style="margin-bottom:15px;",
        ),
        Div(
            Div(
                Span("Start Date: ", style="font-weight:bold;"),
                Span(start if start else "Unknown"),
                style="margin-bottom:5px;",
            ),
            Div(
                Span("End Date: ", style="font-weight:bold;"),
                Span(end if end else "N/A"),
                style="margin-bottom:5px;",
            ),
            Div(
                Span("Date Precision: ", style="font-weight:bold;"),
                Span("Approximate" if is_fuzzy else "Exact"),
                style="font-style:italic; color:var(--text-light);",
            ),
            style="margin-bottom:20px; padding:10px; background:var(--highlight); border-radius:5px;",
        ),
        H2("Description", style="margin-bottom:5px; font-size:1.25rem;"),
        P(desc)
        if desc
        else P(
            "No description available.",
            style="font-style:italic; color:var(--text-light);",
        ),
        H2("Profile Information", style="margin-top:25px; font-size:1.25rem;"),
        Div(
            NotStr(markdown.markdown(transformed_text))
            if transformed_text
            else "No detailed profile information available for this event.",
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
        get_page_title(f"Events - {title}"),
        Div(
            H2("Navigation"),
            A(
                "← Back to Events",
                href=f"/events?domain={current_domain}",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
        page_header_title=title,
        current_domain=current_domain,
    )
