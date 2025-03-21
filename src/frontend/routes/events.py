import arrow
import markdown
from fasthtml.common import H2, A, Div, Li, NotStr, P, Span, Ul

from ..app_config import main_layout, rt
from ..data_access import events_index
from ..filters import events_filter_panel
from ..utils import decode_key, encode_key, format_article_list, transform_profile_text


@rt("/events")
def list_events(request):
    def parse_dt(dt):
        if not dt:
            return None
        try:
            return arrow.get(dt)
        except:
            return None

    q = request.query_params.get("q", "").strip().lower()
    selected_types_raw = request.query_params.getlist("etype")
    selected_types = [t.strip().lower() for t in selected_types_raw if t.strip()]
    start_q = request.query_params.get("start_date", "")
    end_q = request.query_params.get("end_date", "")

    def rowinfo(k, ev):
        startdt = parse_dt(ev.get("start_date", ""))
        return (k, startdt, ev)

    all_rows = [rowinfo(k, e) for k, e in events_index.items()]

    filtered = []
    for k, startdt, ev in all_rows:
        evtype = ev.get("event_type", "").strip().lower()
        title_lower = ev.get("title", "").strip().lower()
        keep = True

        if selected_types:
            if not set(selected_types).issubset({evtype}):
                keep = False
        if q and q not in title_lower:
            keep = False
        if keep and start_q:
            try:
                filter_start = arrow.get(start_q)
                if startdt and startdt < filter_start:
                    keep = False
            except:
                pass
        if keep and end_q:
            try:
                filter_end = arrow.get(end_q)
                if startdt and startdt > filter_end:
                    keep = False
            except:
                pass
        if keep:
            filtered.append((k, startdt, ev))

    filtered.sort(key=lambda x: (x[1].timestamp() if x[1] else 0))
    items = []
    for k, startdt, ev in filtered:
        start_str = startdt.format("YYYY-MM-DD") if startdt else "Unknown"
        event_type = ev.get("event_type", "")
        type_badge = ""
        if event_type:
            from fasthtml.common import Span

            type_badge = Span(event_type, cls="tag")

        link = A(ev["title"], href=f"/events/{encode_key(k)}")
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

    content = Div(
        H2("Events"),
        Div(
            f"{len(filtered)} events found",
            style="margin-bottom:15px; color:var(--text-light);",
        ),
        Ul(*items)
        if items
        else Div(
            "No events match your filters. Try adjusting your criteria.",
            cls="empty-state",
        ),
    )

    is_htmx = request.headers.get("HX-Request") == "true"
    if is_htmx:
        return content
    else:
        return main_layout(
            "GTMO Browse - Events",
            events_filter_panel(q, selected_types, start_q, end_q),
            content,
        )


@rt("/events/{key:path}")
def show_event(key: str):
    actual_key = decode_key(key)
    ev = events_index.get(actual_key)
    if not ev:
        return main_layout(
            "GTMO Browse - Events - Not Found",
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
        H2(f"Event: {title}"),
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
        f"GTMO Browse - Events - {title}",
        Div(
            H2("Navigation"),
            A(
                "← Back to Events",
                href="/events",
                cls="primary",
                style="display:block; margin-bottom:10px;",
            ),
            style="margin-bottom:20px;",
        ),
        detail_content,
    )
