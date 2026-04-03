"""Annotation route for the relevance evaluation harness.

Provides a keyboard-driven interface for labelling articles as relevant,
irrelevant, or noisy (relevant topic but contains junk/sidebar text).
"""

from typing import Dict, List

from fasthtml.common import (
    H2,
    H3,
    A,
    Button,
    Div,
    P,
    Script,
    Small,
    Span,
    Strong,
    Textarea,
)
from starlette.requests import Request

from src.eval.relevance_candidates import RelevanceCandidate, load_relevance_candidates
from src.eval.relevance_labels import (
    RelevanceLabel,
    append_relevance_label,
    load_relevance_labels,
    make_relevance_label,
)

from ..app_config import get_current_domain, nav_bar, rt, titled_with_domain_picker

# Shared HTMX target — progress bar + article card live inside this wrapper
_SWAP_TARGET = "#eval-rel-swap-area"


def _progress_bar(labelled: int, total: int) -> Div:
    """Render progress bar showing annotation completion."""
    pct = (labelled / total * 100) if total > 0 else 0
    return Div(
        Div(
            Span(f"{labelled}/{total} labelled", cls="eval-progress-text"),
            Div(
                Div(
                    style=f"width:{pct:.0f}%; height:100%; background:var(--primary); border-radius:4px; transition:width 0.3s;",
                ),
                style="flex:1; height:8px; background:var(--surface-2); border-radius:4px; margin-left:12px;",
            ),
            style="display:flex; align-items:center; gap:8px;",
        ),
        cls="eval-progress",
    )


def _article_card(candidate: RelevanceCandidate) -> Div:
    """Render the article content panel."""
    title_el = (
        A(
            candidate.title,
            href=candidate.url,
            target="_blank",
            style="color:var(--primary-dark); text-decoration:none;",
        )
        if candidate.url
        else Strong(candidate.title, style="color:var(--primary-dark);")
    )

    meta_parts = []
    if candidate.content_length:
        meta_parts.append(f"{candidate.content_length:,} chars")
    if candidate.published_date:
        meta_parts.append(f"Published: {candidate.published_date}")

    return Div(
        H3(title_el, style="margin:0 0 12px;"),
        Small(
            " | ".join(meta_parts),
            style="color:var(--text-muted); display:block; margin-bottom:16px;",
        )
        if meta_parts
        else Span(),
        P(
            candidate.content_snippet,
            style="font-size:0.9rem; line-height:1.7; color:var(--text); white-space:pre-wrap;",
        ),
        cls="eval-article-card",
    )


def _swap_content(
    candidate: RelevanceCandidate,
    idx: int,
    total: int,
    labelled_count: int,
    domain: str,
    existing_label: str = "",
    existing_notes: str = "",
) -> Div:
    """Render progress bar + article card together (the full HTMX swap area)."""
    label_badge = ""
    if existing_label:
        color_map = {
            "relevant": "var(--success)",
            "irrelevant": "var(--danger)",
            "noisy": "var(--warning)",
        }
        color = color_map.get(existing_label, "var(--text-muted)")
        label_badge = Span(
            f"Labelled: {existing_label}",
            style=f"display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; background:{color}; color:white; margin-left:12px;",
        )

    dq = f"domain={domain}"

    card = Div(
        # Article counter
        Div(
            Strong(f"Article {idx + 1} of {total}"),
            label_badge,
            style="margin-bottom:16px; font-size:0.9rem;",
        ),
        # Question
        H2(
            "Is this article relevant to the domain?",
            style="font-size:1.2rem; margin-bottom:20px; color:var(--primary-dark);",
        ),
        # Article content
        _article_card(candidate),
        # Notes textarea
        Div(
            Textarea(
                existing_notes,
                name="notes",
                id="eval-notes",
                placeholder="Notes (optional — helpful for noisy/edge cases)",
                rows="2",
                style="width:100%; font-family:var(--font-body); font-size:0.9rem; padding:8px 12px; border:1px solid var(--border); border-radius:var(--radius); resize:vertical;",
            ),
            style="margin:16px 0;",
        ),
        # Label buttons
        Div(
            Button(
                "Relevant ",
                Span("Y", cls="eval-kbd"),
                hx_post=f"/eval/relevance/label?{dq}&idx={idx}&label=relevant",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                hx_include="#eval-notes",
                cls="eval-btn eval-btn-yes",
                id="btn-relevant",
            ),
            Button(
                "Not Relevant ",
                Span("N", cls="eval-kbd"),
                hx_post=f"/eval/relevance/label?{dq}&idx={idx}&label=irrelevant",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                hx_include="#eval-notes",
                cls="eval-btn eval-btn-no",
                id="btn-irrelevant",
            ),
            Button(
                "Noisy ",
                Span("B", cls="eval-kbd"),
                hx_post=f"/eval/relevance/label?{dq}&idx={idx}&label=noisy",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                hx_include="#eval-notes",
                cls="eval-btn eval-btn-unsure",
                id="btn-noisy",
                title="Article is about the topic but contains significant off-topic content (sidebar text, sub-articles, etc.)",
            ),
            Button(
                "Skip ",
                Span("S", cls="eval-kbd"),
                hx_get=f"/eval/relevance/article/{min(idx + 1, total - 1)}?{dq}",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                cls="eval-btn eval-btn-skip",
                id="btn-skip",
            ),
            cls="eval-actions",
        ),
        # Navigation
        Div(
            Button(
                "Prev",
                hx_get=f"/eval/relevance/article/{max(idx - 1, 0)}?{dq}",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                style="font-size:0.85rem;",
            )
            if idx > 0
            else Span(),
            Span(
                f"{idx + 1} / {total}",
                style="color:var(--text-muted); font-size:0.85rem;",
            ),
            Button(
                "Next",
                hx_get=f"/eval/relevance/article/{min(idx + 1, total - 1)}?{dq}",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                style="font-size:0.85rem;",
            )
            if idx < total - 1
            else Span(),
            style="display:flex; justify-content:space-between; align-items:center; margin-top:16px; padding-top:16px; border-top:1px solid var(--border);",
        ),
        cls="eval-pair-card",
    )

    return Div(
        _progress_bar(labelled_count, total),
        card,
    )


# Keyboard shortcut script
_KEYBOARD_JS = """
document.addEventListener('keydown', function(e) {
    // Cmd+Enter or Ctrl+Enter in textarea: submit the noisy label with notes
    if (e.target.tagName === 'TEXTAREA' && e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        e.target.blur();
        document.getElementById('btn-noisy')?.click();
        return;
    }

    // Ignore other keys when typing in textarea or input
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;

    const key = e.key.toLowerCase();
    if (key === 'y') {
        document.getElementById('btn-relevant')?.click();
    } else if (key === 'n') {
        document.getElementById('btn-irrelevant')?.click();
    } else if (key === 'b') {
        document.getElementById('eval-notes')?.focus();
    } else if (key === 's') {
        document.getElementById('btn-skip')?.click();
    } else if (key === 'arrowleft') {
        const btns = document.querySelectorAll('[hx-get*="/eval/relevance/article/"]');
        if (btns.length > 0) btns[0].click();
    } else if (key === 'arrowright') {
        const btns = document.querySelectorAll('[hx-get*="/eval/relevance/article/"]');
        if (btns.length > 0) btns[btns.length - 1].click();
    }
});
"""


def _count_labelled(
    candidates: List[RelevanceCandidate],
    gold: Dict[str, RelevanceLabel],
) -> int:
    """Count how many candidates have gold labels."""
    return sum(1 for c in candidates if c.article_id in gold)


@rt("/eval/relevance")
def get_eval_relevance(request: Request):
    """Main relevance annotation page — full page render."""
    current_domain = get_current_domain(request)
    candidates = load_relevance_candidates(current_domain)
    gold = load_relevance_labels(current_domain)

    if not candidates:
        return titled_with_domain_picker(
            f"Relevance Annotation - {current_domain}",
            current_domain,
            [
                nav_bar(current_domain),
                Div(
                    H2("No relevance candidates found"),
                    P(
                        "Run ",
                        Strong("just seed-relevance"),
                        " to sample articles for annotation.",
                    ),
                    cls="empty-state",
                    style="text-align:left;",
                ),
            ],
        )

    # Find first unlabelled article (or start at 0)
    start_idx = 0
    for i, c in enumerate(candidates):
        if c.article_id not in gold:
            start_idx = i
            break

    labelled_count = _count_labelled(candidates, gold)
    total = len(candidates)

    initial = candidates[start_idx]
    existing = gold.get(initial.article_id)

    return titled_with_domain_picker(
        f"Relevance Annotation - {current_domain}",
        current_domain,
        [
            nav_bar(current_domain),
            # Inline guide
            Div(
                P(
                    Strong("Y"),
                    " = relevant | ",
                    Strong("N"),
                    " = not relevant | ",
                    Strong("B"),
                    " = focus notes (noisy) | ",
                    Strong("Cmd+Enter"),
                    " = submit noisy | ",
                    Strong("S"),
                    " = skip | ",
                    Strong("Arrow keys"),
                    " = navigate",
                    style="margin:0; font-size:0.82rem; color:var(--text-muted);",
                ),
                style="margin-bottom:16px; padding:10px 14px; background:var(--sidebar); border-radius:var(--radius); border:1px solid var(--border);",
            ),
            # Swap area: progress bar + article card (HTMX replaces this)
            Div(
                _swap_content(
                    initial,
                    start_idx,
                    total,
                    labelled_count=labelled_count,
                    domain=current_domain,
                    existing_label=existing.label if existing else "",
                    existing_notes=existing.notes if existing else "",
                ),
                id="eval-rel-swap-area",
            ),
            Script(_KEYBOARD_JS),
        ],
    )


@rt("/eval/relevance/article/{idx}")
def get_eval_relevance_article(idx: int, request: Request):
    """Load a specific article (HTMX partial) — returns progress + card."""
    current_domain = get_current_domain(request)
    candidates = load_relevance_candidates(current_domain)
    gold = load_relevance_labels(current_domain)

    if not candidates:
        return Div(P("No candidates loaded."), cls="empty-state")

    idx = max(0, min(idx, len(candidates) - 1))
    candidate = candidates[idx]
    existing = gold.get(candidate.article_id)

    return _swap_content(
        candidate,
        idx,
        len(candidates),
        labelled_count=_count_labelled(candidates, gold),
        domain=current_domain,
        existing_label=existing.label if existing else "",
        existing_notes=existing.notes if existing else "",
    )


@rt("/eval/relevance/label")
async def post_eval_relevance_label(request: Request):
    """Submit a relevance label and return updated progress + next article."""
    current_domain = get_current_domain(request)
    form = await request.form()
    idx = int(request.query_params.get("idx", "0"))
    label = request.query_params.get("label", "irrelevant")
    notes = form.get("notes", "")

    candidates = load_relevance_candidates(current_domain)
    if not candidates or idx >= len(candidates):
        return Div(P("No more candidates."), cls="empty-state")

    candidate = candidates[idx]

    # Save the label
    rl = make_relevance_label(
        article_id=candidate.article_id,
        label=label,
        notes=str(notes),
    )
    append_relevance_label(current_domain, rl)

    # Advance to next unlabelled article
    gold = load_relevance_labels(current_domain)
    next_idx = idx + 1
    for i in range(next_idx, len(candidates)):
        if candidates[i].article_id not in gold:
            next_idx = i
            break
    else:
        next_idx = min(idx + 1, len(candidates) - 1)

    next_candidate = candidates[next_idx]
    next_existing = gold.get(next_candidate.article_id)

    return _swap_content(
        next_candidate,
        next_idx,
        len(candidates),
        labelled_count=_count_labelled(candidates, gold),
        domain=current_domain,
        existing_label=next_existing.label if next_existing else "",
        existing_notes=next_existing.notes if next_existing else "",
    )
