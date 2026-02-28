"""Annotation route for the merge evaluation harness.

Provides a keyboard-driven interface for labelling merge candidate pairs
as same-entity (yes), different (no), or unsure.
"""

from typing import List

from fasthtml.common import (
    H2,
    H3,
    H4,
    Button,
    Div,
    Em,
    P,
    Script,
    Small,
    Span,
    Strong,
    Textarea,
)
from starlette.requests import Request

from src.eval.candidates import MergeCandidate, candidate_key, load_candidates
from src.eval.gold_labels import (
    append_gold_label,
    load_gold_labels,
    make_gold_label,
)

from ..app_config import get_current_domain, nav_bar, rt, titled_with_domain_picker

# Shared HTMX target — progress bar + pair card live inside this wrapper
_SWAP_TARGET = "#eval-swap-area"


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


def _entity_panel(label: str, name: str, entity_type: str, context: str) -> Div:
    """Render one side of the entity comparison."""
    return Div(
        H4(
            label,
            style="margin:0 0 8px; color:var(--text-muted); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em;",
        ),
        H3(name, style="margin:0 0 8px; color:var(--primary-dark);"),
        Span(entity_type, cls="tag", style="margin-bottom:12px;"),
        P(
            context if context else Em("No profile text available"),
            style="font-size:0.9rem; line-height:1.6; color:var(--text); margin-top:12px;",
        ),
        cls="eval-entity-panel",
    )


def _pipeline_info(candidate: MergeCandidate) -> Div:
    """Show pipeline decision info (muted to reduce anchoring bias)."""
    parts = []
    if candidate.pipeline_decision:
        parts.append(f"Pipeline said: {candidate.pipeline_decision}")
    if candidate.pipeline_confidence is not None:
        parts.append(f"@ {candidate.pipeline_confidence:.0%}")
    if candidate.similarity_score is not None:
        parts.append(f"(similarity: {candidate.similarity_score:.1f})")
    if not parts:
        return Div()
    return Div(
        Small(" ".join(parts)),
        cls="eval-pipeline-info",
    )


def _swap_content(
    candidate: MergeCandidate,
    idx: int,
    total: int,
    labelled_count: int,
    domain: str,
    existing_label: str = "",
    existing_notes: str = "",
) -> Div:
    """Render progress bar + pair card together (the full HTMX swap area)."""
    label_badge = ""
    if existing_label:
        color_map = {
            "yes": "var(--success)",
            "no": "var(--danger)",
            "unsure": "var(--warning)",
        }
        color = color_map.get(existing_label, "var(--text-muted)")
        label_badge = Span(
            f"Labelled: {existing_label}",
            style=f"display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; background:{color}; color:white; margin-left:12px;",
        )

    dq = f"domain={domain}"

    pair_card = Div(
        # Pair counter
        Div(
            Strong(f"Pair {idx + 1} of {total}"),
            label_badge,
            style="margin-bottom:16px; font-size:0.9rem;",
        ),
        # Question
        H2(
            "Do these refer to the same real-world entity?",
            style="font-size:1.2rem; margin-bottom:20px; color:var(--primary-dark);",
        ),
        # Side-by-side panels
        Div(
            _entity_panel(
                "Entity A",
                candidate.entity_name,
                candidate.entity_type,
                candidate.entity_context or "",
            ),
            _entity_panel(
                "Entity B",
                candidate.candidate_name,
                candidate.entity_type,
                candidate.candidate_context or "",
            ),
            cls="eval-pair-panels",
        ),
        # Pipeline info (muted)
        _pipeline_info(candidate),
        # Notes textarea
        Div(
            Textarea(
                existing_notes,
                name="notes",
                id="eval-notes",
                placeholder="Notes (optional, required for Unsure)",
                rows="2",
                style="width:100%; font-family:var(--font-body); font-size:0.9rem; padding:8px 12px; border:1px solid var(--border); border-radius:var(--radius); resize:vertical;",
            ),
            style="margin:16px 0;",
        ),
        # Label buttons
        Div(
            Button(
                "Same entity ",
                Span("Y", cls="eval-kbd"),
                hx_post=f"/eval/label?{dq}&idx={idx}&label=yes",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                hx_include="#eval-notes",
                cls="eval-btn eval-btn-yes",
                id="btn-yes",
            ),
            Button(
                "Different ",
                Span("N", cls="eval-kbd"),
                hx_post=f"/eval/label?{dq}&idx={idx}&label=no",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                hx_include="#eval-notes",
                cls="eval-btn eval-btn-no",
                id="btn-no",
            ),
            Button(
                "Unsure ",
                Span("U", cls="eval-kbd"),
                hx_post=f"/eval/label?{dq}&idx={idx}&label=unsure",
                hx_target=_SWAP_TARGET,
                hx_swap="innerHTML",
                hx_include="#eval-notes",
                cls="eval-btn eval-btn-unsure",
                id="btn-unsure",
            ),
            Button(
                "Skip ",
                Span("S", cls="eval-kbd"),
                hx_get=f"/eval/pair/{min(idx + 1, total - 1)}?{dq}",
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
                hx_get=f"/eval/pair/{max(idx - 1, 0)}?{dq}",
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
                hx_get=f"/eval/pair/{min(idx + 1, total - 1)}?{dq}",
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
        pair_card,
    )


# Keyboard shortcut script
_KEYBOARD_JS = """
document.addEventListener('keydown', function(e) {
    // Cmd+Enter or Ctrl+Enter in textarea: submit the unsure label with notes
    if (e.target.tagName === 'TEXTAREA' && e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        e.target.blur();
        document.getElementById('btn-unsure')?.click();
        return;
    }

    // Ignore other keys when typing in textarea or input
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;

    const key = e.key.toLowerCase();
    if (key === 'y') {
        document.getElementById('btn-yes')?.click();
    } else if (key === 'n') {
        document.getElementById('btn-no')?.click();
    } else if (key === 'u') {
        document.getElementById('eval-notes')?.focus();
    } else if (key === 's') {
        document.getElementById('btn-skip')?.click();
    } else if (key === 'arrowleft') {
        const prev = document.querySelector('[hx-get*="/eval/pair/"]:first-of-type');
        if (prev && !prev.disabled) prev.click();
    } else if (key === 'arrowright') {
        const next = document.querySelector('[hx-get*="/eval/pair/"]:last-of-type');
        if (next && !next.disabled) next.click();
    }
});
"""


def _count_labelled(
    candidates: List[MergeCandidate],
    gold: dict,
) -> int:
    """Count how many candidates have gold labels."""
    return sum(1 for c in candidates if candidate_key(c) in gold)


@rt("/eval")
def get_eval(request: Request):
    """Main annotation page — full page render."""
    current_domain = get_current_domain(request)
    candidates = load_candidates(current_domain)
    gold = load_gold_labels(current_domain)

    if not candidates:
        return titled_with_domain_picker(
            f"Merge Annotation - {current_domain}",
            current_domain,
            [
                nav_bar(current_domain),
                Div(
                    H2("No merge candidates found"),
                    P(
                        "Run ",
                        Strong("just seed-candidates"),
                        " to generate candidates from entity data.",
                    ),
                    cls="empty-state",
                    style="text-align:left;",
                ),
            ],
        )

    # Find first unlabelled pair (or start at 0)
    start_idx = 0
    for i, c in enumerate(candidates):
        key = candidate_key(c)
        if key not in gold:
            start_idx = i
            break

    labelled_count = _count_labelled(candidates, gold)
    total = len(candidates)

    initial_candidate = candidates[start_idx]
    initial_key = candidate_key(initial_candidate)
    existing = gold.get(initial_key)

    return titled_with_domain_picker(
        f"Merge Annotation - {current_domain}",
        current_domain,
        [
            nav_bar(current_domain),
            # Inline guide
            Div(
                P(
                    Strong("Y"),
                    " = same entity | ",
                    Strong("N"),
                    " = different | ",
                    Strong("U"),
                    " = unsure (focus notes) | ",
                    Strong("Cmd+Enter"),
                    " = submit unsure | ",
                    Strong("S"),
                    " = skip | ",
                    Strong("Arrow keys"),
                    " = navigate",
                    style="margin:0; font-size:0.82rem; color:var(--text-muted);",
                ),
                style="margin-bottom:16px; padding:10px 14px; background:var(--sidebar); border-radius:var(--radius); border:1px solid var(--border);",
            ),
            # Swap area: progress bar + pair card (HTMX replaces this)
            Div(
                _swap_content(
                    initial_candidate,
                    start_idx,
                    total,
                    labelled_count=labelled_count,
                    domain=current_domain,
                    existing_label=existing.label if existing else "",
                    existing_notes=existing.notes if existing else "",
                ),
                id="eval-swap-area",
            ),
            Script(_KEYBOARD_JS),
        ],
    )


@rt("/eval/pair/{idx}")
def get_eval_pair(idx: int, request: Request):
    """Load a specific pair (HTMX partial) — returns progress + pair."""
    current_domain = get_current_domain(request)
    candidates = load_candidates(current_domain)
    gold = load_gold_labels(current_domain)

    if not candidates:
        return Div(P("No candidates loaded."), cls="empty-state")

    idx = max(0, min(idx, len(candidates) - 1))
    candidate = candidates[idx]
    key = candidate_key(candidate)
    existing = gold.get(key)

    return _swap_content(
        candidate,
        idx,
        len(candidates),
        labelled_count=_count_labelled(candidates, gold),
        domain=current_domain,
        existing_label=existing.label if existing else "",
        existing_notes=existing.notes if existing else "",
    )


@rt("/eval/label")
async def post_eval_label(request: Request):
    """Submit a label and return updated progress + next pair."""
    current_domain = get_current_domain(request)
    form = await request.form()
    idx = int(request.query_params.get("idx", "0"))
    label = request.query_params.get("label", "no")
    notes = form.get("notes", "")

    candidates = load_candidates(current_domain)
    if not candidates or idx >= len(candidates):
        return Div(P("No more candidates."), cls="empty-state")

    candidate = candidates[idx]

    # Save the label
    gl = make_gold_label(
        entity_name=candidate.entity_name,
        candidate_name=candidate.candidate_name,
        entity_type=candidate.entity_type,
        label=label,
        notes=str(notes),
        pipeline_decision=candidate.pipeline_decision,
        pipeline_confidence=candidate.pipeline_confidence,
        similarity_score=candidate.similarity_score,
    )
    append_gold_label(current_domain, gl)

    # Advance to next unlabelled pair
    gold = load_gold_labels(current_domain)
    next_idx = idx + 1
    for i in range(next_idx, len(candidates)):
        key = candidate_key(candidates[i])
        if key not in gold:
            next_idx = i
            break
    else:
        next_idx = min(idx + 1, len(candidates) - 1)

    next_candidate = candidates[next_idx]
    next_key = candidate_key(next_candidate)
    next_existing = gold.get(next_key)

    return _swap_content(
        next_candidate,
        next_idx,
        len(candidates),
        labelled_count=_count_labelled(candidates, gold),
        domain=current_domain,
        existing_label=next_existing.label if next_existing else "",
        existing_notes=next_existing.notes if next_existing else "",
    )
