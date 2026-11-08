"""Move misfiled facts out of a long top-level section before it is split.

Scenario: in the 8 Nov 150-article local run, Qwen filed Vance Spath's 2018
courtroom actions (the trial suspension, the writs for Eliades and Spears)
at the end of "Background". Background passed 6,000 characters, the split
cut it into sub-headings, and Qwen named one "Post-military career": the
trial suspension then sat under Background > Post-military career, where a
reader would believe it. In a 40-article replay the same happened to
Obama's "Role": policy news filed there was split into "Guantanamo closure
efforts / Legal proceedings for detainees", twins of the sub-headings under
"Cases and decisions", and 13 of 19 sampled sentences sat under Role.

So for a reorganised profile, before ``section_split.py`` splits the own
text of a top-level section (the text above its first sub-heading), one call
on the update's model sorts that text the way ``opening_sort.py`` sorts the
opening:

1. Code numbers the section's own cited pieces (``numbered_pieces()``).
2. The model is shown the pieces, the section's heading with what it is for,
   and the profile's other top-level headings (and starter headings it
   lacks), each with its note. It answers with headings and piece numbers.
3. Code moves each piece word for word to the end of its new section's own
   text (right after the last citation there), or under a starter heading it
   creates. A piece left out, put under the section's own heading or under a
   heading the profile doesn't have, stays.
4. Code undoes the move if any sentence or any citation's claim changes.

The split then runs on what is left, only if it is still long.
``profile["section_sorted_json"]`` keeps each sorted heading's own size
after the sort, and it is asked again only once that has grown 25%.
``HINBOX_SECTION_SORT=off`` turns it off.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.engine.opening_sort import _claims, _put, _render, plan_moves
from src.engine.section_split import (
    Ask,
    SectionSplit,
    _clean_heading,
    _headings,
    _sentences,
)
from src.logging_config import get_logger

logger = get_logger("engine.section_sort")

SORT_ENV = "HINBOX_SECTION_SORT"
SORTED_KEY = "section_sorted_json"
RETRY_GROWTH = 1.25

INSTRUCTIONS = """You tidy a profile of the {entity_type} "{entity_name}", \
built from news articles.

The section "{heading}" has grown long. It should only hold {note}. New \
facts may have been added there instead of under the heading they belong \
to. Below is the section's text, cut into numbered pieces (a cited \
sentence, or a few sentences ending at one citation). Put each piece under \
one of these headings:

{listed}

Rules:
- Put every piece number under exactly one heading.
- {heading}: only for pieces that fit what it is for.
- Every other piece goes under the heading for what it mainly tells about \
{entity_name}, using what each heading is for. What happens in or around a \
court case (a ruling, order, motion, hearing, lawyers' argument, testimony, \
evidence request, or what {entity_name} wears or says in the courtroom) goes \
under the legal heading even when it is about detention, CIA custody, \
torture, clothing or health.
- Use the headings above, spelled exactly as given: the heading name only, \
without the description after the colon. Don't make up headings.
- You don't write or change any profile text; code moves the pieces."""


def sort_enabled() -> bool:
    return os.getenv(SORT_ENV, "on").strip().lower() not in ("off", "0", "false")


def _note_for(heading: str, starter: List[str], notes: Dict[str, str]) -> str:
    by_name = {_clean_heading(h).lower(): h for h in starter}
    return notes.get(by_name.get(_clean_heading(heading).lower(), ""), "")


def own_text(text: str, heading: str) -> Optional[Tuple[int, int]]:
    """(start, end) of ``heading``'s own text: from after its heading line to
    its first sub-heading or the next heading. None when it isn't there."""
    from src.engine.profile_edits import _section_end

    end = _section_end(text, heading, own=True)
    if end is None:
        return None
    words = re.escape(_clean_heading(heading))
    head = re.search(rf"^#{{1,6}}[ \t]+{words}[ \t]*$", text[:end], re.MULTILINE)
    return (head.end(), end) if head else None


def top_level(text: str) -> List[str]:
    heads = _headings(text.splitlines())
    top = min((lv for _, lv, _ in heads), default=0)
    return [_clean_heading(t) for _, lv, t in heads if lv == top]


def messages(
    units: List[str],
    spans: List[Tuple[int, int]],
    heading: str,
    note: str,
    targets: List[Tuple[str, str]],
    present: set,
    entity_type: str,
    entity_name: str,
) -> List[Dict[str, str]]:
    listed = [f"- {heading}" + (f": {note}" if note else "")]
    for h, n in targets:
        mark = "" if h.lower() in present else " (not in the profile yet)"
        listed.append(f"- {h}" + (f": {n}" if n else "") + mark)
    system = INSTRUCTIONS.format(
        entity_type=entity_type,
        entity_name=entity_name,
        heading=heading,
        note=note or "what its heading says",
        listed="\n".join(listed),
    )
    numbered = ""
    for n, unit in enumerate(units):
        joined = n and spans[n - 1][1] == spans[n][0]
        numbered += ((" " if joined else "\n\n") if n else "") + f"[{n + 1}] {unit}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f'Text of "{heading}":\n\n{numbered}'},
    ]


def sort_section(
    text: str,
    heading: str,
    entity_type: str,
    entity_name: str,
    starter: List[str],
    notes: Dict[str, str],
    ask: Ask,
) -> Tuple[str, Dict[str, Any]]:
    """The text with ``heading``'s own text sorted, and a report (``status``:
    ``skipped``, ``done``, ``kept``, ``refused`` or ``failed``)."""
    from src.engine.section_reorganise import numbered_pieces

    span = own_text(text, heading)
    report: Dict[str, Any] = {"section": heading}
    if span is None:
        return text, {**report, "status": "skipped", "reason": "no such heading"}
    body = text[span[0] : span[1]].strip()
    units, tail, spans = numbered_pieces(body)
    report.update(chars=len(body), pieces=len(units))
    if len(units) < 2:
        return text, {**report, "status": "skipped", "reason": "one piece"}
    present_top = top_level(text)
    others = [h for h in present_top if h.lower() != heading.lower()]
    low = {h.lower() for h in present_top}
    targets = [(h, _note_for(h, starter, notes)) for h in others]
    targets += [
        (_clean_heading(h), notes.get(h, ""))
        for h in starter
        if _clean_heading(h).lower() not in low
    ]
    present = {_clean_heading(t).lower() for _, _, t in _headings(text.splitlines())}
    msgs = messages(
        units, spans, heading, _note_for(heading, starter, notes), targets,
        present, entity_type, entity_name,
    )  # fmt: skip
    try:
        reply: SectionSplit = ask(msgs)
    except Exception as e:  # keep the section as it is; try again next update
        logger.warning(f"Section sort call failed for {entity_name}: {e}")
        return text, {
            **report,
            "status": "failed",
            "reason": f"{type(e).__name__}: {e}",
        }
    moves = plan_moves(reply, len(units), targets, keep_first=False)
    if not moves:
        return text, {**report, "status": "kept", "own_chars": len(body)}
    keep = [i for i in range(len(units)) if i + 1 not in moves]
    new_body = "\n\n".join(p for p in (_render(units, spans, keep), tail) if p)
    new_text: Optional[str] = (
        text[: span[0]] + "\n\n" + new_body + "\n\n" + text[span[1] :].lstrip()
    )
    groups: Dict[str, List[int]] = {}
    for n in sorted(moves):
        groups.setdefault(moves[n], []).append(n - 1)
    for target, pieces in groups.items():
        new_text = _put(new_text, target, _render(units, spans, pieces), starter)
        if new_text is None:
            return text, {**report, "status": "refused", "reason": f"no {target!r}"}
    new_text = re.sub(r"\n{3,}", "\n\n", new_text).strip() + "\n"
    if _sentences(new_text) != _sentences(text):
        return text, {**report, "status": "refused", "reason": "sentences changed"}
    if _claims(new_text) != _claims(text):
        return text, {**report, "status": "refused", "reason": "claims changed"}
    return new_text, {
        **report,
        "status": "done",
        "moved": {h: len(ps) for h, ps in groups.items()},
        "own_chars": len(new_body),
    }


def sort_long_sections(
    profile: Dict[str, Any],
    entity_type: str,
    entity_name: str,
    kind: Optional[str],
    domain: str,
    ask: Ask,
    min_chars: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``profile`` with each top-level section whose own text passes
    ``min_chars`` sorted first (reorganised profiles only), and the reports."""
    from src.config_loader import get_domain_config
    from src.engine.fact_list import FACT_ROWS_KEY
    from src.engine.section_reorganise import MARK_KEY, starter_headings, starter_notes

    if (
        profile.get(MARK_KEY) != "done"
        or profile.get(FACT_ROWS_KEY)
        or not sort_enabled()
    ):
        return profile, []
    text = str(profile.get("text") or "")
    try:
        sorted_at: Dict[str, int] = json.loads(profile.get(SORTED_KEY) or "{}")
    except (TypeError, ValueError):
        sorted_at = {}
    due = []
    for heading in top_level(text):
        span = own_text(text, heading)
        size = len(text[span[0] : span[1]].strip()) if span else 0
        if size > max(min_chars, sorted_at.get(heading, 0) * RETRY_GROWTH):
            due.append(heading)
    if not due:
        return profile, []
    config = get_domain_config(domain).load_config()
    starter = starter_headings(config, entity_type, kind)
    notes = starter_notes(config, entity_type, kind)
    reports = []
    for heading in due:
        text, report = sort_section(
            text, heading, entity_type, entity_name, starter, notes, ask
        )
        reports.append(report)
        logger.info(
            f"Section sort for {entity_name} '{heading}': {report['status']} "
            f"{report.get('reason') or report.get('moved') or ''}"
        )
        if report["status"] != "failed":
            span = own_text(text, heading)
            sorted_at[heading] = len(text[span[0] : span[1]].strip()) if span else 0
    return {**profile, "text": text, SORTED_KEY: json.dumps(sorted_at)}, reports
