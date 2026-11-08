"""Move facts out of an over-long opening into the sections they belong in.

Scenario: KSM's profile was reorganised at 8,119 characters with a 562
character opening (the text above the first heading: who he is). Later the
edit model anchored new facts on the opening's sentences, because they name
him most plainly: his birthday from leaked documents, Camp 7 guards, the FBI
saying there were no moles, Judge Pohl adjourning the 9/11 case. After 40
articles in a local replay the opening was 11,336 of 25,355 characters, and
nothing capped it: the section split only looks at text under a heading.

For a profile marked ``reorganised: done`` whose opening passes
``HINBOX_OPENING_SORT_CHARS`` (default 2,000):

1. Code numbers the opening's cited sentences (``numbered_pieces()``, as the
   reorganise does); text after its last citation is not numbered and stays.
2. One call on the update's model is shown the pieces and the profile's
   top-level headings, plus the starter headings it lacks, each with what it
   is for, and "Opening" for the pieces that say who the entity is. It
   answers only with headings and piece numbers.
3. Code moves each piece word for word to the end of its section's own text
   (before its sub-headings), creating a missing starter heading in its
   place (``insert_missing_heading()``). Piece 1 always stays; a piece left
   out, or put under a heading the profile doesn't have, stays too.
4. Code compares the sentences before and after, sorted, and undoes the move
   if one differs.

``profile["opening_sorted_chars"]`` keeps the opening's size after a sort (or
when nothing moved, or the answer was refused), and it is asked again only
once the opening has grown 25% past that. ``HINBOX_OPENING_SORT=off`` turns
it off.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.engine.section_split import (
    Ask,
    SectionSplit,
    _clean_heading,
    _headings,
    _sentences,
    make_asker,
)
from src.logging_config import get_logger

logger = get_logger("engine.opening_sort")

SORT_ENV = "HINBOX_OPENING_SORT"
SORT_CHARS = int(os.getenv("HINBOX_OPENING_SORT_CHARS", "2000"))
SORTED_KEY = "opening_sorted_chars"
RETRY_GROWTH = 1.25
OPENING = "Opening"

INSTRUCTIONS = """You tidy a profile of the {entity_type} "{entity_name}", \
built from news articles.

The profile's opening, the text above its first heading, should say in a \
few sentences who {entity_name} is. It has grown long, because new facts \
were added there instead of under the headings. Below is the opening, cut \
into numbered pieces (a cited sentence, or a few sentences ending at one \
citation). Put each piece under one of these headings:

{listed}

Rules:
- Put every piece number under exactly one heading.
- {opening}: only for pieces that say who {entity_name} is (identity, \
nationality, age, the main role or charge they are known for). A piece \
that gives {entity_name}'s age stays there even when it names the hearing or \
date the age was given at. Piece 1 always stays there.
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


def split_opening(text: str) -> Tuple[str, str]:
    """(opening, the rest from the first heading on)."""
    lines = text.splitlines()
    heads = _headings(lines)
    if not heads:
        return "", text
    first = heads[0][0]
    return "\n".join(lines[:first]).strip(), "\n".join(lines[first:]).strip()


def _targets(
    rest: str, starter: List[str], notes: Dict[str, str]
) -> List[Tuple[str, str]]:
    """(heading, note) the model may choose: the top-level headings in the
    profile in their order, then starter headings it lacks."""
    heads = _headings(rest.splitlines())
    top = min((lv for _, lv, _ in heads), default=0)
    present = [_clean_heading(t) for _, lv, t in heads if lv == top]
    low = {h.lower() for h in present}
    by_name = {_clean_heading(h).lower(): h for h in starter}
    out = [(h, notes.get(by_name.get(h.lower(), ""), "")) for h in present]
    out += [(h, notes.get(h, "")) for h in starter if h.lower() not in low]
    return out


def messages(
    units: List[str],
    targets: List[Tuple[str, str]],
    spans: List[Tuple[int, int]],
    entity_type: str,
    entity_name: str,
    present: set,
) -> List[Dict[str, str]]:
    listed = [f"- {OPENING}: who {entity_name} is, in a few sentences"]
    for h, note in targets:
        mark = "" if h.lower() in present else " (not in the profile yet)"
        listed.append(f"- {h}" + (f": {note}" if note else "") + mark)
    system = INSTRUCTIONS.format(
        entity_type=entity_type,
        entity_name=entity_name,
        listed="\n".join(listed),
        opening=OPENING,
    )
    numbered = ""
    for n, unit in enumerate(units):
        joined = n and spans[n - 1][1] == spans[n][0]
        numbered += ((" " if joined else "\n\n") if n else "") + f"[{n + 1}] {unit}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Opening:\n\n{numbered}"},
    ]


def plan_moves(
    reply: SectionSplit, count: int, targets: List[Tuple[str, str]]
) -> Dict[int, str]:
    """Piece number (1-based) -> heading for each piece that leaves the
    opening. A repeat keeps its first place; piece 1, a piece left out and a
    piece under a heading that isn't offered stay."""
    from src.engine.section_reorganise import _starter_for

    names = [h for h, _ in targets]
    moves: Dict[int, str] = {}
    seen: set = set()
    for sub in reply.subsections:
        heading = _starter_for(_clean_heading(sub.heading), names)
        for n in sub.paragraphs:
            if not 1 <= n <= count or n in seen:
                continue
            seen.add(n)
            if heading and n != 1:
                moves[n] = heading
    return moves


def _render(units: List[str], spans: List[Tuple[int, int]], keep: List[int]) -> str:
    """Pieces ``keep`` (0-based, in order) as paragraphs: pieces of one
    paragraph next to each other are joined again."""
    out: List[str] = []
    for k, i in enumerate(keep):
        if k and keep[k - 1] == i - 1 and spans[i - 1][1] == spans[i][0]:
            out[-1] += " " + units[i]
        else:
            out.append(units[i])
    return "\n\n".join(out)


def _put(rest: str, heading: str, block: str, starter: List[str]) -> Optional[str]:
    """``rest`` with ``block`` at the end of ``heading``'s own text, or under
    the starter heading made for it; None when neither works."""
    from src.engine.profile_edits import _section_end
    from src.engine.section_routing import insert_missing_heading

    end = _section_end(rest, heading, own=True)
    if end is None:
        return insert_missing_heading(rest, heading, block, starter)
    return (rest[:end].rstrip() + "\n\n" + block + "\n\n" + rest[end:].lstrip()).strip()


def sort_text(
    text: str,
    entity_type: str,
    entity_name: str,
    starter: List[str],
    ask: Ask,
    notes: Optional[Dict[str, str]] = None,
    min_chars: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """The text with its opening sorted and a report (``status``:
    ``skipped``, ``done``, ``kept``, ``refused`` or ``failed``)."""
    from src.engine.section_reorganise import numbered_pieces

    opening, rest = split_opening(text)
    limit = SORT_CHARS if min_chars is None else min_chars
    report: Dict[str, Any] = {"chars": len(opening)}
    if len(opening) <= limit or not rest:
        return text, {**report, "status": "skipped", "reason": "short opening"}
    units, tail, spans = numbered_pieces(opening)
    report["pieces"] = len(units)
    if len(units) < 2:
        return text, {**report, "status": "skipped", "reason": "one piece"}
    targets = _targets(rest, starter, notes or {})
    present = {_clean_heading(t).lower() for _, _, t in _headings(rest.splitlines())}
    msgs = messages(units, targets, spans, entity_type, entity_name, present)
    try:
        reply = ask(msgs)
    except Exception as e:  # keep the profile as it is; try again next update
        logger.warning(f"Opening sort call failed for {entity_name}: {e}")
        return text, {
            **report,
            "status": "failed",
            "reason": f"{type(e).__name__}: {e}",
        }
    moves = plan_moves(reply, len(units), targets)
    if not moves:
        return text, {**report, "status": "kept", "opening_chars": len(opening)}
    keep = [i for i in range(len(units)) if i + 1 not in moves]
    new_opening = "\n\n".join(p for p in (_render(units, spans, keep), tail) if p)
    groups: Dict[str, List[int]] = {}
    for n in sorted(moves):
        groups.setdefault(moves[n], []).append(n - 1)
    new_rest: Optional[str] = rest
    for heading, pieces in groups.items():
        new_rest = _put(new_rest, heading, _render(units, spans, pieces), starter)
        if new_rest is None:
            return text, {**report, "status": "refused", "reason": f"no {heading!r}"}
    new_text = re.sub(r"\n{3,}", "\n\n", f"{new_opening}\n\n{new_rest}").strip() + "\n"
    if _sentences(new_text) != _sentences(text):
        return text, {**report, "status": "refused", "reason": "sentences changed"}
    return new_text, {
        **report,
        "status": "done",
        "moved": {h: len(ps) for h, ps in groups.items()},
        "opening_chars": len(new_opening),
    }


def sort_opening(
    profile: Dict[str, Any],
    entity_type: str,
    entity_name: str,
    kind: Optional[str],
    domain: str,
    model: str,
    local: bool,
    ask: Optional[Ask] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``profile`` with an over-long opening sorted into its sections, and the
    report (None when it was not tried)."""
    from src.config_loader import get_domain_config
    from src.engine.fact_list import FACT_ROWS_KEY
    from src.engine.section_reorganise import MARK_KEY, starter_headings, starter_notes

    if (
        profile.get(MARK_KEY) != "done"
        or profile.get(FACT_ROWS_KEY)
        or not sort_enabled()
    ):
        return profile, None
    text = str(profile.get("text") or "")
    opening, _ = split_opening(text)
    # Asked again only once the opening has grown 25% past its last size
    last = int(profile.get(SORTED_KEY) or 0)
    if len(opening) <= max(SORT_CHARS, last * RETRY_GROWTH):
        return profile, None
    config = get_domain_config(domain).load_config()
    starter = starter_headings(config, entity_type, kind)
    notes = starter_notes(config, entity_type, kind)
    asker = ask or make_asker(model, local, step="profile.opening_sort")
    new_text, report = sort_text(text, entity_type, entity_name, starter, asker, notes)
    level = "info" if report["status"] in ("done", "kept") else "warning"
    getattr(logger, level)(
        f"Opening sort for {entity_name}: {report['status']} "
        f"{report.get('reason') or report.get('moved') or ''}"
    )
    if report["status"] == "failed":
        return profile, report
    size = report.get("opening_chars", len(opening))
    out = {**profile, SORTED_KEY: size}
    if report["status"] == "done":
        out["text"] = new_text
    return out, report
