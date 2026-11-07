"""Give a long flat profile the starter headings of its kind, moving text word for word.

Scenario: after 20 articles, Khalid Sheik Mohammed's profile is 8,100
characters under the two or three headings its first draft made up
("### Background", "### Legal Proceedings"), with 80% of the text under the
second. Profiles never gain headings after the first draft, so it stays that
way however long it grows.

Once a profile passes ``HINBOX_SECTION_REORGANISE_CHARS`` (default 8,000)
after an update, it is reorganised once:

1. The domain config lists starter headings per entity type and kind
   (``profiles.starter_headings``): a detainee gets Background, Capture and
   CIA custody, Detention at Guantánamo, ...; other people, organizations,
   events and locations get their own lists, or ``default``.
2. Code numbers the profile's paragraphs with ``section_split._units()``, so
   a piece always ends at a citation and every cited claim keeps its text.
   Text before the first heading stays on top as the opening when it ends at
   a citation and is short (``OPENING_CHARS``). The old heading lines are
   dropped.
3. One model call is shown the numbered paragraphs and the starter headings
   (each with what it is for, from the config) and answers only with
   headings and paragraph numbers. It may add a heading
   for a topic no starter heading fits ("Civil lawsuits"), but code refuses
   the whole answer when a new heading shares a word with another heading
   ("Legal proceedings" beside "Military commission or legal case"), so one
   story can't end up in two sections.
4. Code moves the paragraphs, unchanged: starter headings in list order (an
   empty one is left out), then new ones in the order of their first
   paragraph, paragraphs in their old order inside each.
5. Code compares the sentences before and after, sorted, heading lines aside,
   and undoes the move if one differs.

``profile["reorganised"]`` records the outcome (``done`` or ``refused``), so
it is tried once per entity; a failed call (no answer, prompt too long) is
tried again on the next update. The section split
(``section_split.py``) then runs as before on any section still too long.

``HINBOX_SECTION_REORGANISE`` sets which tables this runs for: ``people`` by
default, ``on`` for every table, ``off``, or a comma list
(``section_split.tables_enabled()``). Over 8 Luna dev runs reorganised
person profiles passed 87/96 against 42/48 flat; events 75/96 against 43/48.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.engine.edit_guards import _FILLER, _stem
from src.engine.section_split import (
    Ask,
    SectionSplit,
    _clean_heading,
    _heading_problem,
    _headings,
    _sentences,
    _units,
    make_asker,
)
from src.logging_config import get_logger

logger = get_logger("engine.section_reorganise")

REORGANISE_ENV = "HINBOX_SECTION_REORGANISE"
REORGANISE_CHARS = int(os.getenv("HINBOX_SECTION_REORGANISE_CHARS", "8000"))
MARK_KEY = "reorganised"
MIN_PARAGRAPHS = 6
OPENING_CHARS = 1500  # a longer opening is numbered with the rest
MAX_NEW_HEADINGS = 2
DEFAULT_LEVEL = 3

# Words two headings may share without being about the same thing
_COMMON = _FILLER | set("and or at guantanamo guantánamo other".split())

INSTRUCTIONS = """You organise a profile of the {entity_type} "{entity_name}", \
built from news articles.

The profile has grown long. Below are its paragraphs, numbered. Put each one \
under one of these headings, so that a reader can find a topic quickly:

{starter}

Rules:
- Put every paragraph number under exactly one heading.
- Choose by what the paragraph mainly tells about {entity_name}, using what \
each heading is for. A court ruling, order or hearing goes under the legal \
heading even when it concerns detention or health; a general heading such as \
Background or Overview is only for who or what {entity_name} is and their \
history, not for paragraphs that fit nowhere else.
- Use the headings above, spelled exactly as given: the heading name only, \
without the description after the colon. Leave out a heading no paragraph \
belongs under.
- Only if a paragraph fits none of them, you may add up to {max_new} new \
headings of 2 to 6 plain words. A new heading must not share a word with \
another heading.
- You don't write or change any profile text; code moves the paragraphs."""


def reorganise_enabled(entity_type: Optional[str] = None) -> bool:
    """On for people by default (``section_split.tables_enabled()``)."""
    from src.engine.section_split import tables_enabled

    return tables_enabled(REORGANISE_ENV, entity_type)


def _starter_entries(
    config: Dict[str, Any], entity_type: str, kind: Optional[str]
) -> List[Tuple[str, str]]:
    from src.engine.hub_profiles import _PLURAL

    table = _PLURAL.get(entity_type, entity_type)
    lists = ((config.get("profiles") or {}).get("starter_headings") or {}).get(table)
    lists = lists or {}
    kind = str(kind or "").rsplit(".", 1)[-1].lower()
    entries = []
    for item in lists.get(kind) or lists.get("default") or []:
        # "- Background: origin and life before capture" is a one-key dict
        pairs = item.items() if isinstance(item, dict) else [(item, "")]
        entries += [(str(h), str(note or "")) for h, note in pairs]
    return entries


def starter_headings(
    config: Dict[str, Any], entity_type: str, kind: Optional[str]
) -> List[str]:
    """The starter headings for an entity type (plural or singular) and kind,
    from ``profiles.starter_headings``; ``default`` when the kind has none."""
    return [h for h, _ in _starter_entries(config, entity_type, kind)]


def starter_notes(
    config: Dict[str, Any], entity_type: str, kind: Optional[str]
) -> Dict[str, str]:
    """What each starter heading is for, as the config describes it."""
    return {h: n for h, n in _starter_entries(config, entity_type, kind) if n}


def _key_words(heading: str) -> set:
    words = re.findall(r"[a-z0-9]+", heading.lower())
    return {_stem(w) for w in words if w not in _COMMON}


def _opening_and_body(text: str) -> Tuple[str, str]:
    """Text before the first heading (when it ends at a citation, so its last
    claim can't change), and the rest without heading lines."""
    lines = text.splitlines()
    heads = _headings(lines)
    first = heads[0][0] if heads else 0
    opening = "\n".join(lines[:first]).strip()
    # Number it with the rest when it ends uncited (its last claim would change)
    # or is long: Trump's 7,588 characters above his first heading were most
    # of his profile
    if opening and (_units(opening)[1] or len(opening) > OPENING_CHARS):
        first, opening = 0, ""
    skip = {i for i, _, _ in heads}
    body = "\n".join(
        line for i, line in enumerate(lines) if i >= first and i not in skip
    )
    return opening, body.strip()


def messages(
    units: List[str],
    starter: List[str],
    entity_type: str,
    entity_name: str,
    notes: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    notes = notes or {}
    listed = [f"- {h}" + (f": {notes[h]}" if notes.get(h) else "") for h in starter]
    system = INSTRUCTIONS.format(
        entity_type=entity_type,
        entity_name=entity_name,
        starter="\n".join(listed),
        max_new=MAX_NEW_HEADINGS,
    )
    numbered = "\n\n".join(f"[{n}] {u}" for n, u in enumerate(units, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Paragraphs:\n\n{numbered}"},
    ]


def _starter_for(heading: str, starter: List[str]) -> Optional[str]:
    """The starter heading a reply means: "military commission or legal case",
    or one copied back with its note ("Overview: what the organization is").
    Longest first, so "Role at Guantánamo" isn't read as "Role"."""
    low = heading.lower()
    for h in sorted(starter, key=len, reverse=True):
        name = _clean_heading(h).lower()
        if low == name or (low.startswith(name) and low[len(name)] in ":(-–—"):
            return h
    return None


def plan_sections(
    reply: SectionSplit, count: int, starter: List[str]
) -> Tuple[Optional[List[Tuple[str, List[int]]]], str]:
    """Headings and their paragraph numbers (1-based) in the order they go,
    or None and why the reply was refused."""
    groups: Dict[str, List[int]] = {}
    new: List[str] = []
    placed: Dict[int, str] = {}
    for sub in reply.subsections:
        cleaned = _clean_heading(sub.heading)
        heading = _starter_for(cleaned, starter) or cleaned
        if heading not in groups:
            if heading not in starter:
                problem = _heading_problem(heading, [])
                if problem:
                    return None, problem
                for other in list(starter) + new:
                    if _key_words(heading) & _key_words(other):
                        return None, f"new heading {heading!r} overlaps {other!r}"
                new.append(heading)
            groups[heading] = []
        for n in sub.paragraphs:
            if 1 <= n <= count and n not in placed:  # a repeat keeps its first place
                placed[n] = heading
                groups[heading].append(n)
    if len(new) > MAX_NEW_HEADINGS:
        return None, f"{len(new)} new headings"
    if len(placed) * 2 < count:
        return None, f"only {len(placed)} of {count} paragraphs placed"
    for n in range(1, count + 1):  # a missing one goes with its neighbour
        if n not in placed:
            near = next((m for m in range(n - 1, 0, -1) if m in placed), None)
            near = near or next(m for m in range(n + 1, count + 1) if m in placed)
            placed[n] = placed[near]
            groups[placed[n]].append(n)
    kept = {h: sorted(ns) for h, ns in groups.items() if ns}
    if len(kept) < 2:
        return None, f"{len(kept)} sections"
    order = [h for h in starter if h in kept]
    order += sorted((h for h in kept if h not in starter), key=lambda h: kept[h][0])
    return [(h, kept[h]) for h in order], ""


def reorganise_text(
    text: str,
    entity_type: str,
    entity_name: str,
    starter: List[str],
    ask: Ask,
    min_chars: Optional[int] = None,
    notes: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """The reorganised text and a report (``status``: ``skipped``, ``done``,
    ``refused`` or ``failed``). The text comes back unchanged unless done."""
    limit = REORGANISE_CHARS if min_chars is None else min_chars
    if len(text) <= limit or not starter:
        return text, {"status": "skipped", "reason": "short or no starter headings"}
    opening, body = _opening_and_body(text)
    units, tail = _units(body)
    report: Dict[str, Any] = {"chars": len(text), "paragraphs": len(units)}
    if len(units) < MIN_PARAGRAPHS:
        return text, {**report, "status": "skipped", "reason": "few paragraphs"}
    try:
        reply = ask(messages(units, starter, entity_type, entity_name, notes))
    except Exception as e:  # keep the profile as it is; try again next update
        logger.warning(f"Reorganise call failed for {entity_name}: {e}")
        return text, {
            **report,
            "status": "failed",
            "reason": f"{type(e).__name__}: {e}",
        }
    plan, why = plan_sections(reply, len(units), starter)
    if plan is None:
        return text, {**report, "status": "refused", "reason": why}
    levels = [level for _, level, _ in _headings(text.splitlines())]
    marks = "#" * (min(levels) if levels else DEFAULT_LEVEL)
    out = [opening] if opening else []
    for heading, numbers in plan:
        out.append(f"{marks} {heading}")
        out += [units[n - 1] for n in numbers]
    new_text = "\n\n".join(out + ([tail] if tail else [])).strip() + "\n"
    if _sentences(new_text) != _sentences(text):
        return text, {**report, "status": "refused", "reason": "sentences changed"}
    return new_text, {
        **report,
        "status": "done",
        "sections": [[h, ns] for h, ns in plan],
    }


def reorganise_profile(
    profile: Dict[str, Any],
    entity_type: str,
    entity_name: str,
    kind: Optional[str],
    domain: str,
    model: str,
    local: bool,
    ask: Optional[Ask] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``profile`` with starter headings, once per entity, and the report
    (None when it was not tried)."""
    from src.config_loader import get_domain_config
    from src.engine.fact_list import FACT_ROWS_KEY

    text = str(profile.get("text") or "")
    if (
        profile.get(MARK_KEY)
        or profile.get(FACT_ROWS_KEY)
        or len(text) <= REORGANISE_CHARS
    ):
        return profile, None
    config = get_domain_config(domain).load_config()
    starter = starter_headings(config, entity_type, kind)
    notes = starter_notes(config, entity_type, kind)
    asker = ask or make_asker(model, local, step="profile.section_reorganise")
    new_text, report = reorganise_text(
        text, entity_type, entity_name, starter, asker, notes=notes
    )
    level = "info" if report["status"] == "done" else "warning"
    getattr(logger, level)(
        f"Reorganise for {entity_name}: {report['status']} "
        f"{report.get('reason') or [h for h, _ in report.get('sections', [])]}"
    )
    if report["status"] in ("done", "refused"):
        return {**profile, "text": new_text, MARK_KEY: report["status"]}, report
    return profile, report
