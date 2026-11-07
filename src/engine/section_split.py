"""Split an over-long profile section into sub-sections, moving text word for word.

Scenario: after 30 articles, Khalid Sheik Mohammed's "### Legal proceedings"
holds 14 paragraphs, about 10,000 characters: charges, years of pretrial
hearings, torture evidence, plea talks, the 2024 plea deal. A reader looking
for the plea deal has to read all of it, and so does the edit model, which
sees the whole section on every update.

After an update, each section whose own text (before any sub-heading) passes
``HINBOX_SECTION_SPLIT_CHARS`` (default 6,000) and has at least 6 paragraphs is
split once:

1. Code numbers the section's paragraphs. A paragraph over 1,500 characters is
   cut into runs of whole sentences first. A numbered piece only ends at a
   citation, so uncited text moves with the cited sentence after it and every
   cited claim keeps its exact text.
2. One model call is shown the numbered paragraphs and answers only with
   sub-headings and paragraph numbers: "Charges and arraignment: 1-4",
   "Pretrial hearings: 5-9", "Plea negotiations: 10-14".
3. Code checks the answer. Every heading must be short plain text and new to
   the profile, or the split is refused. A paragraph number given twice keeps
   its first place; one left out goes with the paragraph before it.
4. Code moves the paragraphs, unchanged, under sub-headings one level down
   (``####`` under ``###``). Sub-sections are in the order of their first
   paragraph, and paragraphs keep their order inside each.
5. Code compares the profile's sentences before and after, sorted, heading
   lines aside. If one sentence differs, the split is undone.

The model never writes a word of profile text here, so nothing can be dropped,
reworded or lose its citation. The worst a bad answer can do is put a
paragraph under the wrong sub-heading. Sentences keep their grounding verdicts
too: a claim is compared without heading lines (``_claim_key()``).

Once a profile has its starter headings (``section_reorganise.py``), a
sub-section the split made can grow long in turn: in a 40-article replay al
Nashiri's "#### Legal Mail and Prison Conditions" reached 8,542 characters.
With ``resplit``, such a sub-section (one level under a top-level heading) is
split the same way, but into headings at its own level that take its place
beside its sibling sub-sections, so the outline never goes deeper than one
level under a starter heading. A parent keeps at most ``MAX_SIBLINGS``
sub-sections.

``HINBOX_SECTION_SPLIT`` sets which tables this runs for: ``people`` by
default (on the dev eval events and organizations scored lower once
reorganised), ``on`` for every table, ``off``, or a comma list.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.engine.profile_view import _SENTENCE_GAP, _blocks
from src.logging_config import get_logger

logger = get_logger("engine.section_split")

SPLIT_ENV = "HINBOX_SECTION_SPLIT"
SPLIT_CHARS = int(os.getenv("HINBOX_SECTION_SPLIT_CHARS", "6000"))
MIN_PARAGRAPHS = 6
PARAGRAPH_CHARS = 1500
MAX_LEVEL = 5  # "#####"; a section at that level is not split again
MIN_GROUPS, MAX_GROUPS = 2, 6
MAX_SIBLINGS = 8  # sub-sections under one heading, after a re-split
RETRY_GROWTH = 1.25  # a refused split is asked again once its section grows 25%
REFUSED_KEY = "split_refused_json"  # heading -> chars, as a JSON string
MAX_HEADING_WORDS, MAX_HEADING_CHARS = 8, 70

_HEADING_LINE = re.compile(r"^(#{1,6})\s+(\S.*)$")
_CITED_END = re.compile(r"\^\[[^\]]*\][.,;:!?\"”’)\s]*$")


class SubSection(BaseModel):
    heading: str = Field(description="Short plain-text sub-heading")
    paragraphs: List[int] = Field(description="Numbers of its paragraphs")


class SectionSplit(BaseModel):
    subsections: List[SubSection]


Ask = Callable[[List[Dict[str, str]]], SectionSplit]

INSTRUCTIONS = """You organise one section of a profile of the {entity_type} \
"{entity_name}", built from news articles.

The section "{heading}" has grown long. Below are its paragraphs, numbered. \
Group them under 2 to 5 {grouping} so that a reader can find a topic quickly.

Rules:
- Put every paragraph number under exactly one sub-heading.
- Keep paragraphs that tell one story together, even when they are far \
apart. Group by topic; a period in brackets is fine, as in \
"Plea negotiations (2017–2024)".
- A sub-heading is 2 to 6 plain words: no "#", no citations, no quotes.
- Don't repeat {repeat_rule}: {other_headings}.
- You don't write or change any profile text; code moves the paragraphs."""


DEFAULT_TABLES = "people"  # on the dev eval, events and organizations scored lower


def tables_enabled(env: str, entity_type: Optional[str] = None) -> bool:
    """Whether the step named by ``env`` runs for ``entity_type``: ``off``,
    ``on`` (every table) or a comma list of tables, default ``people``.
    With no ``entity_type``, whether it runs for any table."""
    from src.engine.hub_profiles import _PLURAL

    value = os.getenv(env, DEFAULT_TABLES).strip().lower()
    if value in ("off", "0", "false", ""):
        return False
    if value in ("on", "1", "true", "all"):
        return True
    if entity_type is None:
        return True
    tables = {t.strip() for t in value.split(",")}
    return _PLURAL.get(entity_type, entity_type) in tables


def split_enabled(entity_type: Optional[str] = None) -> bool:
    return tables_enabled(SPLIT_ENV, entity_type)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _sentences(text: str) -> List[str]:
    """The text's sentences, heading lines and spacing aside, sorted. Each
    paragraph or list item is split on its own, so moving one can't join its
    last sentence to the next one's first. So is each line that ends a
    sentence, as the reorganise numbers them: "al Nashiri was transferred."
    on its own line, moved, is still that sentence."""
    out: List[str] = []
    for para in _paragraphs(text, by_line=True):
        joined = " ".join(para.split())
        if joined and not _HEADING_LINE.match(joined):
            out += [s for s in _SENTENCE_GAP.split(joined) if s]
    return sorted(out)


def _headings(lines: List[str]) -> List[Tuple[int, int, str]]:
    """(line index, level, title) of each heading line."""
    out = []
    for i, line in enumerate(lines):
        m = _HEADING_LINE.match(line.strip())
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def _ends_cited(text: str) -> bool:
    return bool(_CITED_END.search(text))


def _units(own: str, max_chars: int = PARAGRAPH_CHARS) -> Tuple[List[str], str]:
    """The pieces of a section's own text that the model numbers and code moves.

    A piece is a paragraph or list item; one over ``PARAGRAPH_CHARS`` is cut
    into runs of whole sentences. A piece only ends where a citation does, so
    uncited text stays with the cited sentence after it: the grounding judge
    reads everything since the previous citation as one claim, and moving
    "He later moved to Karachi." away from "He studied there.^[a2]" would
    change that claim.

    Returns the numbered pieces and the section's uncited tail: text after its
    last citation, which belongs to the claim of the next citation further
    down the profile, so it is not numbered and stays at the section's end.
    The first piece stays first too (sub-sections go in the order of their
    first piece), so text before the section's first citation keeps its claim.

    ``max_chars`` 0 cuts a paragraph after every cited sentence (the
    reorganise, ``unit_spans()``).
    """
    units, tail, _ = unit_spans(own, max_chars)
    return units, tail


_LINE_END = re.compile(r"(?:[.!?][\"”’)]*|\^\[[^\]]*\][.,;:!?\"”’)]*)$")


def _paragraphs(own: str, by_line: bool) -> List[str]:
    """Blank-line blocks, or with ``by_line`` each line that ends a sentence
    (with the lines before it that don't): model-written profiles often put a
    single newline between paragraphs, so a whole section is one block."""
    out: List[str] = []
    for block in _blocks(own):
        if not by_line:
            out.append("\n".join(block))
            continue
        group: List[str] = []
        for line in block:
            group.append(line)
            if _LINE_END.search(line.strip()):
                out.append("\n".join(group))
                group = []
        if group:
            out.append("\n".join(group))
    return out


def unit_spans(
    own: str, max_chars: int = PARAGRAPH_CHARS, by_line: bool = False
) -> Tuple[List[str], str, List[Tuple[int, int]]]:
    """``_units()`` plus, per piece, the first and last paragraph it comes
    from, so code can put two pieces of one paragraph back together
    (``by_line``: a paragraph is a line, see ``_paragraphs()``)."""
    pieces: List[Tuple[int, str]] = []  # (paragraph number, text)
    for index, text in enumerate(_paragraphs(own, by_line)):
        if len(text) <= max_chars:
            pieces.append((index, text))
            continue
        run: List[str] = []
        for sentence in _SENTENCE_GAP.split(text):
            too_long = len(" ".join(run + [sentence])) > max_chars
            if run and too_long and _ends_cited(run[-1]):
                pieces.append((index, " ".join(run)))
                run = []
            run.append(sentence)
        if run:
            pieces.append((index, " ".join(run)))
    units: List[str] = []
    spans: List[Tuple[int, int]] = []
    open_unit, last = False, -1
    for index, text in pieces:
        if open_unit:
            units[-1] += (" " if index == last else "\n\n") + text
            spans[-1] = (spans[-1][0], index)
        else:
            units.append(text)
            spans.append((index, index))
        open_unit, last = not _ends_cited(text), index
    tail = units.pop() if open_unit and units else ""
    if tail:
        spans.pop()
    return units, tail, spans


def long_sections(
    text: str, min_chars: Optional[int] = None, resplit: bool = False
) -> List[Dict[str, Any]]:
    """Sections whose own text (up to the next heading of any level) is long
    enough to split: ``{"line", "level", "heading", "start", "end", "units",
    "tail", "parent"}``, with ``start``/``end`` the line range the split
    replaces. A top-level section (no higher heading above it) is split one
    level down; ``parent`` is None. With ``resplit``, a sub-section one level
    under a top-level heading is split too, into headings at its own level
    that replace its heading line (``start`` is that line); ``parent`` is the
    top-level heading."""
    limit = SPLIT_CHARS if min_chars is None else min_chars
    lines = text.splitlines()
    heads = _headings(lines)
    top = min((lv for _, lv, _ in heads), default=0)
    found = []
    for k, (i, level, title) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        own = "\n".join(lines[i + 1 : end]).strip()
        if level >= MAX_LEVEL or len(own) <= limit:
            continue
        above = [h for h in heads[:k] if h[1] < level]
        parent = above[-1] if above else None
        rest = heads[k + 1 :]
        stop = next((j for j, h in enumerate(rest) if h[1] <= level), len(rest))
        children = sum(h[1] == level + 1 for h in rest[:stop])
        if parent:
            # A sub-section is never split deeper: in a replay of al Nashiri's
            # articles a sub-section the split had made was split again, down
            # to "#####" leaves of 292 characters. Re-split as siblings only
            # under a top-level heading, and only up to MAX_SIBLINGS of them
            if not resplit or parent[1] != top or level != top + 1:
                continue
            p = heads.index(parent)
            rest = heads[p + 1 :]
            stop = next((j for j, h in enumerate(rest) if h[1] <= top), len(rest))
            siblings = [h for h in rest[:stop] if h[1] == level]
            if len(siblings) + MIN_GROUPS - 1 > MAX_SIBLINGS:
                continue
        units, tail = _units(own)
        if len(units) >= MIN_PARAGRAPHS:
            found.append(
                {"line": i, "level": level, "heading": title,
                 "start": i if parent else i + 1, "end": end, "units": units,
                 "tail": tail, "parent": parent[2] if parent else None,
                 "siblings": len(siblings) if parent else 0,
                 "children": children}
            )  # fmt: skip
    return found


def messages(
    section: Dict[str, Any], entity_type: str, entity_name: str, other: List[str]
) -> List[Dict[str, str]]:
    parent = section.get("parent")
    system = INSTRUCTIONS.format(
        entity_type=entity_type,
        entity_name=entity_name,
        heading=section["heading"],
        grouping=(
            f'headings that replace it, beside the other sub-sections of "{parent}"'
            if parent
            else "sub-headings"
        ),
        repeat_rule=(
            "any other heading the profile has (one of yours may keep the name "
            f'"{section["heading"]}")'
            if parent
            else "the section's own heading, or any heading the profile already has"
        ),
        other_headings="; ".join(f'"{h}"' for h in other) or "none",
    )
    numbered = "\n\n".join(f"[{n}] {u}" for n, u in enumerate(section["units"], 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Paragraphs of \"{section['heading']}\":\n\n{numbered}"},
    ]  # fmt: skip


def _clean_heading(heading: str) -> str:
    return " ".join(heading.strip().lstrip("#").strip(" *_\"'“”:.").split())


def _heading_problem(heading: str, taken: List[str]) -> Optional[str]:
    if not heading:
        return "empty heading"
    if len(heading) > MAX_HEADING_CHARS or len(heading.split()) > MAX_HEADING_WORDS:
        return f"heading too long: {heading!r}"
    if any(c in heading for c in "[]^#\n"):
        return f"heading not plain text: {heading!r}"
    if _norm(heading) in {_norm(t) for t in taken}:
        return f"heading already used: {heading!r}"
    return None


def plan_groups(
    reply: SectionSplit, count: int, taken: List[str], max_groups: int = MAX_GROUPS
) -> Tuple[Optional[List[Tuple[str, List[int]]]], str]:
    """Sub-headings and their paragraph numbers (1-based), in the order they
    go, or None and why the reply was refused. ``taken`` holds the profile's
    headings, which a sub-heading may not repeat."""
    groups: List[Tuple[str, List[int]]] = []
    placed: Dict[int, int] = {}
    seen = list(taken)
    for sub in reply.subsections:
        heading = _clean_heading(sub.heading)
        problem = _heading_problem(heading, seen)
        if problem:
            return None, problem
        seen.append(heading)
        mine = []
        for n in sub.paragraphs:
            if 1 <= n <= count and n not in placed:  # a repeat keeps its first place
                placed[n] = len(groups)
                mine.append(n)
        groups.append((heading, mine))
    if len(placed) * 2 < count:
        return None, f"only {len(placed)} of {count} paragraphs placed"
    for n in range(1, count + 1):  # a missing one goes with its neighbour
        if n not in placed:
            near = next((m for m in range(n - 1, 0, -1) if m in placed), None)
            near = near or next(m for m in range(n + 1, count + 1) if m in placed)
            placed[n] = placed[near]
            groups[placed[n]][1].append(n)
    kept = [(h, sorted(ns)) for h, ns in groups if ns]
    if not MIN_GROUPS <= len(kept) <= max_groups:
        return None, f"{len(kept)} sub-sections"
    return sorted(kept, key=lambda g: g[1][0]), ""


def _render(
    level: int, units: List[str], plan: List[Tuple[str, List[int]]], tail: str = ""
) -> List[str]:
    out: List[str] = [""]
    for heading, numbers in plan:
        out += [f"{'#' * (level + 1)} {heading}", ""]
        for n in numbers:
            out += [units[n - 1], ""]
    return out + ([tail, ""] if tail else [])


def split_long_sections(
    text: str,
    entity_type: str,
    entity_name: str,
    ask: Ask,
    min_chars: Optional[int] = None,
    resplit: bool = False,
    refused: Optional[Dict[str, int]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Split each long section once (``resplit``: sub-sections too, see
    ``long_sections()``). Returns the new text and one report per long
    section (``status``: ``split``, ``refused``, ``failed`` or ``waiting``).

    ``refused``: heading -> size of a section whose split was refused on an
    earlier update; it is asked again only once it has grown by
    ``RETRY_GROWTH``. In a local replay al Nashiri's "November 2011
    arraignment" was asked and refused at 9 updates in a row, the same way."""
    sections = long_sections(text, min_chars, resplit)
    if not sections:
        return text, []
    lines = text.splitlines()
    taken = [title for _, _, title in _headings(lines)]
    reports: List[Dict[str, Any]] = []
    # Sub-sections each accepted split added, by parent heading: two re-splits
    # under one parent, or a parent's own text split beside its sub-sections,
    # share the MAX_SIBLINGS room
    added: Dict[str, int] = {}
    # Bottom up, so earlier line numbers stay right (a parent comes after its
    # sub-sections)
    for section in reversed(sections):
        report: Dict[str, Any] = {
            "section": section["heading"],
            "chars": sum(len(u) for u in section["units"]),
            "paragraphs": len(section["units"]),
        }
        reports.insert(0, report)
        sibling = section.get("parent") is not None
        before = (refused or {}).get(section["heading"])
        if before and report["chars"] < before * RETRY_GROWTH:
            report.update(status="waiting", reason=f"refused at {before} chars")
            continue
        other = [t for t in taken if t != section["heading"]]
        # A re-split replaces the heading, so one new heading may keep its name
        if sibling:
            room = (
                MAX_SIBLINGS - section["siblings"] - added.get(section["parent"], 0) + 1
            )
        else:
            room = MAX_SIBLINGS - section["children"] - added.get(section["heading"], 0)
        if room < MIN_GROUPS:
            report.update(status="refused", reason=f"over {MAX_SIBLINGS} sub-sections")
            continue
        try:
            reply = ask(messages(section, entity_type, entity_name, other))
        except Exception as e:  # keep the section as it is
            logger.warning(f"Section split call failed for {entity_name}: {e}")
            report.update(status="failed", reason=f"{type(e).__name__}: {e}")
            continue
        plan, why = plan_groups(
            reply, len(section["units"]), other if sibling else taken,
            min(MAX_GROUPS, room),
        )  # fmt: skip
        if plan is None:
            report.update(status="refused", reason=why)
            continue
        level = section["level"] - 1 if sibling else section["level"]
        new_lines = (
            lines[: section["start"]]
            + _render(level, section["units"], plan, section["tail"])
            + lines[section["end"] :]
        )
        if _sentences("\n".join(new_lines)) != _sentences("\n".join(lines)):
            report.update(status="refused", reason="sentences changed")
            continue
        lines = new_lines
        owner = section["parent"] if sibling else section["heading"]
        added[owner] = added.get(owner, 0) + len(plan) - (1 if sibling else 0)
        taken = (other if sibling else taken) + [h for h, _ in plan]
        report.update(status="split", subsections=[[h, ns] for h, ns in plan])
    new_text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"
    if not any(r["status"] == "split" for r in reports):
        return text, reports
    if _sentences(new_text) != _sentences(text):  # never expected; undo everything
        return text, [
            {**r, "status": "refused", "reason": "sentences changed"} for r in reports
        ]
    return new_text, reports


def make_asker(model: str, local: bool, step: str = "profile.section_split") -> Ask:
    """The split call on ``model``, through the same clients as profile updates."""
    from src.utils.llm import cloud_generation, local_generation
    from src.utils.step_timing import timed_step

    generate = local_generation if local else cloud_generation

    def ask(msgs: List[Dict[str, str]]) -> SectionSplit:
        with timed_step(step):
            reply = generate(
                messages=msgs, response_model=SectionSplit, model=model, max_tokens=2048
            )
        if reply is None:
            raise RuntimeError("section split returned nothing")
        return reply

    return ask


def split_profile(
    profile: Dict[str, Any],
    entity_type: str,
    entity_name: str,
    model: str,
    local: bool,
    ask: Optional[Ask] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``profile`` with its long sections split, and the reports."""
    from src.engine.fact_list import FACT_ROWS_KEY

    if profile.get(FACT_ROWS_KEY):
        # The fact-list updater saves each row's section name and edits only
        # inside it; moved rows would be looked for in the wrong section
        logger.info(f"No section split for {entity_name}: it has saved fact rows")
        return profile, []
    text = str(profile.get("text") or "")
    # Only a profile with starter headings has a known outline: top-level
    # starter headings, their sub-sections one level down
    resplit = profile.get("reorganised") == "done"
    refused = json.loads(profile.get(REFUSED_KEY) or "{}")
    new_text, reports = split_long_sections(
        text, entity_type, entity_name, ask or make_asker(model, local),
        resplit=resplit, refused=refused,
    )  # fmt: skip
    for r in reports:
        level = "info" if r["status"] == "split" else "warning"
        getattr(logger, level)(
            f"Section split for {entity_name} ({r['section']}): {r['status']} "
            f"{r.get('reason') or [h for h, _ in r.get('subsections', [])]}"
        )
    for r in reports:
        if r["status"] == "refused":
            refused[r["section"]] = r["chars"]
        elif r["status"] == "split":
            refused.pop(r["section"], None)
    if refused != json.loads(profile.get(REFUSED_KEY) or "{}"):
        profile = {**profile, REFUSED_KEY: json.dumps(refused, ensure_ascii=False)}
    if new_text == text:
        return profile, reports
    return {**profile, "text": new_text}, reports
