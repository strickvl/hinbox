"""Ask a model which heading each of a few profile sentences belongs under.

Scenario: in the 8 Nov local run Qwen filed Vance Spath's 2018 trial
suspension under "Background", and Obama's closure-policy news under "Role"
(whose note says posts and appointments only). Placement happens in three
places today: the reorganise (whole profile, one call), the edit call's
anchor for each new fact, and the split's sub-headings. Each asks the model
to do something else at the same time (keep paragraphs together, write the
edit, invent sub-headings), and a 3-run hand check found Qwen right on only
33 to 35 of 90 sampled sentences either way.

This prompt asks for placement alone: for each numbered sentence the model
first says in a few words what it mainly tells a reader about the entity,
then names one heading from the list. Code reads the heading back with
``section_reorganise._starter_for()``, so a heading copied with its note
still counts.

On 204 hand-labelled sentences (``just eval placement``, 2 seeds) it filed
93.6% under the right starter heading on local Qwen and 93.6-94.1% on Luna,
against 83.8-85.3% and 85.8-87.3% for the reorganise prompt; most of the
gap was Role and Capture sentences the reorganise put under Background, and
headings it made up ("Statements", "Retirement").

The pipeline uses it twice, for reorganised profiles:

1. The reorganise (``section_reorganise.py``) asks it where each numbered
   piece goes, 8 pieces per call (``HINBOX_REORGANISE_ASSIGN=sections``
   brings back the one-call prompt that may add headings).
2. After each later update, ``place_new()`` finds the cited pieces the
   update added (text not in the old profile), asks where each goes, and
   moves a piece filed under another top-level heading, word for word, to
   the end of its section's own text (or under a starter heading it
   creates). Code undoes the move if any sentence or citation claim
   changes. ``HINBOX_SECTION_PLACE=off`` turns this off.

   Two kinds of piece stay where the edit put them. One answered with the
   first starter heading (Background, who the entity is): in 3 local
   replays of Spath, Trump and Obama, Qwen moved 41 new pieces into
   Background, mostly Guantánamo news that says little about the person
   ("the special envoys had left their jobs", detainee counts), which the
   hand labels left out as unsure. And one that leans on the sentence
   before it ("However, ...", "As of March 2017, ...", "That ..."):
   moved, it lost what it answers.
"""

import os
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.logging_config import get_logger

logger = get_logger("engine.section_place")

PLACE_ENV = "HINBOX_SECTION_PLACE"
# A new piece that opens like this answers the sentence before it ("However,
# later reports...", "As of March 2017, 41 captives..."): in a local replay
# such pieces were moved away from what they qualify
_LEANS_ON_BEFORE = re.compile(
    r"^\W*(?:however|but|yet|still|nevertheless|meanwhile|instead|also|later|"
    r"then|by then|as of|another|that|this|these|those|such|it)\b",
    re.IGNORECASE,
)
BATCH = 8  # pieces per call, as scored by the placement eval

INSTRUCTIONS = """You file sentences from a profile of the {entity_type} \
"{entity_name}", built from news articles, under the profile's headings.

Headings, each with what belongs under it:

{listed}

For each numbered sentence:
1. Copy its first three words.
2. Say in a few words what it mainly tells a reader about {entity_name}.
3. Pick the one heading whose description fits that best.

Rules:
- Go by each heading's description, not by which words appear in it.
- What happens in or around a court case (a ruling, order, motion, hearing, \
lawyers' argument, testimony or evidence request) goes under the legal \
heading even when it is about detention, CIA custody, torture or health.
- A general heading such as Background is only for who {entity_name} is and \
their life or career before the events covered, never for news that fits \
nowhere else.
- A heading about posts or roles is only for which post {entity_name} held \
and when, not for what they did in it.
- Answer with the heading name only, spelled exactly as given, without the \
description after the colon."""


def _starts_required(schema: Dict[str, Any]) -> None:
    schema.setdefault("required", []).append("starts_with")


class Placement(BaseModel):
    model_config = ConfigDict(json_schema_extra=_starts_required)

    sentence: int = Field(description="the sentence number")
    starts_with: str = Field("", description="the sentence's first three words")
    about: str = Field(description="what it mainly tells about the entity")
    heading: str = Field(description="one heading name from the list")


class Placements(BaseModel):
    placements: List[Placement]


AskPlacements = Callable[[List[Dict[str, str]]], Placements]


def messages(
    sentences: List[str],
    starter: List[str],
    notes: Dict[str, str],
    entity_type: str,
    entity_name: str,
) -> List[Dict[str, str]]:
    listed = "\n".join(
        f"- {h}" + (f": {notes[h]}" if notes.get(h) else "") for h in starter
    )
    system = INSTRUCTIONS.format(
        entity_type=entity_type, entity_name=entity_name, listed=listed
    )
    numbered = "\n\n".join(f"[{n}] {s}" for n, s in enumerate(sentences, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Sentences:\n\n{numbered}"},
    ]


def _same_start(copied: str, sentence: str) -> bool:
    from src.engine.section_split import _norm

    words = _norm(copied).split()[:3]
    return not words or words == _norm(sentence).split()[: len(words)]


def read_reply(
    reply: Placements,
    count: int,
    starter: List[str],
    sentences: Optional[List[str]] = None,
) -> List[Optional[str]]:
    """The starter heading each sentence got (None: missing, or a heading not
    in the list); a repeated number keeps its first answer. With
    ``sentences``, an answer whose copied first words aren't its sentence's
    is dropped: in a local replay the "about" text of some answers described
    another sentence of the batch, so the numbering had slipped."""
    from src.engine.section_reorganise import _starter_for
    from src.engine.section_split import _clean_heading

    out: List[Optional[str]] = [None] * count
    seen = set()
    for p in reply.placements:
        if 1 <= p.sentence <= count and p.sentence not in seen:
            seen.add(p.sentence)
            if sentences and not _same_start(p.starts_with, sentences[p.sentence - 1]):
                continue
            out[p.sentence - 1] = _starter_for(_clean_heading(p.heading), starter)
    return out


def make_asker(model: str, local: bool, step: str = "profile.section_place"):
    from src.utils.llm import cloud_generation, local_generation
    from src.utils.step_timing import timed_step

    generate = local_generation if local else cloud_generation

    def ask(msgs: List[Dict[str, str]]) -> Placements:
        with timed_step(step):
            reply = generate(
                messages=msgs, response_model=Placements, model=model, max_tokens=4096
            )
        if reply is None:
            raise RuntimeError("section placement returned nothing")
        return reply

    return ask


def place_enabled() -> bool:
    return os.getenv(PLACE_ENV, "on").strip().lower() not in ("off", "0", "false")


def place_pieces(
    pieces: List[str],
    starter: List[str],
    notes: Dict[str, str],
    entity_type: str,
    entity_name: str,
    ask: AskPlacements,
    batch: int = BATCH,
) -> List[Optional[str]]:
    """The starter heading for each piece, ``batch`` pieces per call; a
    failed call raises."""
    out: List[Optional[str]] = []
    for i in range(0, len(pieces), batch):
        chunk = pieces[i : i + batch]
        reply = ask(messages(chunk, starter, notes, entity_type, entity_name))
        out += read_reply(reply, len(chunk), starter, chunk)
    return out


def as_split(answers: List[Optional[str]]):
    """The answers as the reorganise's reply: heading -> piece numbers."""
    from src.engine.section_split import SectionSplit, SubSection

    groups: Dict[str, List[int]] = {}
    for n, heading in enumerate(answers, 1):
        if heading:
            groups.setdefault(heading, []).append(n)
    return SectionSplit(
        subsections=[SubSection(heading=h, paragraphs=ns) for h, ns in groups.items()]
    )


def _blocks(text: str) -> Iterator[Tuple[Optional[str], int, int]]:
    """(top-level heading it sits under or None, start, end) of the text
    between each two heading lines."""
    from src.engine.section_split import _HEADING_LINE, _clean_heading

    lines = text.splitlines(keepends=True)
    heads = []
    at = 0
    for line in lines:
        m = _HEADING_LINE.match(line.strip())
        if m:
            heads.append((at, at + len(line), len(m.group(1)), m.group(2).strip()))
        at += len(line)
    top_level = min((lv for *_, lv, _ in heads), default=0)
    top, start = None, 0
    for line_start, line_end, level, title in heads:
        yield top, start, line_start
        if level == top_level:
            top = _clean_heading(title)
        start = line_end
    yield top, start, len(text)


def new_pieces(old: str, new: str) -> List[Tuple[Optional[str], int, int]]:
    """(top-level heading, start, end) of each cited piece of ``new`` whose
    text isn't in ``old``."""
    from src.engine.section_split import _norm, unit_spans

    old_norm = _norm(old)
    found = []
    for top, start, end in _blocks(new):
        units, _, _ = unit_spans(new[start:end], 0, by_line=True)
        pos = start
        for unit in units:
            at = new.find(unit, pos, end)
            if at < 0:  # rejoined with other spacing: leave it where it is
                continue
            pos = at + len(unit)
            if _norm(unit) and _norm(unit) not in old_norm:
                found.append((top, at, pos))
    return found


def _cut(text: str, start: int, end: int) -> str:
    if text[start - 1 : start] == " " and text[end : end + 1] in (" ", "\n", ""):
        start -= 1  # the space before a piece cut from inside a paragraph
    return text[:start] + text[end:]


def _drop_empty_headings(text: str, before: str) -> str:
    """``text`` without heading lines left with nothing under them (no text
    and no sub-heading), unless that was already so in ``before``."""
    from src.engine.section_split import _headings

    def empty(lines: List[str]) -> set:
        heads = _headings(lines)
        out = set()
        for k, (i, level, title) in enumerate(heads):
            j, next_level = (
                (heads[k + 1][0], heads[k + 1][1])
                if k + 1 < len(heads)
                else (len(lines), 0)
            )
            if next_level <= level and not "".join(lines[i + 1 : j]).strip():
                out.add((title, i))
        return out

    lines = text.splitlines()
    was = {t for t, _ in empty(before.splitlines())}
    drop = {i for t, i in empty(lines) if t not in was}
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def place_new(
    old: str,
    new: str,
    entity_type: str,
    entity_name: str,
    starter: List[str],
    notes: Dict[str, str],
    ask: AskPlacements,
) -> Tuple[str, Dict[str, Any]]:
    """``new`` with the pieces an update added moved under the starter
    heading the model files them under, and a report (``status``:
    ``skipped``, ``kept``, ``done``, ``refused`` or ``failed``)."""
    from src.engine.opening_sort import _claims, _put
    from src.engine.section_reorganise import _starter_for
    from src.engine.section_split import _sentences

    found = new_pieces(old, new)
    if not found or not starter:
        return new, {"status": "skipped", "reason": "no new pieces"}
    report: Dict[str, Any] = {"pieces": len(found)}
    try:
        answers = place_pieces(
            [new[s:e] for _, s, e in found], starter, notes, entity_type,
            entity_name, ask,
        )  # fmt: skip
    except Exception as e:  # keep the update as the edit placed it
        logger.warning(f"Section placement failed for {entity_name}: {e}")
        return new, {**report, "status": "failed", "reason": f"{type(e).__name__}: {e}"}
    moves = []
    for (top, s, e), heading in zip(found, answers):
        here = _starter_for(top, starter) if top else None
        stays = heading is None or heading == here or heading == starter[0]
        if stays or _LEANS_ON_BEFORE.match(new[s:e]):
            continue
        moves.append((s, e, heading, top or "(opening)"))
    if not moves:
        return new, {**report, "status": "kept"}
    out = new
    for s, e, _, _ in sorted(moves, reverse=True):
        out = _cut(out, s, e)
    out = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", out)
    groups: Dict[str, List[Tuple[int, int]]] = {}
    for s, e, heading, _ in moves:
        groups.setdefault(heading, []).append((s, e))
    for heading, spans in groups.items():
        block = new[spans[0][0] : spans[0][1]]
        for (_, prev_end), (s, e) in zip(spans, spans[1:]):
            gap = (
                " "
                if not new[prev_end:s].strip() and "\n" not in new[prev_end:s]
                else "\n\n"
            )
            block += gap + new[s:e]
        placed = _put(out, heading, block, starter)
        if placed is None:
            return new, {**report, "status": "refused", "reason": f"no {heading!r}"}
        out = placed
    out = _drop_empty_headings(out, new)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    if _sentences(out) != _sentences(new):
        return new, {**report, "status": "refused", "reason": "sentences changed"}
    if _claims(out) != _claims(new):
        return new, {**report, "status": "refused", "reason": "claims changed"}
    moved: Dict[str, int] = {}
    for _, _, heading, here in moves:
        moved[f"{here} -> {heading}"] = moved.get(f"{here} -> {heading}", 0) + 1
    return out, {**report, "status": "done", "moved": moved}


def place_profile(
    profile: Dict[str, Any],
    old_text: str,
    entity_type: str,
    entity_name: str,
    kind: Optional[str],
    domain: str,
    model: str,
    local: bool,
    ask: Optional[AskPlacements] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``profile`` with an update's new pieces filed under their starter
    headings, for profiles reorganised before this update; the report, or
    None when it was not tried."""
    from src.config_loader import get_domain_config
    from src.engine.fact_list import FACT_ROWS_KEY
    from src.engine.section_reorganise import MARK_KEY, starter_headings, starter_notes

    if (
        profile.get(MARK_KEY) != "done"
        or profile.get(FACT_ROWS_KEY)
        or not place_enabled()
    ):
        return profile, None
    config = get_domain_config(domain).load_config()
    starter = starter_headings(config, entity_type, kind)
    notes = starter_notes(config, entity_type, kind)
    text = str(profile.get("text") or "")
    new_text, report = place_new(
        old_text, text, entity_type, entity_name, starter, notes,
        ask or make_asker(model, local),
    )  # fmt: skip
    logger.info(
        f"Section placement for {entity_name}: {report['status']} "
        f"{report.get('reason') or report.get('moved') or ''}"
    )
    return ({**profile, "text": new_text} if new_text != text else profile), report


SUB_INSTRUCTIONS = """You file sentences from the section "{parent}" of a \
profile of the {entity_type} "{entity_name}", built from news articles, \
under the section's sub-headings.

Sub-headings, each with what belongs under it:

{listed}

For each numbered sentence:
1. Copy its first three words.
2. Say in a few words what it is mainly about.
3. Pick the one sub-heading whose description fits that best.{stay_rule}

Rules:
- Go by each sub-heading's description, not by which words appear in it.
- Answer with the sub-heading name only, spelled exactly as given, without \
the description after the colon."""

STAY_RULE = """ If none of them fits it well, answer "{stay}": it then \
stays at the top of "{parent}", above its sub-headings."""
STAY = "None of these"
REFILE_ENV = "HINBOX_SPLIT_REFILE"


def refile_enabled(local: bool = True) -> bool:
    """``HINBOX_SPLIT_REFILE``: ``local`` (default) refiles only with a local
    model, ``on`` always, ``off`` never. In 40-article replays of Spath,
    Trump and Obama it raised Qwen's hand-checked right placements from 44
    to 59 of 90 (3 runs each, alternated), but Luna's went from 87 to 78 of
    120 (4 runs each), all of it sub-heading misses in two runs."""
    value = os.getenv(REFILE_ENV, "local").strip().lower()
    if value in ("off", "0", "false"):
        return False
    return local or value in ("on", "1", "true")


def sub_messages(
    sentences: List[str],
    headings: List[Tuple[str, str]],
    parent: str,
    entity_type: str,
    entity_name: str,
    stay: bool,
) -> List[Dict[str, str]]:
    listed = [f"- {h}" + (f": {c}" if c else "") for h, c in headings]
    system = SUB_INSTRUCTIONS.format(
        parent=parent, entity_type=entity_type, entity_name=entity_name,
        listed="\n".join(listed),
        stay_rule=STAY_RULE.format(stay=STAY, parent=parent) if stay else "",
    )  # fmt: skip
    numbered = "\n\n".join(f"[{n}] {t}" for n, t in enumerate(sentences, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Sentences:\n\n{numbered}"},
    ]


def refile_split(
    plan: List[Tuple[str, List[int]]],
    covers: Dict[str, str],
    units: List[str],
    parent: str,
    entity_type: str,
    entity_name: str,
    ask: AskPlacements,
    stay: bool,
    batch: int = BATCH,
) -> Tuple[List[Tuple[str, List[int]]], List[int], Dict[str, Any]]:
    """The split's sub-headings kept, its pieces filed again by the
    placement-only prompt, ``batch`` per call: (plan, pieces that stay above
    the sub-headings, report). A piece with no usable answer keeps the
    split's choice; a refiling that leaves fewer than 2 sub-headings, or a
    failed call, keeps the split's plan as it was."""
    from src.engine.section_split import MIN_GROUPS

    names = [h for h, _ in plan]
    first = {n: h for h, ns in plan for n in ns}
    # Only the plan's pieces: a re-split's pieces that join a sibling
    # (``section_split.plan_groups()``) are not in it
    numbers = sorted(first)
    answers: Dict[int, Optional[str]] = {}
    options = names + ([STAY] if stay else [])
    try:
        for i in range(0, len(numbers), batch):
            chunk = numbers[i : i + batch]
            texts = [units[n - 1] for n in chunk]
            msgs = sub_messages(
                texts, [(h, covers.get(h, "")) for h in names], parent,
                entity_type, entity_name, stay,
            )  # fmt: skip
            got = read_reply(ask(msgs), len(chunk), options, texts)
            answers.update(zip(chunk, got))
    except Exception as e:
        logger.warning(f"Split refiling failed for {entity_name}: {e}")
        return plan, [], {"status": "failed", "reason": f"{type(e).__name__}: {e}"}
    groups: Dict[str, List[int]] = {h: [] for h in names}
    stayed: List[int] = []
    for n in numbers:
        if answers.get(n) == STAY:
            stayed.append(n)
        else:
            groups[answers.get(n) or first[n]].append(n)
    kept = [(h, ns) for h, ns in groups.items() if ns]
    if len(kept) < MIN_GROUPS:
        return plan, [], {"status": "kept", "reason": f"{len(kept)} sub-sections"}
    changed = sum(1 for n, a in answers.items() if a and a != first[n])
    report = {"status": "done", "refiled": changed, "stayed": len(stayed)}
    return sorted(kept, key=lambda g: g[1][0]), stayed, report
