"""Update a profile by applying a short list of edits instead of rewriting it.

The pipeline's ``update_profile`` asks the model for the whole new profile.
On long profiles that rewrite quietly drops facts: in the replay, single
updates cut a 31k-character profile to 6k. Here the model only says what to
add or change, and code applies that to the old text, so a sentence the model
never mentions can't disappear.

Two kinds of edit (see ``profile_edit.md``):

- ``add``: new text after an anchor (an existing sentence, a ``## heading``,
  or ``END``).
- ``replace``: a new version of one existing sentence. Any citation the old
  sentence had and the new one lacks is put back, so a replace can't drop a
  source.

There is no delete. An edit whose anchor can't be found is recorded; an
unplaced ``add`` goes to the end, an unplaced ``replace`` is skipped.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from src.constants import (
    CLOUD_MODEL,
    OLLAMA_MODEL,
    PROFILE_MAX_TOKENS,
    PROFILE_VIEW_CHARS,
)
from src.engine.profile_view import profile_view, trimmed_note
from src.utils.llm import cloud_generation, local_generation

END = "END"
_CITATION = re.compile(r"\^\[([^\]]+)\]")


class ProfileEdit(BaseModel):
    op: Literal["add", "replace"]
    anchor: str = Field(
        description="An existing sentence copied exactly, a '## heading', or END"
    )
    text: str = Field(description="The new sentence(s), each ending in a citation")


class Disagreement(BaseModel):
    profile_sentence: str = Field(description="The profile sentence, copied exactly")
    article_says: str = Field(description="What the article says instead")


class ProfileEditList(BaseModel):
    # Filled in before the edits, so the model checks for conflicts first
    # (on the dev set it ignored conflicting years when only asked for edits).
    # Asked too broadly, it called different events or extra detail a conflict
    # and wrote "the sources disagree" into the profile, so the prompt narrows it
    disagreements: List[Disagreement] = Field(
        default_factory=list,
        description="Claims where the article gives a different value for the "
        "very same thing as the profile, so both can't be true. Usually empty",
    )
    new_facts: List[str] = Field(
        default_factory=list,
        description="Facts about the entity in the article that the profile "
        "doesn't have yet, in a few words each",
    )
    edits: List[ProfileEdit] = Field(default_factory=list)

    @field_validator("disagreements", "new_facts", "edits", mode="before")
    @classmethod
    def _parse_json_string(cls, value: Any, info: ValidationInfo) -> Any:
        # Bolt sometimes sends a list as a string, the same way on every
        # retry, so each retry burned another ~25s call and still failed
        if not isinstance(value, str):
            return value
        parsed = _parse_list_string(value)
        if parsed is not None:
            return parsed
        # new_facts only steers the model and is never applied, so an
        # unreadable one shouldn't sink otherwise good edits
        return [value] if info.field_name == "new_facts" else value


# Bolt writes a citation's closing bracket then swaps the string's closing
# quote and the object's brace: `...^[id]}"]` instead of `...^[id]"}]`. After
# `}` closes an object, JSON never has a `"` next, so this only matches the slip
_SWAPPED_CLOSE = re.compile(r'\]\}"(\s*[,\]])')
# Or it drops that closing quote altogether: `...^[id]}]`. Every value in an
# edit is a string, so `]}` straight after a citation can only be this slip
_DROPPED_QUOTE = re.compile(r'(\^\[[^\]"]*)\]\}(\s*[,\]])')


def _repairs(value: str) -> List[str]:
    """The text as sent, then with each known Bolt slip fixed."""
    fixed = [
        value,
        _SWAPPED_CLOSE.sub(r']"}\1', value),
        _DROPPED_QUOTE.sub(r'\1]"}\2', value),
    ]
    # ... or it stops before the array's final `]`: `...^[id]"}`
    stripped = value.rstrip()
    if stripped.startswith("[") and stripped.endswith("}"):
        fixed += [f"{text.rstrip()}]" for text in fixed]
    return fixed


def _parse_list_string(value: str) -> Optional[List[Any]]:
    """Read a list the model sent as a string, or None if it can't be read.

    Handles a plain JSON array, several arrays one after another (Bolt sends
    '["fact 1"]\\n["fact 2"]', one array per item) and the slips in
    ``_repairs()``.
    """
    for text in _repairs(value):
        items: List[Any] = []
        decoder, pos, text = json.JSONDecoder(), 0, text.strip()
        try:
            while pos < len(text):
                part, pos = decoder.raw_decode(text, pos)
                if not isinstance(part, list):
                    break
                items.extend(part)
                while pos < len(text) and text[pos] in " \t\r\n,":
                    pos += 1
            else:
                return items
        except json.JSONDecodeError:
            continue
    return None


def _ids(text: str) -> List[str]:
    return [i.strip() for g in _CITATION.findall(text) for i in g.split(",")]


def _squash(text: str) -> str:
    return " ".join(text.split())


def _find(text: str, anchor: str) -> Optional[Tuple[int, int]]:
    """Locate ``anchor`` in ``text``: exactly, else ignoring whitespace."""
    start = text.find(anchor)
    if start >= 0:
        return start, start + len(anchor)
    # Models often re-flow spaces or newlines inside a copied sentence
    words = [re.escape(w) for w in anchor.split()]
    if not words:
        return None
    match = re.search(r"\s+".join(words), text)
    return (match.start(), match.end()) if match else None


def _known_headings(text: str, addition: str) -> Tuple[str, Optional[str]]:
    """``addition`` without the heading lines it opens with that the profile
    already has, and the last of them. In a local replay an ``add`` on END
    began "## Military commission or legal case" and, appended as it was,
    gave KSM's profile a second legal section at the end."""
    lines = addition.splitlines()
    target = None
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
        line = lines[0].strip()
        if line and _section_end(text, line) is None:
            break  # a heading the profile lacks: kept as the model wrote it
        target = line or target
        lines.pop(0)
    if target:
        # A sub-heading after it goes one level under the profile's own level
        # for that heading, whatever level the model wrote them at
        words = re.escape(target.lstrip("#").strip())
        real = re.search(rf"^(#{{1,6}})[ \t]+{words}[ \t]*$", text, re.MULTILINE)
        shift = len(real.group(1)) - (len(target) - len(target.lstrip("#")))
        lines = [_shift_heading(line, shift) for line in lines]
    return "\n".join(lines).strip(), target


def _shift_heading(line: str, shift: int) -> str:
    match = re.match(r"^(#{1,6})([ \t]+\S.*)$", line.strip())
    if not match or not shift:
        return line
    return "#" * min(6, max(1, len(match.group(1)) + shift)) + match.group(2)


_HEADING_LINE = re.compile(r"^#{1,6}[ \t]+\S")


def _unknown_heading(
    text: str, addition: str, starter: Optional[List[str]], report: Dict[str, Any]
) -> str:
    """``addition`` with heading lines the profile lacks: in a profile with
    starter headings every heading line is dropped and its text kept (the
    outline is the starter list), else they are kept but the highest is put
    no higher than the profile's top level, the rest shifted with it so a
    sub-heading stays under its parent. A local model's "## Legal
    Proceedings" on END, in a "###" profile, became the only top-level
    heading that code saw."""
    lines = addition.splitlines()
    heads = [i for i, line in enumerate(lines) if _HEADING_LINE.match(line.strip())]
    if not heads:
        return addition
    if starter:
        report.setdefault("dropped_headings", []).extend(
            lines[i].strip() for i in heads
        )
        kept = [line for i, line in enumerate(lines) if i not in heads]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    levels = [len(m.group(1)) for m in re.finditer(r"^(#{1,6})[ \t]", text, re.M)]
    top = min(len(lines[i].strip()) - len(lines[i].strip().lstrip("#")) for i in heads)
    shift = max(0, min(levels) - top) if levels else 0
    return "\n".join(_shift_heading(line, shift) for line in lines)


def _anchor_heading(text: str, anchor: str) -> Optional[str]:
    """The heading line ``anchor`` is found on, when it is found on one."""
    span = _find(text, anchor)
    if span is None:
        return None
    start = text.rfind("\n", 0, span[0]) + 1
    end = text.find("\n", span[1])
    line = text[start : end if end >= 0 else len(text)].strip()
    return line if line.startswith("#") else None


def _heading_or_missing(text: str, anchor: str) -> bool:
    """True when ``anchor`` isn't in ``text`` or is found only on a heading
    line: a second add on a bare "Detention at Guantánamo", after the first
    created that heading, would otherwise be put inside the heading line."""
    span = _find(text, anchor)
    if span is None:
        return True
    line_start = text.rfind("\n", 0, span[0]) + 1
    return text[line_start:].startswith("#")


def _section_end(text: str, heading: str, own: bool = False) -> Optional[int]:
    """Index where the section under ``heading`` ends, or None if absent.

    The heading is found by its words, whatever its level: models write
    "## Legal proceedings" for a profile's "### Legal proceedings". The
    section runs to the next heading of the same or a higher level, so its
    sub-sections ("#### Plea negotiations" under "### Legal proceedings")
    belong to it and an add goes after them, not before. With ``own``, it
    ends at its first sub-heading instead: in a reorganised profile an add
    on the parent heading went to the end of its last sub-section, and in a
    replay KSM's "#### Combatant status" collected his courtroom outbursts.
    """
    words = heading.strip().lstrip("#").strip()
    if not words:
        return None
    match = re.search(
        rf"^(#{{1,6}})[ \t]+{re.escape(words)}[ \t]*$", text, re.MULTILINE
    )
    if not match:
        return None
    level = 6 if own else len(match.group(1))
    following = re.search(rf"^#{{1,{level}}} ", text[match.end() :], re.MULTILINE)
    return match.end() + following.start() if following else len(text)


_CITATION_RUN = re.compile(r"[.,;:!?\"”’)]*(?:\^\[[^\]]*\][.,;:!?\"”’)]*)+")


def _past_citations(text: str, end: int) -> int:
    """``end`` moved past the citations (and the punctuation around them)
    that follow it straight away, so they stay with the sentence before."""
    run = _CITATION_RUN.match(text, end)
    return run.end() if run else end


def apply_edits(
    text: str, edits: List[ProfileEdit], starter: Optional[List[str]] = None
) -> Tuple[str, Dict[str, Any]]:
    """Apply ``edits`` to ``text`` in order. Returns (new_text, report).

    ``starter``: the starter headings of a reorganised profile. An ``add``
    anchored on one of them (with or without "#", or copied with its note)
    goes at the end of that section, and creates the heading in its place
    when the profile lacks it (``section_routing.py``)."""
    from src.engine.section_routing import insert_missing_heading, starter_anchor

    report: Dict[str, Any] = {"applied": 0, "appended": 0, "skipped": []}
    # Several adds after one sentence go in list order: each one goes after
    # the text the previous one inserted there, not straight after the anchor
    last_added: Dict[str, str] = {}
    for edit in edits:
        addition = edit.text.strip()
        anchor = edit.anchor.strip()
        if edit.op == "add":
            addition, known = _known_headings(text, addition)
            if known and (anchor in ("", END) or _find(text, anchor) is None):
                anchor = known
            if any(_HEADING_LINE.match(x.strip()) for x in addition.splitlines()):
                # Even a missing starter heading written into the text is
                # dropped: in a local replay "## Release or transfer" opened a
                # fact about KSM's trial venue, the heading was made at the
                # end of the profile, and 56 later trial sentences landed in it
                addition = _unknown_heading(text, addition, starter, report)
        if not addition:
            report["skipped"].append(
                {"op": edit.op, "anchor": edit.anchor, "why": "empty"}
            )
            continue
        if edit.op == "replace":
            span = _find(text, anchor) if anchor and anchor != END else None
            if span is None:
                report["skipped"].append(
                    {"op": "replace", "anchor": anchor, "why": "not found"}
                )
                continue
            old = text[span[0] : span[1]]
            missing = [i for i in _ids(old) if i not in _ids(addition)]
            if missing:
                addition += f"^[{', '.join(missing)}]"
            text = text[: span[0]] + addition + text[span[1] :]
            report["applied"] += 1
            continue

        name = starter_anchor(anchor, starter) if starter and anchor != END else None
        if name and (anchor.startswith("#") or _heading_or_missing(text, anchor)):
            if _section_end(text, name) is None:
                placed = insert_missing_heading(text, name, addition, starter or [])
                if placed is not None:
                    text = placed
                    report["applied"] += 1
                    report.setdefault("new_headings", []).append(name)
                    continue
            anchor = f"# {name}"
        if not anchor.startswith("#") and anchor != END:
            # A heading's words without "#" ("Nashiri Proceedings") are that
            # heading: as a sentence anchor the add was glued onto its line
            anchor = _anchor_heading(text, anchor) or anchor
        if anchor.startswith("#"):
            at = _section_end(text, anchor, own=bool(starter))
            if at is not None:
                head, tail = text[:at].rstrip("\n"), text[at:].lstrip("\n")
                text = f"{head}\n{addition}\n" + (f"\n{tail}" if tail else "")
                report["applied"] += 1
                continue
        elif anchor and anchor != END:
            span = _find(text, last_added.get(anchor, anchor))
            if span is not None:
                # Same paragraph, right after the anchor sentence and its
                # citations: an anchor copied without its "^[id]" put the add
                # between the sentence and its citation, and in a local replay
                # 5 of Obama's sentences lost theirs that way
                end = _past_citations(text, span[1])
                text = f"{text[:end]} {addition}{text[end:]}"
                last_added[anchor] = addition
                report["applied"] += 1
                continue
        # END, or an anchor that couldn't be placed: append rather than lose it
        text = f"{text.rstrip()}\n\n{addition}\n"
        report["applied" if anchor == END else "appended"] += 1
    return text, report


def _messages(
    entity_type: str,
    entity_name: str,
    profile_text: str,
    article_text: str,
    article_id: str,
    domain: str,
    model: Optional[str] = None,
    view_chars: Optional[int] = None,
    routing: str = "",
) -> List[Dict[str, str]]:
    from src.config_loader import DomainConfig

    # Local models get profile_edit.local.md (GEPA-tuned on Qwen3.6), cloud
    # models profile_edit.cloud.md; the local one made cloud models skip facts
    template = DomainConfig(domain).load_profile_prompt("edit", model)
    instructions = (
        template.format(
            entity_type=entity_type, entity_name=entity_name, article_id=article_id
        )
        + routing
    )
    budget = PROFILE_VIEW_CHARS if view_chars is None else view_chars
    shown = profile_view(profile_text, article_text, budget)
    user = (
        f"{trimmed_note(profile_text, shown)}"
        f"Existing profile:\n----------------\n{shown}\n\n"
        f"New article (ID: {article_id}):\n----------------\n{article_text}"
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user},
    ]


def _dispute_asker(generate: Any, model: str, entity_type: str, entity_name: str):
    """The model call ``guard_edits()`` makes to check "sources disagree"
    sentences (``dispute_check.py``), or None when the check is off."""
    from src.engine.dispute_check import (
        DisputeVerdicts,
        dispute_check_enabled,
    )
    from src.utils.step_timing import timed_step

    if not dispute_check_enabled():
        return None

    def ask(instructions: str, items: str) -> DisputeVerdicts:
        system = instructions.format(entity_type=entity_type, entity_name=entity_name)
        with timed_step("profile.dispute_check"):
            response = generate(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": items},
                ],
                response_model=DisputeVerdicts,
                model=model,
                max_tokens=4096,
            )
        if response is None:
            raise RuntimeError("dispute check returned nothing")
        return response

    return ask


def edit_profile(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict[str, Any],
    article_text: str,
    article_id: str,
    model: Optional[str] = None,
    domain: str = "guantanamo",
    local: Optional[bool] = None,
    guard: bool = True,
    article_dates: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask the model for edits and apply them. Raises if the call fails.

    ``local`` defaults to whether ``model`` starts with ``ollama/``. With
    ``guard``, repeated sentences and invented "sources disagree" sentences
    are taken out first, and a "sources disagree" sentence that is really a
    later update is rewritten as one (``edit_guards.py``); that needs
    ``article_dates``, the published date of this article and of the
    profile's sources, by article id. ``kind`` (the entity's ``type``) picks
    the starter headings a reorganised profile's new facts are routed to
    (``section_routing.py``).
    """
    from src.engine.corrections_pass import guard_corrections
    from src.engine.edit_guards import guard_edits
    from src.engine.section_routing import routing_note, routing_sections

    local = (model or "").startswith("ollama/") if local is None else local
    generate = local_generation if local else cloud_generation
    model = model or (OLLAMA_MODEL if local else CLOUD_MODEL)
    sections = routing_sections(existing_profile, entity_type, kind, domain)
    text_now = existing_profile.get("text", "")
    routing = routing_note(*sections, entity_name, text_now) if sections else ""
    response = generate(
        messages=_messages(
            entity_type,
            entity_name,
            existing_profile.get("text", ""),
            article_text,
            article_id,
            domain,
            model,
            routing=routing,
        ),
        response_model=ProfileEditList,
        model=model,
        max_tokens=PROFILE_MAX_TOKENS,
    )
    if response is None:
        raise RuntimeError("edit call returned nothing")
    old_text = existing_profile.get("text", "")
    dates = article_dates or {}
    edits, guarded = (
        guard_edits(
            old_text,
            response.edits,
            dates.get(article_id),
            dates,
            ask=_dispute_asker(generate, model, entity_type, entity_name),
        )
        if guard
        else (response.edits, {})
    )
    # A historian's corrections (only inside correction_scope(), during a run)
    edits, by_corrections = guard_corrections(old_text, edits)
    text, report = apply_edits(old_text, edits, sections[0] if sections else None)
    report["routed"] = bool(sections)
    report["guard"] = guarded
    report["corrections"] = by_corrections
    report["profile_chars_shown"] = len(
        profile_view(old_text, article_text, PROFILE_VIEW_CHARS)
    )
    sources = list(existing_profile.get("sources") or [])
    return {
        **existing_profile,
        "text": text,
        "sources": sources + ([article_id] if article_id not in sources else []),
        "edits": [e.model_dump() for e in response.edits],
        "disagreements": [d.model_dump() for d in response.disagreements],
        "edit_report": report,
    }
