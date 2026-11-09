"""File an update's new sentences under the right sub-heading of their section.

Scenario: in the 9 Nov 150-article local run (Qwen), a hand check of 128
sentences that updates added to 13 sectioned people found 55 under the
wrong heading. 46 of them were under the right top-level heading but the
wrong sub-heading, or sat above the sub-headings when one of them fitted.
Obama's sentences about Trump's executive order sat at the top of "Cases
and decisions", above its sub-heading "Guantanamo detention policy";
Rishikof's firing went under "Role" while "Cases and decisions > Dismissal
circumstances" existed. Tracing each one back to the step that put it
there: 31 came from the edit call's anchor (it anchors on the top-level
heading or a nearby sentence, not on the sub-heading), 11 from
``section_place.place_new()``, which moves a piece to the end of the right
top-level section's own text and never into a sub-heading.

This step runs after ``place_new()``. For each cited piece the update added
that sits in a top-level section with 2 or more sub-headings, it asks the
placement-only sub-heading prompt (``section_place.sub_messages()``, the one
the split's refiling uses) which sub-heading it belongs under, or "None of
these", which means the top of the section, above its sub-headings. Code
moves the piece word for word to the end of that sub-section's own text,
right after its last citation (``opening_sort._put()``), and undoes a move
if any sentence or any citation's claim text would change. A piece that
leans on the sentence before it ("However, ...", "As of ...") stays.

The sub-headings are shown by name alone. On 91 hand-labelled sentences
from that run (``just eval subplacement``, 2 seeds), whose run had 46 under
the right place, filing them this way would leave 63 and 73 right with
Qwen and 70 and 66 with Luna. Showing each sub-heading with the first
sentence under it did worse on both (Qwen 53 and 58, Luna 65 and 63): that
sentence was itself misfiled often enough to mislead, and all 7 of Mary
Spears' subpoena sentences followed one. Luna's misses are mostly sentences
that belong above the sub-headings that it forces into one.

``HINBOX_SUB_PLACE``: ``on``, ``off``, ``local`` (local models only).
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.logging_config import get_logger

logger = get_logger("engine.section_subplace")

SUB_PLACE_ENV = "HINBOX_SUB_PLACE"
SUB_PLACE_DEFAULT = "on"

Ask = Callable[[List[Dict[str, str]]], Any]


def sub_place_enabled(local: bool) -> bool:
    value = os.getenv(SUB_PLACE_ENV, SUB_PLACE_DEFAULT).strip().lower()
    if value in ("off", "0", "false"):
        return False
    return local or value != "local"


def outline(text: str) -> Dict[str, List[str]]:
    """Each top-level heading with its sub-headings one level down. A section
    with a heading deeper than that is left out: its outline isn't the one
    the split makes, so where "the end of a sub-section" is isn't clear."""
    from src.engine.section_split import _clean_heading, _headings

    heads = _headings(text.splitlines())
    if not heads:
        return {}
    top = min(level for _, level, _ in heads)
    out: Dict[str, List[str]] = {}
    deep = set()
    parent = None
    for _, level, title in heads:
        if level == top:
            parent = _clean_heading(title)
            out[parent] = []
        elif parent is not None and level == top + 1:
            out[parent].append(_clean_heading(title))
        elif parent is not None:
            deep.add(parent)
    return {p: subs for p, subs in out.items() if p not in deep}


def heading_at(text: str, pos: int) -> Tuple[Optional[str], Optional[str]]:
    """(top-level heading, sub-heading or None) of the text at ``pos``."""
    from src.engine.section_split import _clean_heading, _headings

    lines = text.splitlines(keepends=True)
    heads = _headings(lines)
    if not heads:
        return None, None
    top_level = min(level for _, level, _ in heads)
    starts, at = [], 0
    for line in lines:
        starts.append(at)
        at += len(line)
    top = sub = None
    for i, level, title in heads:
        if starts[i] > pos:
            break
        if level == top_level:
            top, sub = _clean_heading(title), None
        elif level == top_level + 1:
            sub = _clean_heading(title)
    return top, sub


def ask_subheadings(
    sentences: List[str],
    described: List[Tuple[str, str]],
    parent: str,
    entity_type: str,
    entity_name: str,
    ask: Ask,
) -> List[Optional[str]]:
    """The sub-heading, or ``section_place.STAY``, for each sentence, a batch
    per call; None where the answer was missing or unusable. Raises when a
    call fails."""
    from src.engine.section_place import BATCH, STAY, read_reply, sub_messages

    options = [h for h, _ in described] + [STAY]
    out: List[Optional[str]] = []
    for i in range(0, len(sentences), BATCH):
        chunk = sentences[i : i + BATCH]
        msgs = sub_messages(chunk, described, parent, entity_type, entity_name, True)
        out += read_reply(ask(msgs), len(chunk), options, chunk)
    return out


def _move(text: str, pieces: List[str], target: str) -> Optional[str]:
    """``text`` with ``pieces`` cut and put, in order, at the end of
    ``target``'s own text; None when a piece isn't found exactly once or
    a sentence or citation claim would change."""
    from src.engine.opening_sort import _claims, _put
    from src.engine.section_place import _cut, _drop_empty_headings
    from src.engine.section_split import _sentences

    out = text
    for piece in pieces:
        if out.count(piece) != 1:
            return None
        at = out.find(piece)
        out = _cut(out, at, at + len(piece))
    out = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", out)
    placed = _put(out, target, "\n\n".join(pieces), [])
    if placed is None:
        return None
    out = _drop_empty_headings(placed, text)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    if _sentences(out) != _sentences(text) or _claims(out) != _claims(text):
        return None
    return out


def place_new_subs(
    old: str,
    new: str,
    entity_type: str,
    entity_name: str,
    ask: Ask,
) -> Tuple[str, Dict[str, Any]]:
    """``new`` with the cited pieces an update added moved under the
    sub-heading the model files them under, and a report (``status``:
    ``skipped``, ``kept``, ``done`` or ``failed``; ``moved``: from -> to
    counts; ``refused``: moves undone)."""
    from src.engine.section_place import _LEANS_ON_BEFORE, STAY, new_pieces

    sections = {p: subs for p, subs in outline(new).items() if len(subs) >= 2}
    asked: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for top, s, e in new_pieces(old, new):
        if top not in sections or _LEANS_ON_BEFORE.match(new[s:e]):
            continue
        asked.setdefault(top, []).append((new[s:e], heading_at(new, s)[1]))
    if not asked:
        return new, {"status": "skipped", "reason": "no new pieces under sub-headings"}
    report: Dict[str, Any] = {"pieces": sum(len(v) for v in asked.values())}
    moves: Dict[str, List[Tuple[str, str]]] = {}  # target -> (piece, from)
    try:
        for top, pieces in asked.items():
            texts = [p for p, _ in pieces]
            described = [(sub, "") for sub in sections[top]]
            answers = ask_subheadings(
                texts, described, top, entity_type, entity_name, ask
            )
            for (piece, here), answer in zip(pieces, answers):
                target = top if answer == STAY else answer
                if answer is None or target == (here or top):
                    continue
                moves.setdefault(target, []).append((piece, here or top))
    except Exception as e:  # keep the update as it was placed
        logger.warning(f"Sub-heading placement failed for {entity_name}: {e}")
        return new, {**report, "status": "failed", "reason": f"{type(e).__name__}: {e}"}
    out, moved, refused = new, {}, 0
    for target, group in moves.items():
        placed = _move(out, [p for p, _ in group], target)
        if placed is None:
            refused += len(group)
            continue
        out = placed
        for _, here in group:
            key = f"{here} -> {target}"
            moved[key] = moved.get(key, 0) + 1
    status = "done" if moved else "kept"
    return out, {**report, "status": status, "moved": moved, "refused": refused}


def subplace_profile(
    profile: Dict[str, Any],
    old_text: str,
    entity_type: str,
    entity_name: str,
    model: str,
    local: bool,
    ask: Optional[Ask] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``profile`` with an update's new pieces filed under sub-headings, for
    profiles reorganised before this update; the report, or None when it was
    not tried."""
    from src.engine.fact_list import FACT_ROWS_KEY
    from src.engine.section_place import make_asker
    from src.engine.section_reorganise import MARK_KEY

    if (
        profile.get(MARK_KEY) != "done"
        or profile.get(FACT_ROWS_KEY)
        or not sub_place_enabled(local)
    ):
        return profile, None
    text = str(profile.get("text") or "")
    new_text, report = place_new_subs(
        old_text, text, entity_type, entity_name,
        ask or make_asker(model, local, step="profile.section_subplace"),
    )  # fmt: skip
    logger.info(
        f"Sub-heading placement for {entity_name}: {report['status']} "
        f"{report.get('reason') or report.get('moved') or ''}"
    )
    return ({**profile, "text": new_text} if new_text != text else profile), report
