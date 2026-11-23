"""Two sentences of a finished page that tell one fact, each adding a detail,
told once: the earlier sentence is rewritten to hold both, and the later one
is dropped.

Why: in round 39 (23 Nov) the page review judged about 34 pairs "same" on the
14 tuning pages and changed one sentence. A "same" second sentence is dropped
only when it adds nothing, and is otherwise trimmed to what it adds; on real
pages the second telling nearly always carries one more name or date, so
every trim lost something and was refused, and Rishikof's firing stayed told
three times. By the stages rule this is one more small step after the
review: for those pairs, one call per batch writes the fact once from both
sentences and their notes, and code keeps it only when it has every name,
number and date of both and is no longer than both together; then the usual
checks of a sentence against its notes run on it.

Round 38 joined notes the same way and 3 of 18 joins were false, all where a
speaker or a group moved ("the CIA director revealed" took on details no
speaker gave; "including Hambali" moved between groups). So a pair is not
folded when either sentence says who said or reported something, or names
members of a group ("including", "such as"), or when either is itself a
fold.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.engine.written_page import Ask, Note, Written

FOLD = """You edit an encyclopedia entry about "{name}". Each numbered pair \
below is two sentences of the entry that tell one and the same fact, each \
adding a detail, with the notes they were written from (sentences from news \
articles, each with its article's date).

For each pair write one sentence that tells the fact once, with every name, \
number and date of both sentences, in plain encyclopedia style. Keep who \
did what exactly as the sentences say. Add nothing that neither sentence \
says. If the two do not tell the same fact, or one sentence cannot hold \
both, give an empty text. Give only the sentence itself, with no note \
numbers in it."""

BATCH = 6
MAX_WORDS = 60
FOLDED = "told in an earlier sentence, folded into it"

# A sentence that says who said or reported a thing, or lists members of a
# group: folding it moved a speaker or a member in round 38
_RISKY = re.compile(
    r"\b(?:said|says|reported|revealed|described|told|wrote|testified|"
    r"announced|stated|according|acknowledged|admitted|denied|claimed|"
    r"including|such as)\b",
    re.I,
)


class Fold(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pair: int = Field(description="The pair's number")
    text: str = Field(description="The one sentence, or '' when it can't be done")


class Folds(BaseModel):
    model_config = ConfigDict(extra="ignore")
    folds: List[Fold]


def foldable(a: Written, b: Written, folds: Sequence[str] = ()) -> bool:
    """Neither sentence names a speaker or a group's members, neither is
    itself a fold (one of ``folds``), and both are kept."""
    return not (
        _RISKY.search(a.text) or _RISKY.search(b.text)
        or a.dropped or b.dropped or a.text in folds or b.text in folds
        or len(a.text.split()) + len(b.text.split()) > 2 * MAX_WORDS
    )  # fmt: skip


def _marks(text: str) -> set:
    """Numbers, month and capitalised words, any position."""
    from src.engine.written_checks import _numbers

    return _numbers(text) | {w.lower() for w in re.findall(r"\b[A-Z][\w'’-]+", text)}


def refusal(text: str, a: str, b: str) -> str:
    """Why ``text`` can't stand for both ``a`` and ``b``, or "" when it can:
    it must have every number and capitalised word of both past their first
    word (a sentence's first word is capitalised anyway), at most
    ``MAX_WORDS`` words, and be no longer than both together."""
    from src.engine.written_review import _marks as marks_past_first

    if not text:
        return "given_up"
    if len(text) > len(a) + len(b) or len(text.split()) > MAX_WORDS:
        return "too_long"
    if (marks_past_first(a) | marks_past_first(b)) - _marks(text):
        return "lost_mark"
    return ""


def keeps_both(text: str, a: str, b: str) -> bool:
    return not refusal(text, a, b)


def _messages(
    pairs: Sequence[Tuple[Written, Written]], notes: Sequence[Note], name: str
) -> List[Dict[str, str]]:
    from src.engine.written_page import CHECK_NOTE_CHARS

    by_n = {n.n: n for n in notes}
    parts = []
    for k, (a, b) in enumerate(pairs, 1):
        lines = "\n".join(
            f"  - [{by_n[n].date or 'no date'}] {by_n[n].text[:CHECK_NOTE_CHARS]}"
            for n in dict.fromkeys(a.notes + b.notes)
            if n in by_n
        )
        parts.append(f"Pair {k}:\nA: {a.text}\nB: {b.text}\nTheir notes:\n{lines}")
    return [
        {"role": "system", "content": FOLD.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def fold_pairs(
    written: List[Written], pairs: Sequence[Tuple[Written, Written]],
    notes: Sequence[Note], name: str, ask: Ask, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with each foldable pair (earlier, later) told once: the
    earlier sentence dropped for its fold, placed right after it, and the
    later dropped. A fold the code or the model check refuses leaves both.
    Each fold under ``report["folded"]`` as [fold, earlier, later]."""
    from src.engine.written_checks import code_check, model_check
    from src.engine.written_page import strip_refs

    log: List[List[str]] = report.setdefault("folded", [])
    folds = {f[0] for f in log}
    todo: List[Tuple[Written, Written]] = []
    used: set = set()
    for a, b in pairs:
        if id(a) not in used and id(b) not in used and foldable(a, b, folds):
            todo.append((a, b))
            used |= {id(a), id(b)}
    report["fold_asked"] = report.get("fold_asked", 0) + len(todo)
    new: Dict[int, Tuple[Written, Written]] = {}  # id(a) -> (fold, b)
    refused: Dict[str, int] = report.setdefault("fold_refused", {})
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            reply = ask(_messages(batch, notes, name), Folds, 4096)
        except Exception as exc:
            report.setdefault("errors", []).append(f"fold: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for f in reply.folds or []:
            if not 1 <= f.pair <= len(batch):
                continue
            a, b = batch[f.pair - 1]
            text = strip_refs(" ".join(str(f.text or "").split()))[0]
            why = "twice" if id(a) in new else refusal(text, a.text, b.text)
            if why:
                refused[why] = refused.get(why, 0) + 1
                continue
            new[id(a)] = (
                Written(
                    text, list(dict.fromkeys(a.notes + b.notes)), a.section,
                    a.para, revised=a.text,
                ),
                b,
            )  # fmt: skip
    fresh = [w for w, _ in new.values()]
    code_check(fresh, notes, name)
    model_check(fresh, notes, name, ask, report)
    out: List[Written] = []
    for w in written:
        out.append(w)
        fold, b = new.get(id(w), (None, None))
        if fold is None:
            continue
        if fold.dropped:
            refused["check"] = refused.get("check", 0) + 1
            continue
        w.dropped, b.dropped = "folded with a later sentence", FOLDED
        out.append(fold)
        log.append([fold.text, w.text, b.text])
    return out
