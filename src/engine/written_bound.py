"""A sentence that opens on its note's article month for an act the note
does not date says only that the act had happened by then.

Why: on 23 Nov the checkers of 14 tuning pages (cr) found 15 false dates in
288 sentences, and 9 were "In <article month>, X did ..." for an act the
note tells without a date: a ruling decided two months before the
article, or an agency's plan from years earlier, opened on the article's
month. Telling the
writer the note gives no date (round 45) did not change how often it wrote
them (42 openers before and after), and re-dating from an earlier note
(rounds 44-47) only works when one exists and tells the same act, which
was 0 of 30 on the held-out pages. But the article's month is always a
true bound: the act had happened by then.

By the stages rule: code finds each such opener (a cited note of that
month whose text gives no year, a sentence that does not tell what someone
said, since a statement is dated by when it was made); one call per batch
says whether the note tells the act as news of that time or as something
that may be older, and for the second rewrites only the opening and the
verb ("By January 2018, the official had struck down ..."); code keeps the
rewrite only when it opens "By <that month>," and changes at most
``MAX_CHANGED`` other words, every name and number kept.
"""

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

BOUND = """You check dates in an encyclopedia entry about "{name}". Each \
numbered sentence opens on the month its note's article appeared, but the \
note's text does not say when the act happened.

For each, say in a few words how the note tells the act, then answer:
- told: "new" if the note tells it as news of that time (it happened in the \
days or weeks before the article: "on Tuesday", "this week", "has just", a \
new count, an announcement); "older" if it may be older (a ruling, event or \
fact the article recalls or states as background); "unsure" otherwise.
- text: for "older" or "unsure", the sentence rewritten to say only that it \
was so by then: open with "By <Month Year>," and change the verb to fit \
("had struck down", "had been"), changing no other words. Empty for \
"new"."""

BATCH = 8
MAX_CHANGED = 3  # words besides the opening

# a note that tells news of its day: dated by its article
_NEWS = re.compile(
    r"\b(?:today|yesterday|this (?:week|month)|last week|on (?:Monday|Tuesday|"
    r"Wednesday|Thursday|Friday|Saturday|Sunday))\b",
    re.I,
)


class Bound(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int = Field(description="The sentence's number")
    how: str = Field(default="", description="How the note tells the act")
    told: str = Field(description='"new", "older" or "unsure"')
    text: str = Field(default="", description="The rewritten sentence, or empty")


class Bounds(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: List[Bound]


def candidates(
    written: Sequence[Any], by_n: Dict[int, Any], orig: Optional[Dict[str, Any]] = None
) -> List[Tuple[Any, Any, List[Any]]]:
    """(sentence, opener match, its cited notes of that month) for each kept
    sentence opening "In/On <month> <year>," on a cited note's article month
    whose notes of that month give no year and tell no news of their day,
    and that does not tell what someone said."""
    from src.engine.written_retell import _tells_saying
    from src.engine.written_when_who import _MONTHS, _OPENER, _YEARS, sources_of

    out = []
    for w in written:
        if w.dropped or w.summary:
            continue
        m = _OPENER.match(w.text)
        if not m or _tells_saying(w.text[m.end() :]):
            continue
        month = f"{m.group(4)}-{_MONTHS.index(m.group(2)) + 1:02d}"
        own = [
            n for n in sources_of(w, by_n, orig or {}) if str(n.date or "")[:7] == month
        ]
        if not own or any(_YEARS.search(n.text) or _NEWS.search(n.text) for n in own):
            continue
        out.append((w, m, own))
    return out


def _marks(text: str) -> set:
    from src.engine.written_checks import _numbers

    return _numbers(text) | set(re.findall(r"\b[A-Z][\w'’-]+", text))


def usable(old: str, new: str, head: str, opener_end: int) -> bool:
    """``new`` opens on ``head`` and changes at most ``MAX_CHANGED`` words of
    the rest of ``old``, keeping every name and number."""
    if not new.startswith(head + " "):
        return False
    a, b = old[opener_end:].split(), new[len(head) :].split()
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    if changed > MAX_CHANGED:
        return False
    return _marks(old[opener_end:]) <= _marks(new[len(head) :])


def _messages(
    todo: Sequence[Tuple[Any, Any, List[Any]]], name: str
) -> List[Dict[str, str]]:
    parts = []
    for i, (w, _, own) in enumerate(todo, 1):
        lines = "\n".join(f"  Its note [{n.date}]: {n.text}" for n in own)
        parts.append(f"Sentence {i}: {w.text}\n{lines}")
    return [
        {"role": "system", "content": BOUND.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def bound_openers(
    written: List[Any], by_n: Dict[int, Any], name: str, ask: Any,
    report: Dict[str, Any], orig: Optional[Dict[str, Any]] = None,
) -> None:  # fmt: skip
    """Rewrite each opener the model calls older or unsure as "By <month>,";
    each change under ``report["bounded"]`` as [old, new], every answer
    under ``report["bound_answers"]``."""
    todo = candidates(written, by_n, orig)
    report["bound_asked"] = report.get("bound_asked", 0) + len(todo)
    log: List[List[str]] = report.setdefault("bounded", [])
    answers: List[Dict[str, Any]] = report.setdefault("bound_answers", [])
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            reply = ask(_messages(batch, name), Bounds, 4096)
        except Exception as exc:
            report.setdefault("errors", []).append(f"bound: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for c in reply.checks or []:
            if not 1 <= c.sentence <= len(batch):
                continue
            w, m, _ = batch[c.sentence - 1]
            told = str(c.told or "").strip().lower()
            new = " ".join(str(c.text or "").split())
            entry = {"sentence": w.text, "how": str(c.how or "")[:300],
                     "told": told, "text": new, "done": "left"}  # fmt: skip
            answers.append(entry)
            if told == "new" or w.dropped:
                continue
            lead = re.match(r"^\W*", w.text).group(0)
            head = f"{lead}By {m.group(2)} {m.group(4)},"
            if not usable(w.text, new, head, m.end()):
                entry["done"] = "refused"
                continue
            log.append([w.text, new])
            w.text, entry["done"] = new, "bounded"
