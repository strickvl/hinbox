"""One narrow check on the finished page: is each date the date of what it
dates, and is each act given to whoever did it.

Why: on 22 Nov the blind readers of 16 held-out pages (pages never used
while tuning) named, more than any other factual fault, an article's
publication date put on an older event it only looks back on (five of 16
readers) and an act or statement given to the wrong person, or a related
case's detail given to this one (three). The general sentence check asks
about both among many things and let them through; by the stages rule a
step that struggles is split, so this asks only these two questions. It
runs on the page as it will be shown, after the drafts are picked and the
code passes have dated and moved sentences.

A wrong sentence is mended only when the model's fix changes a few words,
brings in no name, number or month its notes lack, and takes out no date
its notes write or the date pass worked out; nothing is dropped. Round 34
dropped 12 true sentences and 0 false ones on the tuning pages (some with
the model's own reasoning ending "so correct"), and 3 of its 5 wrong mends
took out a year or day the date pass had rightly worked out ("on Monday" in
a Monday article). A sentence it would have dropped, or whose fix is
refused, goes under ``report["when_who_flagged"]``; every mend under
``report["when_who"]``.
"""

import difflib
import re
from datetime import date
from typing import Any, Callable, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field

WHEN_WHO = """You check two things in each numbered sentence of an \
encyclopedia entry about "{name}", against the notes it was written from. \
Each note starts with its article's date and weekday in brackets: the day the \
article appeared, not always the day of what it tells.

1. WHEN. For each date in the sentence (a year, a month, a day): do its notes \
date that thing to that time? A date written in the note's own text wins over \
its bracket date. An article's date is right for what the article reports as \
new or current ("said on Tuesday", "is on hunger strike", "this week") and for \
when something was said or reported; a day or year worked out from it ("on \
Monday" in a Monday article is that day; "In September" in an April article \
is the September before) is right. It is wrong for an older event, ruling, \
report or document the article only looks back on: "In April 2018, the \
Senate report found ..." from an April 2018 article about a 2014 report is \
wrong.
2. WHO. Is each act, decision or statement given to the person or body the \
notes give it to, and is it about this entry's subject and not a related \
person or case the notes describe? A judge's ruling given to a lawyer, or \
one detainee's transfer to another, is wrong.

For each sentence: copy its first three words; say in a few words what you \
found, or "none"; then give "when" (right, wrong or no date) and "who" (right \
or wrong). Answer "wrong" only when you are sure the notes contradict the \
sentence. If something is wrong, give the sentence mended with as few words \
changed as possible: the right date or actor from the notes, or the wrong \
date taken out when the notes give none. Leave "fixed" empty otherwise. Judge \
only by the notes shown."""

BATCH = 8
NOTE_CHARS = 1500
MAX_NOTES = 6
MAX_CHANGED = 8  # words a fix may add or remove


class WhenWho(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int = Field(description="The sentence's number")
    starts_with: str = Field(description="Its first three words, copied")
    problem: str = Field(default="", description="What you found, or none")
    when: str = Field(description="right, wrong or no date")
    who: str = Field(description="right or wrong")
    fixed: str = Field(default="", description="The mended sentence, or empty")


class WhenWhoChecks(BaseModel):
    model_config = ConfigDict(extra="ignore")
    checks: List[WhenWho]


def _dated(note: Any) -> str:
    """ "2018-09-24, a Monday", or the stored date as it is."""
    raw = str(note.date or "")
    try:
        day = date.fromisoformat(raw[:10])
    except ValueError:
        return raw or "no date"
    return f"{raw[:10]}, a {day.strftime('%A')}"


def _messages(
    batch: Sequence[Any], by_n: Dict[int, Any], name: str
) -> List[Dict[str, str]]:
    parts = []
    for i, w in enumerate(batch, 1):
        shown = [by_n[n] for n in w.notes[:MAX_NOTES] if n in by_n]
        lines = "\n".join(f"  - [{_dated(n)}] {n.text[:NOTE_CHARS]}" for n in shown)
        parts.append(f"Sentence {i}: {w.text}\nIts notes:\n{lines}")
    return [
        {"role": "system", "content": WHEN_WHO.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


_DATE_WORD = re.compile(
    r"^(?:(?:19|20)\d\d|\d{1,2}|jan\w*|feb\w*|mar\w*|apr\w*|may|june?|july?|"
    r"aug\w*|sept?\w*|oct\w*|nov\w*|dec\w*|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday)$",
    re.I,
)
# Words that say who said or reported a thing: a mend that takes them out
# puts the article's date on the event (round 34: "In August 2012,
# prosecutors said the centre had closed in June 2003" lost "prosecutors said")
_SAYING = re.compile(
    r"^(?:said|says|say|told|stated|reported|according|announced|wrote)$", re.I
)


def _words(text: str) -> List[str]:
    return re.findall(r"[\w’'$%-]+", text)


def usable_fix(
    old: str, fixed: str, sources: Sequence[str], name: str,
    page_names: Sequence[str], derived: Sequence[str] = (),
) -> bool:  # fmt: skip
    """``fixed`` changes at most ``MAX_CHANGED`` words of ``old``, keeps at
    least half of them, adds no name, number or month its notes lack, and
    takes out no saying word and no date word its notes write or the date
    pass added (``derived``)."""
    from src.engine.written_checks import missing_details

    fixed = fixed.strip()
    if not fixed or fixed == old.strip():
        return False
    a, b = _words(old), _words(fixed)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    kept = sum(i2 - i1 for op, i1, i2, _, _ in ops if op == "equal")
    if changed > MAX_CHANGED or kept < len(a) / 2:
        return False
    hay = " ".join(sources).lower()
    keep = {d.lower() for d in derived}
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            continue
        for x in (x for x in a[i1:i2] if x not in b[j1:j2]):
            if _SAYING.match(x):
                return False
            low = x.lower()
            if _DATE_WORD.match(x) and (
                low in keep or re.search(rf"\b{re.escape(low)}\b", hay)
            ):
                return False
    return not missing_details(fixed, sources, name, page_names)


def _first(text: str) -> str:
    from src.engine.written_checks import _first_words

    return _first_words(text, 2)


def when_who(
    written: List[Any], notes: Sequence[Any], name: str,
    ask: Callable[..., Any], report: Dict[str, Any],
) -> None:  # fmt: skip
    """Mend kept sentences the model finds misdated or given to the wrong
    actor, when the mend passes ``usable_fix()``; nothing is dropped, and a
    failed call leaves its sentences as they are."""
    from src.engine.written_checks import page_names_of

    by_n = {n.n: n for n in notes}
    names = page_names_of(notes)
    log: List[List[str]] = report.setdefault("when_who", [])
    flagged: List[List[str]] = report.setdefault("when_who_flagged", [])
    # the words the date pass added to each sentence it dated
    added: Dict[str, List[str]] = {}
    for before, after in report.get("own_dates") or []:
        added[after] = [x for x in _words(after) if x not in _words(before)]
    todo = [w for w in written if not w.dropped and not w.summary]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            reply = ask(_messages(batch, by_n, name), WhenWhoChecks, 4096)
        except Exception as exc:
            report.setdefault("errors", []).append(f"when_who: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for c in reply.checks or []:
            if not 1 <= c.sentence <= len(batch):
                continue
            w = batch[c.sentence - 1]
            if _first(c.starts_with) != _first(w.text):
                continue  # the answer is about another sentence
            when = str(c.when or "").strip().lower()
            who = str(c.who or "").strip().lower()
            if when != "wrong" and who != "wrong":
                continue
            problem = str(c.problem or "")[:200]
            sources = [by_n[n].text for n in w.notes if n in by_n]
            fix = str(c.fixed or "").strip()
            if usable_fix(w.text, fix, sources, name, names, added.get(w.text, ())):
                log.append([w.text, fix, problem])
                w.text = fix
            else:
                flagged.append([w.text, fix, problem])
