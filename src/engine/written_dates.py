"""Dates written out in code where a sentence leans on its article for them.

Why: in round 29 (22 Nov) the best pages still had "As of Wednesday, 100
captives were reported on hunger strike", "officials counted 37 hunger
strikers on Friday", "In September, a Yemeni prisoner was found dead in his
cell" (the note is from April 2013, so September 2012) and "the Pentagon's
November policy banning release of detainee artwork". Each sentence's own
notes carry their article's date, which is enough to name the day or year,
so code writes it in; no model call, nothing else in the sentence changes.
"""

import re
from datetime import date, timedelta
from typing import Any, List, Optional, Sequence

from src.engine.written_groups import _MONTH_OF, _MONTH_RE

_MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
]  # fmt: skip
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_WEEKDAY = re.compile(
    r"\b(on|as of|by|since|until|through|last)\s+(" + "|".join(_DAYS) + r")\b", re.I
)
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
# "In September", "On April 13", "the Pentagon's November policy": a month
# after a word that sets a time, with no year in the sentence
_BARE_MONTH = re.compile(
    rf"(\b(?i:in|on|by|since|until|through|during|late|early|mid-?|last)\s+|[’']s\s+)"
    rf"(?i:{_MONTH_RE})(\.?)(\s+\d{{1,2}}\b)?",
)


def _article_day(w: Any, by_n: dict, word: str = "") -> Optional[date]:
    """The article date of the sentence's note that uses ``word`` (else its
    first dated note)."""
    notes = [by_n[n] for n in w.notes if n in by_n and len(str(by_n[n].date)) >= 10]
    if word:
        notes = [n for n in notes if word.lower() in n.text.lower()] or notes
    for n in notes:
        try:
            return date.fromisoformat(str(n.date)[:10])
        except ValueError:
            continue
    return None


def _named(d: date, year: bool) -> str:
    return f"{_MONTHS[d.month - 1]} {d.day}" + (f", {d.year}" if year else "")


def date_sentence(w: Any, by_n: dict) -> str:
    """``w``'s text with its weekdays and year-less months dated from its
    notes' articles."""
    text = w.text
    has_year = bool(_YEAR.search(text))

    def weekday(m: "re.Match[str]") -> str:
        d = _article_day(w, by_n, m.group(2))
        if d is None:
            return m.group(0)
        back = (d.weekday() - _DAYS.index(m.group(2).capitalize())) % 7
        return f"{m.group(1)} {_named(d - timedelta(days=back), not has_year)}"

    text = _WEEKDAY.sub(weekday, text)
    if _YEAR.search(text):
        return text
    d = _article_day(w, by_n)
    if d is None:
        return text

    def month(m: "re.Match[str]") -> str:
        n = _MONTH_OF.get(m.group(2)[:3].lower(), 0)
        if not n or not m.group(2)[0].isupper():
            return m.group(0)  # "may", "march": a verb
        year = d.year - (n > d.month)
        day = m.group(4) or ""
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{day}{',' if day else ''} {year}"

    return _BARE_MONTH.sub(month, text, count=1)


def own_dates(written: List[Any], notes: Sequence[Any]) -> List[List[str]]:
    """Date every kept sentence's weekdays and year-less months in place;
    each change as [before, after]."""
    by_n = {n.n: n for n in notes}
    changed = []
    for w in written:
        if w.dropped:
            continue
        new = date_sentence(w, by_n)
        if new != w.text:
            changed.append([w.text, new])
            w.text = new
    return changed
