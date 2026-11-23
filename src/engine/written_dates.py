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
    r"\b(on|as of|by|since|until|through|last)\s+(?:a\s+)?(" + "|".join(_DAYS) + r")\b",
    re.I,
)
# A month already given with its year in the sentence: "In March 2013"
_MONTH_YEAR = re.compile(
    rf"(?i:{_MONTH_RE})\.?\s+(?:\d{{1,2}}(?:\s*(?:and|-|–)\s*\d{{1,2}})?,?\s+)?"
    r"(?:19|20)\d\d"
)
# A year right after the matched month: "February 6–7, 2013" (round 32 made
# it "February 6–7, 2013, 2013")
_YEAR_NEXT = re.compile(r"^,?\s+(?:19|20)\d\d\b")
# A month joined to a later one that has its year: "in May and July 2003"
# (round 34 made it "May 2015 and July 2003" from a 2015 article)
_JOINED = re.compile(
    rf"^(?:\s+\d{{1,2}})?,?\s*(?:and|to|through|or|-|–)\s+(?i:{_MONTH_RE})\.?"
    r"(?:\s+\d{1,2}(?:\s*(?:-|–)\s*\d{1,2})?)?,?\s+(?:19|20)\d\d\b"
)
# The same year twice in a row: a change that makes it is refused
_TWICE = re.compile(r"\b((?:19|20)\d\d),?\s+\1\b")
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
# "In September", "On April 13", "the Pentagon's November policy": a month
# after a word that sets a time, with no year in the sentence
_BARE_MONTH = re.compile(
    rf"(\b(?i:in|on|as of|by|since|until|through|during|late|early|mid-?|last)\s+|[’']s\s+)"
    rf"(?i:{_MONTH_RE})(\.?)(\s+\d{{1,2}}(?:\s*(?:and|-|–)\s*\d{{1,2}})?\b)?",
)
# A date that lies ahead of its article: "declarations ... by March 19" in a
# February 2018 article is March 2018 (round 30 wrote "by March 19, 2017")
_AHEAD = re.compile(
    r"\b(?:by|until|till|before|deadline|due|scheduled|set for|week of|next|"
    r"will|would|planned|expected|is to|are to|was to|were to)\b",
    re.I,
)
# "on April 24, 2013 100 captives": a comma after the year it was given
_NO_COMMA = re.compile(
    r"(\b[A-Z][a-z]+\.? \d{1,2}(?:\s*(?:and|-|–)\s*\d{1,2})?, (?:19|20)\d\d)(?= [\w$])"
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


def _article_days(w: Any, by_n: dict) -> List[date]:
    out = []
    for k in w.notes:
        try:
            out.append(date.fromisoformat(str(by_n[k].date)[:10]))
        except (KeyError, ValueError):
            continue
    return out


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
    if _article_day(w, by_n) is None:
        return text
    # A month the sentence already dates is left alone ("In March 2013 ...
    # by late March"); another month is dated even beside a year (round 31:
    # "In February 2018, Zaki blamed ... the Pentagon's November policy")
    dated = {_MONTH_OF.get(m.group(1)[:3].lower()) for m in _MONTH_YEAR.finditer(text)}

    def month(m: "re.Match[str]") -> str:
        n = _MONTH_OF.get(m.group(2)[:3].lower(), 0)
        if not n or not m.group(2)[0].isupper():
            return m.group(0)  # "may", "march": a verb
        if n in dated or (_YEAR.search(text) and not dated):
            return m.group(0)  # dated already, or a year with no month to go by
        if _YEAR_NEXT.match(text[m.end() :]) or _JOINED.match(text[m.end() :]):
            return m.group(0)
        # the note that names this month, else the first dated one (round 32
        # dated "by March 19" from another note's 2017 article)
        day_word = f"{m.group(2)} {m.group(4).strip()}" if m.group(4) else ""
        named = day_word and any(
            day_word.lower() in by_n[k].text.lower() for k in w.notes if k in by_n
        )
        d = _article_day(w, by_n, day_word if named else m.group(2))
        if d is None:
            return m.group(0)
        if not any(
            m.group(2).lower() in by_n[k].text.lower() for k in w.notes if k in by_n
        ):
            # no note names the month: an article from that month dates it,
            # and notes from different years leave it (round 47: "In June"
            # citing a February 2018 note and a June 2018 one became June
            # 2017, from the first note)
            days = _article_days(w, by_n)
            same = [x for x in days if x.month == n]
            if same:
                d = same[0]
            elif len({x.year for x in days}) > 1:
                return m.group(0)
        if n > d.month:
            ahead = _AHEAD.search(
                text[max(0, m.start() - 40) : m.start() + len(m.group(1))]
            )
            if ahead:
                year = d.year  # a deadline or plan, later in the article's year
            elif n - d.month <= 2 or (
                dated and "'" not in m.group(1) and "’" not in m.group(1)
            ):
                # a month or two on: past or coming, unclear; or a time
                # phrase in a sentence that dates another month (round 32:
                # "In March 2013 ... in June" became June 2012). A possessive
                # names a thing from before the article ("In February 2018,
                # ... the Pentagon's November ban") and is dated.
                return m.group(0)
            else:
                year = d.year - 1  # "In September" in an April article
        else:
            year = d.year
        day = m.group(4) or ""
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{day}{',' if day else ''} {year}"

    text = _BARE_MONTH.sub(month, text)
    if text == w.text:
        return text
    text = _NO_COMMA.sub(r"\1,", text)
    return w.text if _TWICE.search(text) and not _TWICE.search(w.text) else text


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
