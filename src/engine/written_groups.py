"""A section's notes put in order and in groups by code, before the section
writer sees them, and missed notes written back in their place.

Why: on 21 Nov most of the worst sentences on the written pages came from
the stage that adds missed notes at the end of a section, alone and out of
context ("resettled ... in June 2017", the article's date; "the Periodic
Review Board denied his release" seven sentences after "his" was named).
The section writer, handed fifteen notes in the plan's order, wrote about
nine. So code now sorts a section's notes by the date they name (else
their article's), puts notes about the same person or the same month
together (three at most), and shows them as numbered groups to be told in
that order. A filled sentence goes after the last kept sentence of its
section that tells an earlier note, not at the end.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GROUP_MAX = 3


_MONTH_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# "September 11, 2002", "Sept. 10-11, 2002", "April 2003", "2002": round 18
# read "September 11, 2002" as 2002 with no month, so on every Karachi draw
# bin Attash's September 2002 capture came before Hamdoun's February arrest
_DATE = re.compile(
    rf"\b(?:{_MONTH_RE}\.?\s+(?:(\d{{1,2}})(?:\s*[-–]\s*\d{{1,2}})?,?\s+)?)?"
    r"((?:19|20)\d\d)\b",
    re.IGNORECASE,
)
_MONTH_OF = {
    m: i
    for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)
}

When = Tuple[int, int, int]


def text_when(text: str) -> Optional[When]:
    """(year, month, day) of the first date ``text`` names, 0 for an
    unnamed month or day; None when it names no year."""
    m = _DATE.search(text or "")
    if not m:
        return None
    month = _MONTH_OF.get((m.group(1) or "")[:3].lower(), 0)
    return int(m.group(3)), month, int(m.group(2) or 0)


def event_when(note: Any) -> When:
    """The date the note's text names first, else its article's date, else
    (9999, 0, 0) so undated notes go last."""
    found = text_when(note.text)
    if found:
        return found
    date = str(note.date or "")
    if len(date) >= 7 and date[:4].isdigit() and date[5:7].isdigit():
        year, month = int(date[:4]), int(date[5:7])
        # "On April 21, 84 were on hunger strike": a day with no year is in
        # the article's year, or the year before when its month is later
        # (round 20 left the camp's hunger-strike section unsorted by day)
        m = _DAY_ONLY.search(note.text)
        if m:
            day_month = _MONTH_OF.get(m.group(1)[:3].lower(), 0)
            return year - (day_month > month), day_month, int(m.group(2))
        # "In March", "By late March": early before the 15th, late after
        m = _MONTH_ONLY.search(note.text)
        if m:
            only = _MONTH_OF.get(m.group(2)[:3].lower(), 0)
            part = (m.group(1) or "").lower()
            day = (
                5 if part.startswith("early") else 25 if part.startswith("late") else 15
            )
            return year - (only > month), only, day
        return year, month, 0
    return 9999, 0, 0


_MONTH_ONLY = re.compile(
    rf"\b(?:in|by|since|during|until|through)\s+(early|late|mid-?)?\s*{_MONTH_RE}\b(?!\.?\s+\d)",
    re.I,
)
_DAY_ONLY = re.compile(rf"\b{_MONTH_RE}\.?\s+(\d{{1,2}})\b(?!,?\s+(?:19|20)\d\d)", re.I)


def group_notes(notes: Sequence[Any], name: str) -> List[List[Any]]:
    """``notes`` sorted by ``event_when`` and cut into groups: a note joins
    the group before it when it names the same month or a person that group
    names, up to ``GROUP_MAX`` notes."""
    from src.engine.written_plan import _surnames

    ordered = sorted(notes, key=event_when)
    groups: List[List[Any]] = []
    for n in ordered:
        if groups and len(groups[-1]) < GROUP_MAX:
            last = groups[-1]
            when, before = event_when(n), event_when(last[-1])
            same_month = when[:2] == before[:2] and when[1]
            people = set().union(*(_surnames(x.text, name) for x in last))
            if same_month or (_surnames(n.text, name) & people):
                last.append(n)
                continue
        groups.append([n])
    return groups


GROUPS_NOTE = (
    "The notes come in numbered groups, in date order. Tell the groups in "
    "that order, each in one or two sentences (a group may share a sentence "
    "with the group before it when they are about the same thing), and skip "
    "none."
)


def groups_block(groups: Sequence[Sequence[Any]], block: Any) -> str:
    return "\n\n".join(f"Group {i}:\n{block(g)}" for i, g in enumerate(groups, 1))


def place_after(
    out: List[Any], section: str, w: Any, order: Dict[int, int],
    when: Callable[[Any], When],
) -> int:  # fmt: skip
    """Index in ``out`` after which the filled sentence ``w`` goes in
    ``section``. A sentence naming a date goes after the last kept sentence
    there dated no later (round 18 put a June 2017 fill on Bin Atef's page
    before his 2009 clearance, by the note before it in the list); one
    naming none goes after the last kept sentence telling only earlier
    notes of the section's ``order``. Before the section's first sentence
    when none fits; at its end when its notes have no place in the order."""
    mine = [i for i, x in enumerate(out) if x.section == section]
    if not mine:
        return len(out) - 1
    kept = [i for i in mine if not out[i].dropped]
    own = text_when(w.text)
    if own:
        before = [i for i in kept if when(out[i]) <= own]
        return before[-1] if before else mine[0] - 1
    keys = [order[n] for n in w.notes if n in order]
    if not keys:
        return mine[-1]
    key = min(keys)
    before = [
        i
        for i in kept
        if any(n in order for n in out[i].notes)
        and max(order[n] for n in out[i].notes if n in order) < key
    ]
    return before[-1] if before else mine[0] - 1


def sentence_when(by_n: Dict[int, Any]) -> Callable[[Any], When]:
    """A written sentence's date: the first it names, else its notes'
    earliest."""

    def when(w: Any) -> When:
        own = text_when(w.text)
        if own:
            return own
        dates = [event_when(by_n[n]) for n in w.notes if n in by_n]
        return min(dates) if dates else (9999, 0, 0)

    return when
