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

from typing import Any, Dict, List, Sequence, Tuple

GROUP_MAX = 3


def event_when(note: Any) -> Tuple[int, int]:
    """(year, month) the note's text names first, else its article's date,
    else (9999, 0) so undated notes go last."""
    from src.engine.written_checks import _MONTH_N, _WHEN

    m = _WHEN.search(note.text)
    if m:
        return int(m.group(2)), _MONTH_N.get((m.group(1) or "").lower(), 0)
    date = str(note.date or "")
    if len(date) >= 7 and date[:4].isdigit() and date[5:7].isdigit():
        return int(date[:4]), int(date[5:7])
    return 9999, 0


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
            same_month = event_when(n) == event_when(last[-1]) and event_when(n)[1]
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
    out: List[Any], section: str, notes: Sequence[int], order: Dict[int, int]
) -> int:
    """Index in ``out`` after which a sentence citing ``notes`` goes: after
    the last kept sentence of ``section`` telling only earlier notes of the
    section's order; before its first sentence when none does; at the end of
    the section when its notes have no place in the order."""
    mine = [i for i, w in enumerate(out) if w.section == section]
    if not mine:
        return len(out) - 1
    keys = [order[n] for n in notes if n in order]
    if not keys:
        return mine[-1]
    key = min(keys)
    before = [
        i
        for i in mine
        if not out[i].dropped
        and any(n in order for n in out[i].notes)
        and max(order[n] for n in out[i].notes if n in order) < key
    ]
    return before[-1] if before else mine[0] - 1
