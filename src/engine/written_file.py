"""Filing a written page's notes into its planned sections, as a stage of
its own after the plan.

Why: the plan call does two jobs at once, choosing the sections and
putting every note in one. On 21 Nov, in three Luna draws of the Karachi
page (21 notes), the plan put 5 to 8 notes in no section, Daniel Pearl's
killing, the Karachi Six and al Sharabi's capture among them every time;
they reached the page only because code put them back, by shared words,
sometimes in the wrong section (Bin Atef's 2007 prison profile after his
2016 court case). The repository's placement step showed the same thing:
filing one sentence at a time under fixed headings was right on 93.6% of
hand labels, against 84-87% when one call made the headings and filed.

So the plan now only names the sections (and what is left out, which the
filing may overrule); ``file_notes()`` then asks, 8 notes per call, which
section each note belongs in, or why it is left out. A note with no usable
answer keeps the plan's choice.
"""

import re
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field

FILE = """You sort notes into the sections of an encyclopedia entry about \
the {what} "{name}". The notes are sentences taken from news articles. The \
sections:
{sections}

For each note give its number, its first three words copied, and the \
heading of the one section it belongs in, exactly as written above. Or, to \
leave it out, give the heading "leave out" and why: "someone else" (mainly \
about another person or thing, {name} only a passing mention), "no fact" \
(a vague or judging sentence) or "repeats N" (says less than note N). Use \
most notes: a note about a major event that happened at or to {name} (an \
attack, a killing, a capture, a trial) goes in a section, even when it \
names other people. For a place, an arrest, raid or killing there is about \
the place."""


class FiledNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    note: int
    starts_with: str = Field(description="The note's first three words, copied")
    section: str = Field(description="A heading as written, or 'leave out'")
    why: str = Field(default="", description="Why it is left out, or ''")


class Filing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    notes: List[FiledNote]


BATCH = 8


def _key(text: str) -> str:
    return " ".join(str(text or "").lower().strip(" #\"'.").split())


_EVENT = re.compile(
    r"\b(?:killed|killing|murder\w*|beheaded|arrested|arrests?|captured|capture|"
    r"raid\w*|attack\w*|bomb\w*|abducted|kidnapp\w*|confess\w*|seized)\b",
    re.IGNORECASE,
)
_SAYS = (
    r"(?:said|says|told|blamed|described|noted|called|recalled|added|does not|"
    r"did not|wants|wanted|asked|warned|denied|admitted)"
)


def must_keep(note: Any, name: str, what: str) -> bool:
    """A note the filing may not leave out: for a place, an event that names
    the place (Daniel Pearl's killing in Karachi was left out in two Karachi
    draws of three, 21 Nov); for a person, their own words or wishes, the
    surname opening the note ("Zaki blamed the Pentagon's ban on detainee
    artwork", "Zaki does not want his last name published", left out as "no
    fact" in every draw)."""
    text = note.text
    if what.startswith("place"):
        return bool(
            re.search(rf"\b{re.escape(name)}\b", text, re.I) and _EVENT.search(text)
        )
    if what.startswith("person") and name.split():
        surname = re.escape(name.split()[-1])
        return (
            re.match(rf"\W*(?:[\w,]+\s+){{0,3}}?{surname}\b\W*\s*{_SAYS}\b", text, re.I)
            is not None
        )
    return False


def _names(text: str, name: str) -> bool:
    """The note names the person by surname ("Bin Atef's attorney George
    Clarke described him as 'a big, friendly, funny guy'", left out as "no
    fact" on 21 Nov)."""
    surname = name.split()[-1] if name.split() else ""
    return (
        len(surname) > 2
        and re.search(rf"\b{re.escape(surname)}\b", text, re.I) is not None
    )


def _shared(note: Any, part: Any, by_n: Dict[int, Any]) -> int:
    from src.engine.written_checks import _content

    words = _content(note.text)
    return sum(len(words & _content(by_n[k].text)) for k in part.notes if k in by_n)


def file_messages(
    plan: Any, batch: Sequence[Any], name: str, what: str, block: Any
) -> List[Dict[str, str]]:
    sections = "\n".join(
        f"- {p.heading}: {p.covers}" if p.covers else f"- {p.heading}"
        for p in plan.parts
    )
    return [
        {
            "role": "system",
            "content": FILE.format(what=what, name=name, sections=sections),
        },
        {"role": "user", "content": f"Notes:\n{block(batch)}"},
    ]


def file_notes(
    plan: Any, notes: Sequence[Any], name: str, what: str, ask: Any,
    report: Dict[str, Any], block: Any,
) -> None:  # fmt: skip
    """Re-file ``plan``'s notes, in place, one batch of 8 per call. Opening
    notes the plan put in no section stay out of the sections."""
    from src.engine.written_checks import _first_words
    from src.engine.written_plan import LeftOut, opening_notes

    if len(plan.parts) < 2:
        return  # one section: nothing to choose
    heads = {_key(p.heading): p for p in plan.parts}
    placed = {n for p in plan.parts for n in p.notes}
    only_opening = set(opening_notes(plan)) - placed
    todo = [n for n in notes if n.n not in only_opening]
    answers: Dict[int, FiledNote] = {}
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        try:
            reply = ask(file_messages(plan, batch, name, what, block), Filing, 2048)
        except Exception as exc:
            report.setdefault("errors", []).append(f"filing: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        by_n = {n.n: n for n in batch}
        for a in reply.notes or []:
            note = by_n.get(a.note)
            if note is None or a.note in answers:
                continue
            if a.starts_with and _first_words(a.starts_with, 2) != _first_words(
                note.text, 2
            ):
                continue  # an answer about another note
            answers[a.note] = a
    moved, left, kept = 0, 0, 0
    by_all = {n.n: n for n in notes}
    for n, a in answers.items():
        section = _key(a.section)
        old = next((p for p in plan.parts if n in p.notes), None)
        named = what.startswith("person") and _names(by_all[n].text, name)
        no_fact = (a.why or "no fact").lower().startswith("no fact")
        if section.startswith("leave") and (
            must_keep(by_all[n], name, what) or (named and no_fact)
        ):
            plan.left_out = [o for o in plan.left_out if o.note != n]
            if old is None:  # into the section sharing most words with it
                best = max(plan.parts, key=lambda p: _shared(by_all[n], p, by_all))
                best.notes.append(n)
            kept += 1
            continue
        if section.startswith("leave"):
            why = a.why or "no fact"
            if old is not None:
                old.notes.remove(n)
            plan.left_out = [o for o in plan.left_out if o.note != n]
            plan.left_out.append(LeftOut(note=n, why=why))
            left += 1
            continue
        part = heads.get(section)
        if part is None or part is old:
            continue
        if old is not None:
            old.notes.remove(n)
        plan.left_out = [o for o in plan.left_out if o.note != n]
        part.notes.append(n)
        moved += 1
    # The plan's order for the notes it had (it was asked for date order),
    # newly filed ones after them; the section writer tells them in date order
    for p in plan.parts:
        p.notes = list(dict.fromkeys(p.notes))
    plan.parts = [p for p in plan.parts if p.notes]
    report["filing"] = {
        "answered": len(answers), "moved": moved, "left_out": left, "kept": kept,
    }  # fmt: skip
