"""Plan a written page before writing it, then write it a section at a time.

Why: on 21 Nov one call wrote the Karachi page straight from its 23 notes
(``WRITE`` in ``written_page.py``). It came out as seven sentences, opening
with one detainee's family and listing other men's arrests in note order.
Written by hand from the same notes, the page told three things in order:
the arrests of 2002-03 by year, Daniel Pearl's killing, then one family's
home, last because it is least about the city. It also weighed two dates
for one capture (three later reports, the Red Cross among them, against
one) and dropped two notes that were a model's filler. So the work is
split the way a writer does it:

1. ``plan()``: one call reads every note and says what the entity is (from
   the notes), why it matters in them, the parts of the page in the order a
   reader needs them, which notes go in each part, which notes are left out,
   and which notes give different dates or numbers for the same thing.
2. ``write_planned()``: one call for the opening (it sees the plan, so it
   can sum up the whole page) and one per part (it sees only that part's
   notes, the conflicts among them, and the plan's outline).

The result is a ``WPage``, so the checks and storage in ``written_page.py``
are the same for both ways of writing.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

PLAN = """You plan an encyclopedia entry about the {what} "{name}", for a \
reference work built from news reporting. You are given numbered notes: \
sentences taken from articles over many years, in no useful order. Don't \
write the entry yet; decide what it should say.

1. identity: what {name} is, in a few words, from the notes only ("a city \
in Pakistan", "a Yemeni man held at Guantánamo", "a U.S. Navy judge").
2. why_it_matters: one sentence on why {name} matters in these notes.
3. parts: the entry's sections, 1 to {max_parts}, in the order a reader \
needs them. The opening (written separately) says what {name} is, so no \
part is an overview, introduction or summary. Put the parts most about \
{name} itself first, and a part that is mainly about one other person or \
thing last. The usual topics for this kind of page are:
{headings}
Name each part after what its notes tell, in a few plain words ("Arrests \
in 2002 and 2003", "Military commission case", "Release to Oman"), not a \
general word like "Background", "Role" or "Notable events", and never a \
single date or article. For each part give its heading, a few words on \
what it tells, and the numbers of its notes, in the order they should be \
told (usually date order). Each note goes in one part at most.
4. left_out: notes to leave out: notes mainly about someone or something \
else, where {name} is only a setting or a passing mention; vague or \
judging sentences with no fact in them ("the city represents ..."); and \
notes that say less than another note already chosen. Keep a note about a \
major event that happened at or to {name} (an attack, a killing, a \
capture, a trial) even when it names other people.
5. conflicts: only real contradictions: two accounts of the same event \
that cannot both be true, such as two different years or places for one \
capture. For each, what it is about, what one side says and its notes, \
what the other side says and its notes. Not a conflict: an exact date and \
a vaguer one that fits it ("April 2003" and "April 29, 2003"); two \
descriptions that can both be true ("Saudi-born" and "Yemeni"); a count \
or post that changed over time (40 captives one year, 41 another); two \
different events. Most pages have none.
6. opening_notes: the notes that say what {name} is."""

SECTION = """You write one section of an encyclopedia entry about the \
{what} "{name}" ({identity}), for a reference work built from news \
reporting. The whole entry is planned like this:
{outline}

Write the section "{heading}" ({covers}) from its numbered notes below.

Rules:
- Write as a careful historian for a general reader: plain, neutral words, \
with the dates in the sentences. Tell it in order, joining related facts: \
several similar facts become one sentence or one paragraph, not one \
sentence each. Use short paragraphs, each about one thing.
- Use only what the notes say. Never add a name, number, date, place, \
cause or motive the notes don't give.
- Keep who said what: a claim, allegation, confession or assessment stays \
attributed. Loaded words (terrorist, jihadist, mastermind, extremist) \
appear only in quotation marks or attributed to whoever used them. Quoted \
words stay exactly as quoted.
- {conflicts}
- Write about the events, not about the notes: never "the notes say", \
"in the note" or "according to the notes".
- The opening already says what {name} is: don't say it again.
- Every sentence lists the numbers of the notes it uses in its notes \
field; never write note numbers into the sentence itself.
- About {words} words. Don't repeat what other sections cover."""

OPENING = """You write the opening of an encyclopedia entry about the \
{what} "{name}", for a reference work built from news reporting. The entry \
is planned like this:
{outline}

Write one to three sentences. The first says only what {name} is \
({identity}), in one short sentence, with no one person's story in it. The \
next sum up the main facts the sections tell ({why}), so a reader who \
stops here knows them. State the facts themselves: never "{name} matters \
because", "in this record" or "in these notes".

Use only what the notes below say, and keep who said what. Write about \
{name}, not about the notes ("the notes say"). Every sentence lists the \
numbers of the notes it uses in its notes field; never write note numbers \
into the sentence itself."""

NO_CONFLICT = "When two notes give different dates or numbers for the same thing, give both and cite both."
WEIGH = (
    "These notes disagree, and the section must say how: {items}. Weigh them: "
    "say which account more of the notes, the later reports or the more direct "
    "source (an official record, the Red Cross, a court filing) support, give "
    "the other too, and cite both. Don't just say the sources disagree."
)


class PlanPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    heading: str
    covers: str = Field(description="A few words on what this part tells")
    notes: List[int]


class Conflict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    about: str = Field(description="The one event or number they disagree on")
    one_side: str = Field(description="What one group of notes says")
    one_notes: List[int]
    other_side: str = Field(description="What the other group of notes says")
    other_notes: List[int]

    @property
    def notes(self) -> List[int]:
        return list(dict.fromkeys(self.one_notes + self.other_notes))

    def describe(self) -> str:
        nums = lambda ns: ", ".join(str(n) for n in ns)  # noqa: E731
        return (
            f"{self.about}: {self.one_side} (notes {nums(self.one_notes)}) against "
            f"{self.other_side} (notes {nums(self.other_notes)})"
        )


# Parts the opening already covers
_GENERIC = {"overview", "introduction", "summary", "about", "general"}


def _compatible(a: str, b: str) -> bool:
    """One side's numbers all in the other's: an exact date against a vaguer
    one that fits it ("April 2003", "April 29, 2003"), so no conflict."""
    na, nb = set(re.findall(r"\d+", a)), set(re.findall(r"\d+", b))
    return bool(na and nb) and (na <= nb or nb <= na)


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    identity: str
    why_it_matters: str
    parts: List[PlanPart]
    left_out: List[int]
    conflicts: List[Conflict]
    opening_notes: List[int]


def _outline(plan: Plan) -> str:
    return "\n".join(f"- {p.heading}: {p.covers}" for p in plan.parts)


def clean_plan(plan: Plan, numbers: Sequence[int], max_parts: int = 6) -> Plan:
    """Parts with known, unrepeated notes; empty parts and repeats dropped."""
    known = set(numbers)
    seen: set = set()
    parts = []
    first: List[int] = []
    for part in plan.parts[:max_parts]:
        ns = [
            n for n in dict.fromkeys(part.notes or []) if n in known and n not in seen
        ]
        seen.update(ns)
        heading = " ".join(str(part.heading or "").strip("# ").split())
        if ns and heading.lower() in _GENERIC:
            first += ns  # an overview part: its notes go to the opening
        elif ns and heading:
            parts.append(PlanPart(heading=heading, covers=part.covers or "", notes=ns))
    conflicts = []
    for c in plan.conflicts or []:
        one = [n for n in c.one_notes if n in seen]
        other = [n for n in c.other_notes if n in seen and n not in one]
        if one and other and not _compatible(c.one_side, c.other_side):
            conflicts.append(
                c.model_copy(update={"one_notes": one, "other_notes": other})
            )
    return Plan(
        identity=plan.identity or "", why_it_matters=plan.why_it_matters or "",
        parts=parts,
        left_out=[n for n in plan.left_out or [] if n in known and n not in seen],
        conflicts=conflicts,
        opening_notes=list(dict.fromkeys(
            [n for n in plan.opening_notes or [] if n in known] + first)),
    )  # fmt: skip


def plan_messages(
    notes_text: str, name: str, what: str, headings: Sequence[Tuple[str, str]],
    max_parts: int,
) -> List[Dict[str, str]]:  # fmt: skip
    heads = "\n".join(f"- {h}: {n}" if n else f"- {h}" for h, n in headings)
    return [
        {"role": "system", "content": PLAN.format(
            what=what, name=name, headings=heads or "- (choose your own)",
            max_parts=max_parts)},
        {"role": "user", "content": f"Notes about {name}:\n{notes_text}"},
    ]  # fmt: skip


def section_messages(
    plan: Plan, part: PlanPart, notes_text: str, name: str, what: str, words: int
) -> List[Dict[str, str]]:
    mine = set(part.notes)
    items = [c.describe() for c in plan.conflicts if mine & set(c.notes)]
    conflicts = WEIGH.format(items="; ".join(items)) if items else NO_CONFLICT
    system = SECTION.format(
        what=what, name=name, identity=plan.identity, outline=_outline(plan),
        heading=part.heading, covers=part.covers, conflicts=conflicts, words=words,
    )  # fmt: skip
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Notes for this section:\n{notes_text}"},
    ]


def opening_messages(
    plan: Plan, notes_text: str, name: str, what: str
) -> List[Dict[str, str]]:
    system = OPENING.format(
        what=what, name=name, identity=plan.identity, why=plan.why_it_matters,
        outline=_outline(plan),
    )  # fmt: skip
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Notes:\n{notes_text}"},
    ]


def max_parts(n_notes: int) -> int:
    return 2 if n_notes < 8 else 4 if n_notes < 40 else 6


def opening_notes(plan: Plan, limit: int = 12) -> List[int]:
    """The plan's notes on what the entity is, then each part's first two."""
    out = list(plan.opening_notes)
    for part in plan.parts:
        out += part.notes[:2]
    return list(dict.fromkeys(out))[:limit]


def section_words(total: int, part: PlanPart, plan: Plan) -> int:
    used = sum(len(p.notes) for p in plan.parts) or 1
    return max(40, int(total * len(part.notes) / used))


def failed(report: Dict[str, Any], step: str, exc: Exception) -> None:
    report.setdefault("errors", []).append(f"{step}: {str(exc)[:200]}")


def write_planned(
    notes: Sequence[Any], name: str, what: str, headings: Sequence[Tuple[str, str]],
    words: int, ask: Any, report: Dict[str, Any], block: Any, page_model: Any,
    section_model: Any, sentence_model: Any,
) -> Optional[Any]:  # fmt: skip
    """A ``page_model`` (``WPage``) written by plan, opening and sections, or
    None when the plan call fails. ``block`` prints notes; the models are
    passed in from ``written_page`` so this module needs no import of it."""
    by_n = {n.n: n for n in notes}
    k = max_parts(len(notes))
    try:
        raw = ask(plan_messages(block(notes), name, what, headings, k), Plan, 4096)
    except Exception as exc:
        failed(report, "plan", exc)
        return None
    report["calls"] = report.get("calls", 0) + 1
    plan = clean_plan(raw, list(by_n), k)
    report["plan"] = {
        "identity": plan.identity, "why": plan.why_it_matters,
        "parts": [{"heading": p.heading, "notes": p.notes} for p in plan.parts],
        "left_out": plan.left_out,
        "conflicts": [c.model_dump() for c in plan.conflicts],
    }  # fmt: skip
    if not plan.parts:
        return None
    sections = []
    for part in plan.parts:
        shown = [by_n[n] for n in part.notes]
        msgs = section_messages(
            plan, part, block(shown), name, what, section_words(words, part, plan)
        )
        try:
            reply = ask(msgs, section_model, 4096)
        except Exception as exc:
            failed(report, f"section {part.heading}", exc)
            continue
        report["calls"] += 1
        sections.append({"heading": part.heading, "paragraphs": [
            p.model_dump() for p in reply.paragraphs]})  # fmt: skip
    shown = [by_n[n] for n in opening_notes(plan)]
    try:
        opening = ask(
            opening_messages(plan, block(shown), name, what), sentence_model, 2048
        )
        report["calls"] += 1
        first = [s.model_dump() for s in opening.sentences]
    except Exception as exc:
        failed(report, "opening", exc)
        first = []
    return page_model.model_validate({"opening": first, "sections": sections})
