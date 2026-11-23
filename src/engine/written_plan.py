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

1. The plan: one call reads every note and says what the entity is (from
   the notes), the parts of the page in the order a reader needs them,
   which notes go in each part, and which notes are left out and why.
2. Contradictions (``find_conflicts()``): one call per part, on that
   part's notes only, lists notes that give different dates, places or
   numbers for the same event.
3. Sections: one call per part, shown only its notes and the
   contradictions among them, told to weigh them.
4. The opening (``opening_messages()``): one call, last, shown the written
   sections, so it sums them up instead of repeating the first one.

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
thing last; for a person, the parts follow the order of their life \
(capture, detention, release). The usual topics for this kind of page are:
{headings}
Name each part after what its notes tell, in a few plain words ("Arrests \
in 2002 and 2003", "Military commission case", "Release to Oman"), not a \
general word like "Background", "Role" or "Notable events", and never a \
single date or article. For each part give its heading, a few words on \
what it tells, and the numbers of its notes, in the order they should be \
told (usually date order). Each note goes in one part at most.
4. left_out: the notes to leave out, each with why: "someone else" (mainly \
about another person or thing, {name} only a setting or a passing \
mention), "no fact" (a vague or judging sentence, "the city represents \
..."), or "repeats N" (says less than note N, which is used). Use most \
notes: a note about a major event that happened at or to {name} (an \
attack, a killing, a capture, a trial) is never left out, even when it \
names other people. For a place, a note on something that happened there \
(an arrest, a raid, a killing) is about the place, not "someone else".
5. opening_notes: the notes that say what {name} is."""

SECTION = """You write one section of an encyclopedia entry about the \
{what} "{name}" ({identity}), for a reference work built from news \
reporting. The whole entry is planned like this:
{outline}

Write the section "{heading}" ({covers}) from its numbered notes below.

Rules:
- Write as a careful historian for a general reader: plain, neutral words, \
with the dates in the sentences (the month as well as the year when the \
note gives it), in date order. Tell it in order, joining related facts: \
several similar facts become one sentence or one paragraph, not one \
sentence each. Use short paragraphs, each about one thing.
- Use only what the notes say. Never add a name, number, date, place, \
cause or motive the notes don't give.
- Keep who said what: a claim, allegation, confession or assessment stays \
attributed. Loaded words (terrorist, jihadist, mastermind, extremist) \
appear only in quotation marks or attributed to whoever used them. Quoted \
words stay exactly as quoted.
- {conflicts}
- Each note shows its article's date. You may use it for when something \
was said, as the month and year ("In May 2007, Zaki said"), never as the \
day of the event itself. Never put it on an older report, document or \
event the note only looks back on (not "In April 2018, the Senate report \
said"), nor on a lasting fact ("he did not want his name published"): \
leave those undated unless the note dates them. Date the event or the \
statement, not the article: never "a May 2017 article said".
- Name a person in full (as the notes give the name) the first time the \
section mentions them. Don't join facts from different dates into one \
sentence unless each keeps its own date.
- Write about the events, not about the notes: never "the notes say", \
"in the note" or "according to the notes".
- The opening says what {name} is: don't say it again here.
- Every sentence lists the numbers of the notes it uses in its notes \
field; never write note numbers into the sentence itself.
- About {words} words. Don't repeat what other sections cover."""

OPENING = """You write the opening of an encyclopedia entry about the \
{what} "{name}", for a reference work built from news reporting. The \
entry's sections are already written; they are below, each sentence with \
the numbers of its notes, after the notes on what {name} is.

Write one to three sentences. The first says only what {name} is \
({identity}), in one short sentence, from the notes on what it is, with no \
one person's story in it. The next sum up the sections in more general \
words, a phrase or a clause for each section ("Pakistani forces arrested \
several men in the city in 2002 and 2003 and handed them to U.S. custody"), \
so a reader who stops here knows what the page tells. Don't copy a \
section's sentence, and don't list every name and date.

Use only what the notes and sections say, and keep who said what. Use the \
past tense for what has ended (a man released in 2016 "was held"). Write \
about {name}, never about "the entry", "the notes", "this record" or why \
{name} "matters". Every sentence lists the numbers of the notes it uses in its \
notes field; never write note numbers into the sentence itself."""

CONFLICTS = """You read the notes for one section of an encyclopedia entry \
about the {what} "{name}". The notes are sentences taken from news \
articles. Find the places where they contradict each other: two notes \
that give a different date, place or number for the same event happening \
to the same person or thing, so that both cannot be true (one says he was \
captured on Sept. 11, 2002, another that he was arrested in April 2003).

For each contradiction give who or what it is about, the event, what one \
side says and its notes, and what the other side says and its notes. \
Not a contradiction: an exact date and a vaguer one that fits it ("April \
2003" and "April 29, 2003"); two descriptions that can both be true \
("Saudi-born" and "Yemeni"); a count or a post that changed over time; \
two different people or events. Most sections have none: then give an \
empty list."""

NO_CONFLICT = "When two notes give different dates or numbers for the same thing, give both and cite both."
WEIGH = (
    "These notes disagree: {items}. Tell it as one event with two accounts, in "
    "one or two sentences that cite both sides, and say how many and which "
    'sources give each ("One account dates his capture to Sept. 11, 2002; '
    'three later reports, among them the Red Cross, give April 29, 2003."). '
    "Don't judge which is right in your own words, don't tell it as two "
    "events, and don't state either account as fact elsewhere in the section."
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


class LeftOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    note: int
    why: str = Field(description='"someone else", "no fact" or "repeats N"')


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    identity: str
    why_it_matters: str
    parts: List[PlanPart]
    left_out: List[LeftOut]
    opening_notes: List[int]
    conflicts: List[Conflict] = Field(default_factory=list)  # filled per part


class Conflicts(BaseModel):
    model_config = ConfigDict(extra="ignore")
    conflicts: List[Conflict]


def _outline(plan: Plan) -> str:
    return "\n".join(f"- {p.heading}: {p.covers}" for p in plan.parts)


def _counts_over_time(c: Conflict, dates: Dict[int, str]) -> bool:
    """Two counts with no year in either side, from articles of different
    months ("40 prisoners" in October 2018, "41 captives" in February): a
    count that changed, not a contradiction."""
    if re.search(r"\b(?:19|20)\d\d\b", f"{c.one_side} {c.other_side}"):
        return False
    one = {dates.get(n, "")[:7] for n in c.one_notes} - {""}
    other = {dates.get(n, "")[:7] for n in c.other_notes} - {""}
    return bool(one and other) and not one & other


def _surnames(text: str, name: str) -> set:
    """Last words of the people (two or more capitalised words) a text
    names, the page's own name left out."""
    own = {w.lower() for w in name.replace("-", " ").split()}
    last = (p.split()[-1].lower() for p in _PERSON.findall(text.replace("-", " ")))
    return {w for w in last if w not in own and w not in _NOT_PEOPLE}


# Last words of institutions and places, not people: on 21 Nov the "people"
# no page sentence named included the Periodic Review Board, the Wall Street
# Journal, the Associated Press and the United States
_NOT_PEOPLE = {
    "board", "journal", "press", "states", "court", "committee", "department",
    "agency", "cross", "council", "ministry", "office", "force", "forces",
    "army", "navy", "times", "post", "news", "service", "union", "nations",
    "bay", "camp", "commission", "base", "prison", "center", "centre",
    "administration", "government", "party", "group", "network", "herald",
    "tribune", "review", "university", "school", "airport", "street",
}  # fmt: skip


def _dated_people(text: str, name: str) -> set:
    """Surnames of the people named in the 80 characters before each year
    of ``text``, in the same sentence: whose dates the note gives."""
    out = set()
    for m in _YEAR.finditer(text):
        before = re.split(r"[.;]\s", text[max(0, m.start() - 80) : m.start()])[-1]
        out |= _surnames(before, name)
    return out


def _side(c_notes: Sequence[int], texts: Dict[int, str], name: str, dated: bool) -> set:
    pick = _dated_people if dated else _surnames
    return set().union(*(pick(texts.get(n, ""), name) for n in c_notes))


def _different_people(c: Conflict, texts: Dict[int, str], name: str) -> bool:
    """Two men, not one event told twice: the two sides name no person in
    common, or, where both give years, no person named just before a year
    on one side is named so on the other. On 21 Nov local Qwen set Paracha's
    notes against Suhayl al Sharabi's (both called "forever prisoners") and
    wrote "Sources disagree on the identity of the detainee"; next run the
    note giving al Sharabi's May 2002 capture also named Paracha, so sharing
    a name was not enough."""
    for dated in (False, True):
        one = _side(c.one_notes, texts, name, dated)
        other = _side(c.other_notes, texts, name, dated)
        if one and other and not one & other:
            return True
    return False


def _sides_agree(c: Conflict) -> bool:
    """The two sides agree: one only less exact, or adding up to the other
    (Jakarta's "10 Indonesians and a Dutch citizen" killed against 11 dead,
    listed as a conflict in four rounds running, 21-22 Nov)."""
    from src.engine.written_conflicts import _agree

    one, two = c.one_side or "", c.other_side or ""
    return _agree(f"By one, {one}, while another account says {two}") or _agree(
        f"By one, {two}, while another account says {one}"
    )


def clean_conflicts(
    conflicts: Sequence[Conflict], allowed: Any, dates: Optional[Dict[int, str]] = None,
    texts: Optional[Dict[int, str]] = None, name: str = "",
) -> List[Conflict]:  # fmt: skip
    """Conflicts with notes on both sides (among ``allowed``), the sides'
    values not fitting inside each other, not a count that changed over
    time, not about two different people, and not the same notes twice."""
    out, seen = [], set()
    for c in conflicts:
        one = [n for n in dict.fromkeys(c.one_notes) if n in allowed]
        other = [
            n for n in dict.fromkeys(c.other_notes) if n in allowed and n not in one
        ]
        c = c.model_copy(update={"one_notes": one, "other_notes": other})
        key = frozenset(one + other)
        if not one or not other or key in seen or _compatible(c.one_side, c.other_side):
            continue
        if _counts_over_time(c, dates or {}):
            continue
        if bool(re.search(r"\d", c.one_side)) != bool(re.search(r"\d", c.other_side)) or (
            re.search(r"\bwithout (?:a |specifying )?(?:date|day)|\bundated\b|no date",
                      f"{c.one_side} {c.other_side}", re.I)
        ):  # fmt: skip
            continue  # a date against none (bin al Shibh, 21 Nov): nothing disputed
        if texts and _different_people(c, texts, name):
            continue
        if _sides_agree(c):
            continue
        seen.add(key)
        out.append(c)
    return out


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
    conflicts = clean_conflicts(plan.conflicts or [], seen)
    return Plan(
        identity=plan.identity or "", why_it_matters=plan.why_it_matters or "",
        parts=parts,
        left_out=[o for o in plan.left_out or [] if o.note in known and o.note not in seen],
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
    plan: Plan, notes_text: str, sections: Sequence[Dict[str, Any]], name: str,
    what: str,
) -> List[Dict[str, str]]:  # fmt: skip
    """The opening is written last, from the written sections, so it sums
    them up instead of saying the first section's facts twice (on 21 Nov,
    written from the notes, Luna's Karachi opening listed the arrests its
    first section then listed again)."""
    lines = []
    for sec in sections:
        lines.append(f"## {sec['heading']}")
        for para in sec["paragraphs"]:
            for sent in para.get("sentences") or []:
                ns = ", ".join(str(n) for n in sent.get("notes") or [])
                lines.append(f"- {sent.get('text', '')} (notes {ns})")
    system = OPENING.format(what=what, name=name, identity=plan.identity)
    body = f"Notes on what {name} is:\n{notes_text}\n\nThe sections:\n" + "\n".join(
        lines
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def conflict_messages(
    notes_text: str, name: str, what: str, pairs: Sequence[str] = ()
) -> List[Dict[str, str]]:
    user = f"Notes for this section:\n{notes_text}"
    if pairs:
        user += (
            "\n\nPairs worth a close look (they name the same person and give "
            "different years; they may still be two different events):\n"
            + "\n".join(f"- {p}" for p in pairs)
        )
    return [
        {"role": "system", "content": CONFLICTS.format(what=what, name=name)},
        {"role": "user", "content": user},
    ]


_PERSON = re.compile(
    r"\b[A-Z][a-z]+(?:\s+(?:al|bin|ibn|el|abu|de|van)?\s*[A-Z][a-z]+)+"
)
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")


def _pairs(notes: Sequence[Any], name: str) -> List[Tuple[Any, Any, str, set, set]]:
    info = []
    for n in notes:
        people = {p for p in _PERSON.findall(n.text) if p.lower() != name.lower()}
        info.append((n, people, set(_YEAR.findall(n.text))))
    out = []
    for i, (a, pa, ya) in enumerate(info):
        for b, pb, yb in info[i + 1 :]:
            shared = pa & pb
            if shared and ya and yb and not ya & yb:
                out.append((a, b, sorted(shared, key=len)[-1], ya, yb))
    return out


def candidate_pairs(notes: Sequence[Any], name: str) -> List[str]:
    """Pairs of notes naming the same person (two or more capitalised words,
    not the page's own name) with different years in their text. On 21 Nov
    the contradiction step found bin Attash's two capture dates in one Luna
    run and not in the next, from the same notes; pointing at the pair takes
    the luck out of noticing it."""
    return [
        f"notes {a.n} and {b.n} ({who}: {', '.join(sorted(ya))} "
        f"against {', '.join(sorted(yb))})"
        for a, b, who, ya, yb in _pairs(notes, name)
    ][:6]


def cross_part_conflicts(
    plan: Plan, by_n: Dict[int, Any], name: str, what: str, ask: Any,
    report: Dict[str, Any], block: Any, dates: Dict[int, str],
) -> List[Conflict]:  # fmt: skip
    """The contradiction step for same-person pairs the plan put in two
    parts (local Qwen filed bin Attash's 2002 capture under "Arrests in
    2002" and his April 2003 arrest under "Arrests in 2003", so neither
    part's step saw both). Each conflict found moves all its notes into the
    part holding its first side, so one section tells it."""
    part_of = {n: i for i, p in enumerate(plan.parts) for n in p.notes}
    planned = [by_n[n] for n in part_of]
    cross = [
        (a, b) for a, b, *_ in _pairs(planned, name) if part_of[a.n] != part_of[b.n]
    ][:6]
    if not cross:
        return []
    mine = list({n.n: n for pair in cross for n in pair}.values())
    part = PlanPart(heading="(across parts)", covers="", notes=[n.n for n in mine])
    found = find_conflicts(
        plan, part, block(mine), name, what, ask, report, dates, mine
    )
    for c in found:
        home = plan.parts[part_of[c.one_notes[0]]]
        for n in c.notes:
            if n not in home.notes:
                plan.parts[part_of[n]].notes.remove(n)
                home.notes.append(n)
    plan.parts = [p for p in plan.parts if p.notes]
    return found


def max_parts(n_notes: int) -> int:
    return 2 if n_notes < 8 else 4 if n_notes < 40 else 6


def opening_notes(plan: Plan, limit: int = 6) -> List[int]:
    """The plan's notes on what the entity is, else the first part's first."""
    out = list(plan.opening_notes) or [p.notes[0] for p in plan.parts[:1]]
    return list(dict.fromkeys(out))[:limit]


def section_words(total: int, part: PlanPart, plan: Plan) -> int:
    used = sum(len(p.notes) for p in plan.parts) or 1
    return max(40, int(total * len(part.notes) / used))


def failed(report: Dict[str, Any], step: str, exc: Exception) -> None:
    report.setdefault("errors", []).append(f"{step}: {str(exc)[:200]}")


def find_conflicts(
    plan: Plan, part: PlanPart, notes_text: str, name: str, what: str, ask: Any,
    report: Dict[str, Any], dates: Optional[Dict[int, str]] = None,
    shown: Optional[Sequence[Any]] = None,
) -> List[Conflict]:  # fmt: skip
    """One call on one part's notes: which of them contradict each other.
    Its own step because inside the plan call Luna found none on Karachi
    (bin Attash: captured Sept. 11, 2002, or arrested April 2003) once told
    not to over-call them, while Qwen still called a vaguer date a conflict."""
    if len(part.notes) < 2:
        return []
    pairs = candidate_pairs(shown or [], name)
    report.setdefault("conflict_pairs", []).extend(pairs)
    try:
        reply = ask(conflict_messages(notes_text, name, what, pairs), Conflicts, 2048)
    except Exception as exc:
        failed(report, f"conflicts {part.heading}", exc)
        return []
    report["calls"] = report.get("calls", 0) + 1
    texts = {n.n: n.text for n in shown or []}
    return clean_conflicts(reply.conflicts or [], set(part.notes), dates, texts, name)


def restore_on_topic(plan: Plan, by_n: Dict[int, Any], name: str, what: str) -> None:
    """A note left out as "someone else" goes back in when the note tagger
    called it about the entity, or when it names the place, or the person's
    surname: on 21 Nov Luna's plan left Daniel Pearl's killing in Karachi
    out of the Karachi page in two runs of two, and Bin Atef's lawyer's
    account of him out of his page as "someone else". It joins the part
    sharing most words."""
    from src.engine.written_checks import _content

    for o in list(plan.left_out):
        note = by_n.get(o.note)
        if note is None or not (o.why or "").lower().startswith("someone"):
            continue
        low = note.text.lower()
        surname = name.split()[-1].lower() if name.split() else ""
        named = (what.startswith("place") and name.lower() in low) or (
            what.startswith("person") and len(surname) > 2
            and re.search(rf"\b{re.escape(surname)}\b", low) is not None
        )  # fmt: skip
        if note.tag != "about" and not named:
            continue
        words = _content(note.text)
        part = max(plan.parts, key=lambda p: sum(
            len(words & _content(by_n[n].text)) for n in p.notes if n in by_n))  # fmt: skip
        part.notes.append(o.note)
        plan.left_out.remove(o)


def restore_repeats(plan: Plan, by_n: Dict[int, Any], name: str) -> None:
    """A note left out as "repeats N" that holds a name, number or date note
    N lacks goes back in, after N (on 21 Nov Luna left Bin Atef's 2009
    clearance out as repeating a note that never gave it)."""
    from src.engine.written_checks import _content, missing_details

    for o in list(plan.left_out):
        m = re.search(r"repeats\D*(\d+)", o.why or "")
        n = int(m.group(1)) if m else 0
        part = next((p for p in plan.parts if n in p.notes), None)
        if o.note not in by_n or n not in by_n:
            continue
        if part is None and any(x.note == n for x in plan.left_out):
            # Each left out as repeating the other (Luna, Bin Atef's notes 12
            # and 13, 21 Nov): neither told. This one goes in where it fits.
            words = _content(by_n[o.note].text)
            best = max(plan.parts, key=lambda p: sum(
                len(words & _content(by_n[k].text)) for k in p.notes if k in by_n))  # fmt: skip
            best.notes.append(o.note)
            plan.left_out.remove(o)
            continue
        if part is None:
            continue
        extra = _content(by_n[o.note].text) - _content(by_n[n].text)
        if missing_details(by_n[o.note].text, [by_n[n].text], name) or len(extra) >= 3:
            part.notes.insert(part.notes.index(n) + 1, o.note)
            plan.left_out.remove(o)


def life_order(parts: List[PlanPart], by_n: Dict[int, Any]) -> List[PlanPart]:
    """A person's parts sorted by the middle one of their notes' earliest
    years (so one note looking back on a capture doesn't pull a section
    forward); a part naming no year keeps its place after the part before. On 21 Nov two Luna
    plans of Bin Atef's page made a last section of his 2007 prison profile
    and his lawyer's account, after his 2016 transfer and the 2017 ruling."""
    keys, last = [], 0
    for p in parts:
        years = sorted(
            min(int(y) for y in found)
            for n in p.notes
            if n in by_n and (found := _YEAR.findall(by_n[n].text))
        )
        last = years[len(years) // 2] if years else last
        keys.append(last)
    order = sorted(range(len(parts)), key=lambda i: (keys[i], i))
    return [parts[i] for i in order]


def restore_orphans(plan: Plan, by_n: Dict[int, Any]) -> None:
    """A note the plan neither put in a part nor left out goes into the part
    sharing most words with it. On 21 Nov one Luna plan of the Karachi page
    dropped Suhayl al Sharabi's capture this way, and as it was in no part
    and not in the list of notes left out, no later stage noticed."""
    from src.engine.written_checks import _content

    placed = {n for p in plan.parts for n in p.notes} | {o.note for o in plan.left_out}
    placed |= set(opening_notes(plan))  # the opening tells them (an "Overview" part's)
    for n, note in by_n.items():
        if n in placed or not plan.parts:
            continue
        words = _content(note.text)
        best = max(plan.parts, key=lambda p: sum(
            len(words & _content(by_n[k].text)) for k in p.notes if k in by_n))  # fmt: skip
        best.notes.append(n)


def file_opening_details(plan: Plan, by_n: Dict[int, Any], name: str) -> List[int]:
    """An opening note that holds two or more names, places or numbers the
    plan's identity and the entity's own name lack also goes into the body
    part sharing most words with it; the numbers filed. The opening says
    only what the entity is, so the rest of such a note was written nowhere:
    on 23 Nov (round 41c) Speer's note "a U.S. Army soldier ... from
    Albuquerque, N.M., and a father of two, including a son named Tanner"
    gave the lead one sentence, and the hometown and family left the page."""
    from src.engine.written_checks import _content, _numbers

    placed = {n for p in plan.parts for n in p.notes}
    own = {w.lower() for w in re.findall(r"[\w'’-]+", f"{name} {plan.identity}")}
    filed = []
    for n in opening_notes(plan):
        note = by_n.get(n)
        if note is None or n in placed or not plan.parts:
            continue
        caps = {w.lower() for w in re.findall(r"\b[A-Z][\w'’-]+", note.text)[1:]}
        extra = (caps | _numbers(note.text)) - own
        if len(extra) < 2:
            continue
        # Shared words per note, not in all: summed, the biggest part won
        # (four Speer draws: his hometown filed under the Khadr proceedings,
        # not his service); first in its part, as background
        words = _content(note.text)
        best = max(plan.parts, key=lambda p: sum(
            len(words & _content(by_n[k].text)) for k in p.notes if k in by_n
        ) / max(len(p.notes), 1))  # fmt: skip
        best.notes.insert(0, n)
        filed.append(n)
    return filed


from src.engine.written_groups import (  # noqa: E402
    GROUPS_NOTE,
    group_notes,
    groups_block,
)


def write_planned(
    notes: Sequence[Any], name: str, what: str, headings: Sequence[Tuple[str, str]],
    words: int, ask: Any, report: Dict[str, Any], block: Any, page_model: Any,
    section_model: Any, sentence_model: Any,
) -> Optional[Any]:  # fmt: skip
    """A ``page_model`` (``WPage``) written in stages: the plan, then for each
    part a check for contradicting notes and the section itself, then the
    opening from the written sections. None when the plan call fails.
    ``block`` prints notes; the models are passed in from ``written_page`` so
    this module needs no import of it."""
    plan = make_plan(notes, name, what, headings, ask, report, block)
    if plan is None:
        return None
    return write_from_plan(
        plan, notes, name, what, words, ask, report, block, page_model,
        section_model, sentence_model,
    )  # fmt: skip


def make_plan(
    notes: Sequence[Any], name: str, what: str, headings: Sequence[Tuple[str, str]],
    ask: Any, report: Dict[str, Any], block: Any,
) -> Optional[Plan]:  # fmt: skip
    """The plan with its notes filed, restored and ordered and its conflicts
    found; None when the plan call fails or has no parts."""
    by_n = {n.n: n for n in notes}
    k = max_parts(len(notes))
    try:
        raw = ask(plan_messages(block(notes), name, what, headings, k), Plan, 4096)
    except Exception as exc:
        failed(report, "plan", exc)
        return None
    report["calls"] = report.get("calls", 0) + 1
    plan = clean_plan(raw, list(by_n), k)
    if not plan.parts:
        report["plan"] = {"identity": plan.identity, "parts": []}
        return None
    from src.engine.written_file import file_notes

    file_notes(plan, notes, name, what, ask, report, block)
    before = {n for p in plan.parts for n in p.notes}
    restore_repeats(plan, by_n, name)
    restore_on_topic(plan, by_n, name, what)
    restore_orphans(plan, by_n)
    # Off unless asked: in 8 Speer draws the plan filed the note itself 5
    # times, and this filed it 3 times, 2 into the wrong part (23 Nov)
    opening_filed = (
        file_opening_details(plan, by_n, name) if report.get("opening_details") else []
    )
    restored = sorted({n for p in plan.parts for n in p.notes} - before)
    for p in plan.parts:
        p.notes = list(dict.fromkeys(p.notes))
    if what.startswith("person"):
        plan.parts = life_order(plan.parts, by_n)
    if report.get("events", False):  # one event's notes in one part
        from src.engine.written_events import gather_events

        gather_events(plan, by_n, name, what, ask, report)
    dates = {n.n: n.date for n in notes}
    for part in plan.parts:
        mine = [by_n[n] for n in part.notes]
        plan.conflicts += find_conflicts(
            plan, part, block(mine), name, what, ask, report, dates, mine
        )
    plan.conflicts += cross_part_conflicts(
        plan, by_n, name, what, ask, report, block, dates
    )
    texts = {n.n: n.text for n in notes}
    plan.conflicts = clean_conflicts(plan.conflicts, set(by_n), dates, texts, name)
    from src.engine.written_pick import confirm_conflicts

    plan.conflicts = confirm_conflicts(
        plan.conflicts, by_n, name, what, ask, report, block
    )
    report["plan"] = {
        "identity": plan.identity, "why": plan.why_it_matters,
        "parts": [{"heading": p.heading, "notes": p.notes} for p in plan.parts],
        "opening_notes": opening_notes(plan), "restored": restored,
        "opening_filed": opening_filed,
        "left_out": [o.model_dump() for o in plan.left_out],
        "conflicts": [c.model_dump() for c in plan.conflicts],
    }  # fmt: skip
    return plan


def write_from_plan(
    plan: Plan, notes: Sequence[Any], name: str, what: str, words: int, ask: Any,
    report: Dict[str, Any], block: Any, page_model: Any, section_model: Any,
    sentence_model: Any,
) -> Any:  # fmt: skip
    """The sections, each from its grouped notes, then the opening."""
    by_n = {n.n: n for n in notes}
    sections = []
    for part in plan.parts:
        groups = group_notes([by_n[n] for n in part.notes], name)
        part.notes = [n.n for g in groups for n in g]  # the fill keeps this order
        msgs = section_messages(
            plan, part, groups_block(groups, block), name, what,
            section_words(words, part, plan),
        )  # fmt: skip
        msgs[-1]["content"] = f"{GROUPS_NOTE}\n\n{msgs[-1]['content']}"
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
            opening_messages(plan, block(shown), sections, name, what),
            sentence_model, 2048,
        )  # fmt: skip
        report["calls"] += 1
        first = [s.model_dump() for s in opening.sentences]
    except Exception as exc:
        failed(report, "opening", exc)
        first = []
    return page_model.model_validate({"opening": first, "sections": sections})
