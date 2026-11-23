"""Notes about one event gathered into one part of the plan, so the page
tells the event in one place.

Why: on 23 Nov the held-out readers still counted 27 to 37 repeats at every
round from 26 to 40, and the tuning pages' readers named where they come
from: one event told in two or three sections, because several articles
reported it and the plan filed those articles' notes under different parts
(Rishikof's firing on 5 February 2018 told in three sections, the Baker
contempt ruling in three). Joining notes (rounds 37-38) and trimming or
folding the finished sentences (rounds 39-40) work on pairs of tellings of
one fact; most repeats are tellings of one event that each say something
else too, and each carries its source ("a December 2017 report said"), so
those steps rightly leave them alone. A section writer already joins
related notes it sees together, and each part's conflicts are looked for
within the part; so this step makes sure it sees them together.

By the stages rule: code pairs notes filed under different parts that share
a person and several words, or most of their words (``note_pairs()``); one
call per batch says whether a pair is about one and the same event and
which of the two sections is about that event; the other note moves, right
after its pair, into that section. Nothing is rewritten or dropped.

Round 41's first version moved the later-dated note next to the earlier
one; on 14 tuning pages 4 of 7 judged moves were right (Rishikof) and 3 put
a note in the wrong section: Speer's who-he-was note and his death went
into "Plea and sentencing" because the plea note was dated earlier and
named the firefight, and Bin Atef's summary note (cleared in 2009, then
resettled) went into the release part and took the clearance with it. So
the model now names the section, and a note that says who or what an
entity is never moves, nor is it a home: it tells more than one event.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

EVENT = """You check the plan of an encyclopedia entry about the {what} \
"{name}". Each numbered pair below is two notes (sentences from news \
articles, each with its article's date in brackets) that the plan put in \
different sections.

For each pair, say in a few words what event each note is about, then \
answer:
- one: both are about one and the same event or act (the same firing, \
ruling, capture, hearing, transfer, statement), even when one reports it \
later or adds its reasons, its outcome or reactions to it.
- two: different events, or the same kind of thing at different times (two \
hearings, two rulings, two counts on different days), or one note only \
mentions the other's event in passing as background to something else.
When unsure, say "two". For "one", also say which note's section is \
about that event itself ("A" or "B"); the other note will join it. A \
section that only mentions the event as background to something later \
(a plea that names the attack, a trial that names the capture) is not \
about it. If neither section is about it, give "". And say for each note \
whether it is mainly about that event ("yes"), or mainly about something \
else and only names the event ("no": a note on a whole program, on a \
plea, on other lawyers' resignations that mentions the firing).

Name the pair, then events, a_mainly, b_mainly, verdict and home."""

BATCH = 10
MAX_PAIRS = 60


class PairEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pair: int = Field(description="The pair's number")
    events: str = Field(default="", description="What event each note is about")
    a_mainly: str = Field(
        default="", description="yes: note A is mainly about that event"
    )
    b_mainly: str = Field(
        default="", description="yes: note B is mainly about that event"
    )
    verdict: str = Field(description="one or two")
    home: str = Field(
        default="", description='"A", "B" or "": the section about that event'
    )


class PairEvents(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pairs: List[PairEvent]


def cross_pairs(
    parts: Sequence[Any], by_n: Dict[int, Any], name: str
) -> List[Tuple[int, int]]:
    """Note-number pairs from two different parts, most alike first, at most
    ``MAX_PAIRS``."""
    from src.engine.written_same import note_pairs

    part_of = {n: i for i, p in enumerate(parts) for n in p.notes if n in by_n}
    order = list(part_of)
    notes = [by_n[n] for n in order]
    pairs = [(order[i], order[j]) for i, j in note_pairs(notes, name, limit=0)]
    return [(a, b) for a, b in pairs if part_of[a] != part_of[b]][:MAX_PAIRS]


def _messages(
    pairs: Sequence[Tuple[Any, str, Any, str]], name: str, what: str
) -> List[Dict[str, str]]:
    body = "\n\n".join(
        f'Pair {k}:\n  A (section "{ha}") [{a.date or "no date"}] {a.text}\n'
        f'  B (section "{hb}") [{b.date or "no date"}] {b.text}'
        for k, (a, ha, b, hb) in enumerate(pairs, 1)
    )
    return [
        {"role": "system", "content": EVENT.format(what=what, name=name)},
        {"role": "user", "content": body},
    ]


# "X is a Yemeni national ...", "He was a U.S. Army soldier ...": a note on
# who or what the entity is
_IDENTITY = re.compile(
    r"^\W*(?:[A-Z][\w.'’-]*\s+){0,6}(?:is|was|are|were)\s+(?:a|an|the|one)\b"
)


def is_identity(note: Any, plan: Any) -> bool:
    """A note the plan gives the opening, or one that opens by saying who or
    what someone is."""
    return note.n in (plan.opening_notes or []) or bool(_IDENTITY.match(note.text))


def gather_events(
    plan: Any, by_n: Dict[int, Any], name: str, what: str, ask: Any,
    report: Dict[str, Any],
) -> None:  # fmt: skip
    """Move each note the model calls about the same event as a note of
    another part into the part the model says is about that event, right
    after its pair. Each move under ``report["events_moved"]`` as [moved
    note's text, its pair's text, the part it joined]."""
    part_of = {n: p for p in plan.parts for n in p.notes}
    pairs = [
        (a, b) for a, b in cross_pairs(plan.parts, by_n, name)
        if not is_identity(by_n[a], plan) and not is_identity(by_n[b], plan)
    ]  # fmt: skip
    report["event_pairs"] = len(pairs)
    one: List[Tuple[int, int]] = []  # (home note, note that moves)
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start : start + BATCH]
        shown = [
            (by_n[a], part_of[a].heading, by_n[b], part_of[b].heading) for a, b in batch
        ]
        try:
            reply = ask(_messages(shown, name, what), PairEvents, 4096)
        except Exception as exc:
            report.setdefault("errors", []).append(f"events: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for p in reply.pairs or []:
            if not 1 <= p.pair <= len(batch) or str(p.verdict).strip().lower() != "one":
                continue
            a, b = batch[p.pair - 1]
            home = str(p.home or "").strip().strip("\"'").upper()
            # the note that moves must be mainly about the event (round 41b:
            # the CIA program note, a note on the Cole lawyers and Khadr's
            # plea each moved for naming it)
            mainly = {"A": str(p.b_mainly), "B": str(p.a_mainly)}.get(home, "")
            if mainly.strip().lower().startswith("y"):
                one.append((a, b) if home == "A" else (b, a))
    log: List[List[Any]] = report.setdefault("events_moved", [])
    moved: set = set()
    anchors: set = set()
    for stay, go in one:
        if go in moved or stay in moved or go in anchors:
            # each note moves once, never onto a moved note, and a note
            # another has joined stays (round 41b: Mattis's affidavit moved
            # away from the incidents note that had joined it)
            continue
        home = next((p for p in plan.parts if stay in p.notes), None)
        away = next((p for p in plan.parts if go in p.notes), None)
        if home is None or away is None or home is away:
            continue
        away.notes.remove(go)
        home.notes.insert(home.notes.index(stay) + 1, go)
        moved.add(go)
        anchors.add(stay)
        log.append([by_n[go].text, by_n[stay].text, home.heading])
    plan.parts = [p for p in plan.parts if p.notes]
