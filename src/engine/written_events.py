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
call per batch says whether a pair is about one and the same event; a note
called "one" with an earlier-dated note moves, right after it, into that
note's part. Nothing is rewritten or dropped.
"""

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
When unsure, say "two"."""

BATCH = 10
MAX_PAIRS = 60


class PairEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pair: int = Field(description="The pair's number")
    events: str = Field(default="", description="What event each note is about")
    verdict: str = Field(description="one or two")


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
    pairs: Sequence[Tuple[Any, Any]], name: str, what: str
) -> List[Dict[str, str]]:
    body = "\n\n".join(
        f"Pair {k}:\n  A [{a.date or 'no date'}] {a.text}\n  B [{b.date or 'no date'}] {b.text}"
        for k, (a, b) in enumerate(pairs, 1)
    )
    return [
        {"role": "system", "content": EVENT.format(what=what, name=name)},
        {"role": "user", "content": body},
    ]


def gather_events(
    plan: Any, by_n: Dict[int, Any], name: str, what: str, ask: Any,
    report: Dict[str, Any],
) -> None:  # fmt: skip
    """Move each note the model calls about the same event as a note of
    another part next to the earlier-dated of the two, in that one's part.
    Each move under ``report["events_moved"]`` as [moved, next to, part]."""
    pairs = cross_pairs(plan.parts, by_n, name)
    report["event_pairs"] = len(pairs)
    one: List[Tuple[int, int]] = []
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start : start + BATCH]
        try:
            reply = ask(
                _messages([(by_n[a], by_n[b]) for a, b in batch], name, what),
                PairEvents,
                4096,
            )
        except Exception as exc:
            report.setdefault("errors", []).append(f"events: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for p in reply.pairs or []:
            if 1 <= p.pair <= len(batch) and str(p.verdict).strip().lower() == "one":
                one.append(batch[p.pair - 1])
    log: List[List[Any]] = report.setdefault("events_moved", [])
    moved: set = set()
    for a, b in one:
        # the earlier telling stays; an undated note counts as latest
        first, later = sorted((a, b), key=lambda n: (by_n[n].date or "9999", n))
        if later in moved or first in moved:
            continue  # each note moves once, and never onto a moved note
        home = next((p for p in plan.parts if first in p.notes), None)
        away = next((p for p in plan.parts if later in p.notes), None)
        if home is None or away is None or home is away:
            continue
        away.notes.remove(later)
        home.notes.insert(home.notes.index(first) + 1, later)
        moved.add(later)
        log.append([later, first, home.heading])
    plan.parts = [p for p in plan.parts if p.notes]
