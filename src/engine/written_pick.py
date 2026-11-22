"""Two small model calls for the written page: one confirming the plan's
disagreements before anything is written, and one choosing each section
from the drafts.

Why the first: in round 26 (22 Nov) most of the worst sentences were "accounts
differ" sentences the plan had asked for, where the two accounts did not
disagree: hunger-strike counts on April 1 and April 21, staff numbers in
April and August, three Americans among 80 wounded, a recess that began after
30 minutes and lasted 35, one book described twice. Code can't tell these
from a real dispute (bin Attash captured in September 2002 or arrested in
April 2003), so one call reads each planned disagreement with its notes.

Why the second: rounds 25 and 26 chose each section by a code score, and
two rounds of new terms took agreement with a careful reader from 31 of 44
picks to 22 of 37. What decides a pick (a date lost, "that month" pointing at
the wrong month, the sentence before the plea told after it) needs the drafts
read side by side, so one call per section reads them and names the best.
The code score stays as the fallback when the call fails.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

CONFIRM = """You check disagreements found in the notes for an encyclopedia \
entry about the {what} "{name}". Each item gives one account and another, with \
the notes behind each (the date in brackets is the article's date). Say for \
each whether the two accounts really contradict each other about one single \
fact:

- "contradict": both cannot be true: two different dates for the same \
one-time event, two different counts of the same thing at the same time, one \
saying something happened and the other that it did not.
- "compatible": both can be true: counts or states on different days or \
months; a part and the whole (three Americans among 80 wounded; 10 and 1 \
making 11); two measures of different things (when a recess began and how \
long it lasted); the same thing described in other words; one account only \
less exact than the other.

Give the item number, the single fact in a few words, then the verdict. When \
unsure, say "compatible"."""


class ConflictVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    item: int
    fact: str = Field(default="", description="The one fact, in a few words")
    verdict: str = Field(description='"contradict" or "compatible"')


class ConflictVerdicts(BaseModel):
    model_config = ConfigDict(extra="ignore")
    items: List[ConflictVerdict]


def confirm_conflicts(
    conflicts: Sequence[Any], by_n: Dict[int, Any], name: str, what: str,
    ask: Any, report: Dict[str, Any], block: Callable[[Sequence[Any]], str],
) -> List[Any]:  # fmt: skip
    """The planned conflicts one call calls real contradictions; all of them
    when the call fails."""
    if not conflicts:
        return list(conflicts)
    body = "\n\n".join(
        f"Item {k}: {c.about}\nOne account: {c.one_side}\n"
        f"{block([by_n[n] for n in c.one_notes if n in by_n])}\n"
        f"Another account: {c.other_side}\n"
        f"{block([by_n[n] for n in c.other_notes if n in by_n])}"
        for k, c in enumerate(conflicts, 1)
    )
    msgs = [
        {"role": "system", "content": CONFIRM.format(what=what, name=name)},
        {"role": "user", "content": body},
    ]
    try:
        reply = ask(msgs, ConflictVerdicts, 1024)
    except Exception as exc:
        report.setdefault("errors", []).append(f"confirm conflicts: {str(exc)[:200]}")
        return list(conflicts)
    report["calls"] = report.get("calls", 0) + 1
    compatible = {
        v.item
        for v in reply.items or []
        if (v.verdict or "").strip().lower().startswith("compat")
    }
    report["conflicts_dropped"] = [
        c.model_dump() for k, c in enumerate(conflicts, 1) if k in compatible
    ]
    return [c for k, c in enumerate(conflicts, 1) if k not in compatible]


PICK = """You choose the best of {n} drafts of one section, "{heading}", of an \
encyclopedia entry about the {what} "{name}". The notes the section is written \
from come first; the date in brackets is the note's article's date, which is \
not always when the event happened. Judge the drafts in this order:

1. It keeps every fact of the notes: each date, name, number and detail. A \
draft that loses a fact the others keep is worse.
2. Every date is right: each event dated as its notes date it (not by an \
article's date when the note says it happened earlier), and "that month", \
"he", "the document" point to what the sentence before names.
3. It tells events in date order, a cause before what follows from it.
4. It says nothing twice.
5. It says that accounts differ only where the notes give two accounts of \
one thing.
6. It reads like an encyclopedia: plain, clear sentences.

Say in a sentence why, then give the number of the best draft."""


class Pick(BaseModel):
    model_config = ConfigDict(extra="ignore")
    why: str = Field(default="", description="One sentence: why this draft")
    best: int = Field(description="The best draft's number")


def ask_pick(
    texts: Sequence[str], heading: str, notes: Sequence[Any], name: str, what: str,
    ask: Any, report: Dict[str, Any], block: Callable[[Sequence[Any]], str],
) -> Optional[int]:  # fmt: skip
    """Index of the draft one call calls best for the section, None when the
    drafts are the same or the call fails or answers out of range."""
    if len(set(texts)) < 2:
        return None
    drafts = "\n\n".join(
        f"Draft {k}:\n{t or '(empty)'}" for k, t in enumerate(texts, 1)
    )
    msgs = [
        {"role": "system", "content": PICK.format(
            n=len(texts), heading=heading, what=what, name=name)},
        {"role": "user", "content": f"Notes:\n{block(notes)}\n\n{drafts}"},
    ]  # fmt: skip
    try:
        reply = ask(msgs, Pick, 512)
    except Exception as exc:
        report.setdefault("errors", []).append(f"pick {heading}: {str(exc)[:200]}")
        return None
    report["calls"] = report.get("calls", 0) + 1
    if not 1 <= reply.best <= len(texts):
        return None
    report.setdefault("picks", {})[heading] = {"best": reply.best - 1, "why": reply.why}
    return reply.best - 1
