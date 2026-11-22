"""A last stage over the whole finished page: facts told twice in different
places, and sentences that give different accounts of one thing without
saying so.

Why: every earlier stage looks at one section or one plan part at a time,
and on 21 Nov (round 19) a full read of the long pages found the CIA page
giving Khalid Sheik Mohammed's 183 waterboardings four times, and Majid
Khan "not subjected to waterboarding" in one section and waterboarded twice
in another. Round 20 asked one call to read the whole page and list such
faults: over 31 pages it listed almost nothing (not the 183, not
Rishikof's fine waived, then not dismissed, then struck down), three of
its four "repeat" drops took a real fact off the page (Bin al Shibh held
at the ISI facility, Khadr's abuse allegations), and one of its two
rewrites merged when Rishikof was fired with why.

So it is now split into small steps:

1. Code picks candidate pairs: two kept sentences sharing most of their
   words, or a number that is not a year and two other words.
2. One call judges each pair: "same" (the second tells again a fact the
   first tells), "conflict" (two accounts of one thing, neither saying so)
   or "different" (different events, people, days, or a count that changed).
3. A "same" second sentence is dropped only when code finds nothing in it
   that the first lacks (words, numbers, people); otherwise it is rewritten
   once to keep only what it adds. A "conflict" second sentence is
   rewritten once from its own notes as another account ("Another account
   dates ..."); the first stays as it is. Every rewrite goes through the
   usual checks, and when one fails the old sentence stays: a repeat is
   better than a lost fact.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.engine.written_page import Ask, Note, Written

MAX_PAIRS = 24

JUDGE = """You check pairs of sentences from one encyclopedia entry about \
the {what} "{name}". For each pair say how sentence B stands to sentence A:

- "same": B tells again a fact A already tells (it may add other facts \
too).
- "conflict": A and B give different accounts of the same thing (the same \
event, person and question: a date, a count on the same day, whether \
something happened), and neither says that accounts differ.
- "different": anything else: different events, people, places or days; \
a count or a state that changed over time; a later decision; one sentence \
giving more detail than the other without disagreeing.

Give the pair's number, B's first three words copied, and the verdict. \
When unsure, say "different"."""


class PairVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pair: int
    starts_with: str = Field(default="", description="B's first three words, copied")
    verdict: str = Field(description='"same", "conflict" or "different"')


class Verdicts(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pairs: List[PairVerdict]


# A sentence that already says the accounts differ
_TOLD = re.compile(
    r"\bdiffer|\bdisagree|\banother (?:account|report)|\bother accounts?\b|"
    r"\baccounts\b",
    re.I,
)
_YEARLIKE = re.compile(r"^(?:19|20)\d\d$")


def _facts(w: Written, name: str) -> Tuple[set, set, set]:
    from src.engine.written_checks import _content, _numbers
    from src.engine.written_plan import _surnames

    return _content(w.text), _numbers(w.text), _surnames(w.text, name)


def candidate_pairs(kept: Sequence[Written], name: str) -> List[Tuple[int, int]]:
    """Index pairs (earlier, later) of section sentences worth judging, most
    alike first, at most ``MAX_PAIRS``."""
    facts = [_facts(w, name) for w in kept]
    scored = []
    for j, b in enumerate(kept):
        if not b.section:
            continue
        for i in range(j):
            if not kept[i].section:
                continue
            (wa, na, pa), (wb, nb, pb) = facts[i], facts[j]
            shared = wa & wb
            small = min(len(wa), len(wb)) or 1
            counts = {n for n in na & nb if not _YEARLIKE.match(n) and len(n) > 1}
            years = {n for n in na & nb if _YEARLIKE.match(n)}
            # A person or most words plus a year: round 21 never paired Bin
            # Atef's "36" with "mid-30s", nor Karachi's April 2003 arrests
            # told twice
            dated = years and ((pa & pb) or len(shared) >= 4)
            if (
                (len(shared) >= 4 and len(shared) >= 0.5 * small)
                or (counts and len(shared) >= 2)
                or (dated and len(shared) >= 3)
            ):
                # Same-section pairs first: a long page has more than
                # MAX_PAIRS, and a repeat inside one section reads worst
                same = kept[i].section == kept[j].section
                scored.append((not same, -len(shared) / small, i, j))
    scored.sort()
    return [(i, j) for _, _, i, j in scored[:MAX_PAIRS]]


def _adds_nothing(b: Written, a: Written, name: str) -> bool:
    """Every word, number and person of ``b`` is in ``a``."""
    (wa, na, pa), (wb, nb, pb) = _facts(a, name), _facts(b, name)
    return bool(wb) and wb <= wa and nb <= na and pb <= pa


def _marks(text: str) -> set:
    """Numbers, dates and capitalised words past the first word."""
    from src.engine.written_checks import _numbers

    caps = {w.lower() for w in re.findall(r"\b[A-Z][\w'’-]+", text)[1:]}
    return _numbers(text) | caps


def _lost(old: Written, first: Written, new: Written) -> bool:
    """The trim of ``old`` dropped a name, date or number the first sentence
    lacks. Round 21: a Rishikof trim kept one clause and lost the summons to
    William Castle's office, the clearances and the March 2018 filing; two
    Speer trims lost Capt. Justin Swick; three lost the year."""
    return bool((_marks(old.text) - _marks(first.text)) - _marks(new.text))


def judge_messages(
    kept: Sequence[Written], pairs: Sequence[Tuple[int, int]], name: str, what: str
) -> List[Dict[str, str]]:
    body = "\n\n".join(
        f"Pair {k}:\nA: {kept[i].text}\nB: {kept[j].text}"
        for k, (i, j) in enumerate(pairs, 1)
    )
    return [
        {"role": "system", "content": JUDGE.format(what=what, name=name)},
        {"role": "user", "content": body},
    ]


def _trim(a: Written) -> str:
    return (
        "check: it tells again what an earlier sentence of the entry tells "
        f'("{a.text}"). Keep only what it adds to that sentence, as a '
        "sentence that reads well on its own; give it up if it adds nothing"
    )


def _other_account(a: Written) -> str:
    return (
        "check: it gives a different account from an earlier sentence of "
        f'the entry ("{a.text}") without saying so. Tell it as another '
        "account of the same thing, with its source and date as its notes "
        'give them ("Another account dates ... to ..."); if its notes show '
        "it is about a different thing or time, make that plain instead"
    )


def review_page(
    written: List[Written], notes: Sequence[Note], name: str, what: str,
    ask: Ask, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with repeats across the page trimmed or dropped and
    unflagged disagreements told as another account."""
    from src.engine.written_checks import _first_words
    from src.engine.written_conflicts import lost_or_added
    from src.engine.written_revise import revise

    kept = [w for w in written if not w.dropped]
    pairs = candidate_pairs(kept, name)
    stats = {"pairs": len(pairs), "same": 0, "conflict": 0, "dropped": 0}
    report["review"] = stats
    if not pairs:
        return written
    try:
        reply = ask(judge_messages(kept, pairs, name, what), Verdicts, 2048)
    except Exception as exc:
        report.setdefault("review_errors", []).append(str(exc)[:200])
        return written
    report["calls"] = report.get("calls", 0) + 1
    marked: Dict[int, Tuple[str, Written]] = {}  # id -> (verdict, first sentence)
    for v in reply.pairs or []:
        if not 1 <= v.pair <= len(pairs):
            continue
        a, b = (kept[k] for k in pairs[v.pair - 1])
        if v.starts_with and _first_words(v.starts_with, 2) != _first_words(b.text, 2):
            continue
        verdict = (v.verdict or "").strip().lower()
        if a.dropped or b.dropped or id(b) in marked:
            continue
        if verdict == "same":
            stats["same"] += 1
            (_, _, people_a), (_, _, people_b) = _facts(a, name), _facts(b, name)
            if people_b - people_a:
                # Round 22: a Karachi trim kept "handed al Baluchi and bin
                # Attash over to the CIA" and lost al Baluchi's arrest, which
                # the first sentence (about bin Attash) never told
                continue
            if _adds_nothing(b, a, name):
                b.dropped = "repeats an earlier sentence of the page"
                stats["dropped"] += 1
            else:
                b.dropped, marked[id(b)] = _trim(a), ("same", a)
        elif verdict == "conflict":
            stats["conflict"] += 1
            if any(_TOLD.search(w.text) for w in (a, b)):
                continue  # one of them already says so
            b.dropped, marked[id(b)] = _other_account(a), ("conflict", a)
    if not marked:
        return written
    out = revise(
        written, notes, name, ask, report,
        pick=lambda w: id(w) in marked, tag="reviewed",
    )  # fmt: skip
    # A rewrite that failed, or a sentence given up, stays as it was
    restored = 0
    for i, w in enumerate(out):
        if id(w) not in marked:
            continue
        nxt = out[i + 1] if i + 1 < len(out) else None
        ok = nxt is not None and nxt.revised == w.text and not nxt.dropped
        kind, first = marked[id(w)]
        if ok and kind == "conflict" and lost_or_added(w.text, nxt.text):
            nxt.dropped, ok = "rewrite lost a source or added a day", False
        if ok and kind == "same" and _lost(w, first, nxt):
            nxt.dropped, ok = "trim lost a name, date or number", False
        if not ok:
            w.dropped = ""
            restored += 1
    stats["restored"] = restored
    return out
