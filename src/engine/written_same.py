"""Notes that tell one and the same fact, joined into one note before the
page is planned.

Why: on 22 Nov the held-out readers (16 pages never used while tuning, two
readers each) counted 27 to 35 repeats at every round since 26, and ten of
them named the same cause: one filing, decision or figure told twice
because several articles reported it. Each round's repeat passes worked
on the written sentences and helped only the tuning pages. By the stages
rule this works one step earlier, on the notes: code pairs notes that
share names, numbers or most of their words; one model call per batch of
pairs says whether a pair tells the same fact; code joins a pair only when
one note holds every number, name and month of the other and nearly all
its words. The joined note keeps the fuller text and every citation of
both, so the writer sees the fact once, with all its sources.
"""

import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

SAME = """You compare pairs of notes about the {what} "{name}". The notes \
are sentences from news articles; each starts with its article's date in \
brackets.

For each numbered pair, say in a few words what each note tells, then \
answer:
- same: both tell one and the same fact (the same act, event, decision, \
statement or figure, at the same time). One may word it differently or give \
less detail.
- different: they tell different facts, or the same kind of thing at \
different times (two counts on two days, two hearings, two releases), or \
they disagree."""

BATCH = 10
MAX_PAIRS = 80
WORD_SHARE = 0.8  # of the dropped note's words the kept note must have


class PairSame(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pair: int = Field(description="The pair's number")
    tells: str = Field(default="", description="What each note tells, in a few words")
    verdict: str = Field(description="same or different")


class PairsSame(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pairs: List[PairSame]


_MONTH = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b\.?")


def _facts(text: str, name: str) -> Tuple[set, set, set]:
    from src.engine.written_checks import _content, _numbers
    from src.engine.written_plan import _surnames

    months = {m.group(0)[:3].lower() for m in _MONTH.finditer(text)}
    return _content(text), _numbers(text) | months, _surnames(text, name)


def note_pairs(notes: Sequence[Any], name: str) -> List[Tuple[int, int]]:
    """Index pairs of notes worth asking about: a shared count, a shared
    person and year, or most words shared; most alike first."""
    facts = [_facts(n.text, name) for n in notes]
    scored = []
    for j in range(len(notes)):
        for i in range(j):
            (wa, na, pa), (wb, nb, pb) = facts[i], facts[j]
            shared = wa & wb
            small = min(len(wa), len(wb)) or 1
            counts = {
                x
                for x in na & nb
                if x.isdigit() and not re.fullmatch(r"(?:19|20)\d\d", x)
            }
            if (
                (len(shared) >= 4 and len(shared) >= 0.5 * small)
                or (counts and len(shared) >= 2)
                or ((pa & pb) and (na & nb) and len(shared) >= 3)
            ):
                scored.append((-len(shared) / small, i, j))
    scored.sort()
    return [(i, j) for _, i, j in scored[:MAX_PAIRS]]


def holds(big: str, small: str, name: str) -> bool:
    """``big`` has every number, month and person of ``small`` and
    ``WORD_SHARE`` of its words: joining loses nothing."""
    (wb, nb, pb), (ws, ns, ps) = _facts(big, name), _facts(small, name)
    return ns <= nb and ps <= pb and len(ws & wb) >= WORD_SHARE * max(len(ws), 1)


def _messages(
    pairs: Sequence[Tuple[Any, Any]], name: str, what: str
) -> List[Dict[str, str]]:
    body = "\n\n".join(
        f"Pair {k}:\n  A [{a.date or 'no date'}] {a.text}\n  B [{b.date or 'no date'}] {b.text}"
        for k, (a, b) in enumerate(pairs, 1)
    )
    return [
        {"role": "system", "content": SAME.format(what=what, name=name)},
        {"role": "user", "content": body},
    ]


def join_same(
    notes: List[Any], name: str, what: str, ask: Callable[..., Any],
    report: Dict[str, Any],
) -> List[Any]:  # fmt: skip
    """``notes`` with each pair the model calls the same, and that
    ``holds()`` allows, joined into the fuller note; renumbered. Each join
    is listed under ``report["same_joined"]`` as [kept, dropped]."""
    from src.engine.written_page import Note, _numbered

    pairs = note_pairs(notes, name)
    same: List[Tuple[int, int]] = []
    for start in range(0, len(pairs), BATCH):
        batch = pairs[start : start + BATCH]
        try:
            reply = ask(
                _messages([(notes[i], notes[j]) for i, j in batch], name, what),
                PairsSame,
                4096,
            )
        except Exception as exc:
            report.setdefault("errors", []).append(f"same: {str(exc)[:200]}")
            continue
        report["calls"] = report.get("calls", 0) + 1
        for p in reply.pairs or []:
            if 1 <= p.pair <= len(batch) and str(p.verdict).strip().lower() == "same":
                same.append(batch[p.pair - 1])
    into: Dict[int, int] = {}  # dropped index -> the index that holds it

    def home(i: int) -> int:
        while i in into:
            i = into[i]
        return i

    out = list(notes)
    log: List[List[str]] = report.setdefault("same_joined", [])
    for i, j in same:
        a, b = home(i), home(j)
        if a == b:
            continue
        big, small = (a, b) if len(out[a].text) >= len(out[b].text) else (b, a)
        if not holds(out[big].text, out[small].text, name):
            big, small = small, big
            if not holds(out[big].text, out[small].text, name):
                continue
        k, d = out[big], out[small]
        # the kept text's own date: its "on Monday" and "this week" are its
        # article's
        out[big] = Note(
            k.n, k.key, k.text, list(dict.fromkeys(k.articles + d.articles)),
            k.date, k.heading, k.tag,
            list(dict.fromkeys((k.keys or [k.key]) + (d.keys or [d.key]))),
        )  # fmt: skip
        into[small] = big
        log.append([k.text, d.text])
    return _numbered([n for i, n in enumerate(out) if i not in into])
