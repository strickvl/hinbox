"""Give a dropped sentence of a written page one chance to be rewritten.

Why: on 21 Nov the checks dropped whole planned sections because their one
or two sentences each had one bad detail (an exact day the notes don't
give, "Pakistani" for a man the notes don't call that). The rest of each
sentence was right. So the sentences the checks drop, with what each check
found, go to one call (8 at a time) that rewrites each to say only what its
notes say, or gives it up. The rewrites go through the same code check and
model check; one still failing is dropped for good. Only one round.

The rewritten sentence is put right after the one it replaces, with
``revised`` set to that sentence's text, so the page keeps its order and
the record shows both.
"""

from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.engine.written_page import (
    CHECK_BATCH,
    CHECK_NOTE_CHARS,
    Ask,
    Note,
    Written,
    _first_words,
    code_check,
    model_check,
)

REVISE = """You fix sentences of an encyclopedia entry about "{name}". \
Each sentence below was taken out because a check found a problem with it, \
given after it, against the notes it was written from (sentences from news \
articles, each with its article's date).

Rewrite each sentence so it says only what its notes say: take out or \
correct the detail the check named, keep the rest, and keep its place in \
the entry (it must still read well after the sentence before it). Keep \
who said what. Never add a detail the notes don't give. If nothing in the \
sentence can be kept, give an empty text. Give only the sentence itself, \
with no note numbers in it."""


class Revision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int
    starts_with: str = Field(description="The old sentence's first three words, copied")
    text: str = Field(description="The rewritten sentence, or '' to give it up")


class Revisions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    revisions: List[Revision]


def revisable(w: Written) -> bool:
    return bool(w.notes) and (
        w.dropped.startswith("not in its notes") or w.dropped.startswith("check")
    )


def revise_messages(
    batch: Sequence[Written], notes: Sequence[Note], name: str
) -> List[Dict[str, str]]:
    by_n = {n.n: n for n in notes}
    parts = []
    for i, w in enumerate(batch, 1):
        lines = "\n".join(
            f"  - [{by_n[n].date or 'no date'}] {by_n[n].text[:CHECK_NOTE_CHARS]}"
            for n in w.notes
            if n in by_n
        )
        problem = w.dropped.removeprefix("check: ")
        parts.append(f"Sentence {i}: {w.text}\nProblem: {problem}\nIts notes:\n{lines}")
    return [
        {"role": "system", "content": REVISE.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def revise(
    written: List[Written], notes: Sequence[Note], name: str, ask: Ask,
    report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with a rewrite after each dropped sentence the model could
    fix; the rewrites are checked like any sentence."""
    todo = [w for w in written if revisable(w) and not w.summary]
    new: Dict[int, Written] = {}
    for start in range(0, len(todo), CHECK_BATCH):
        batch = todo[start : start + CHECK_BATCH]
        try:
            reply = ask(revise_messages(batch, notes, name), Revisions, 4096)
        except Exception as exc:
            report.setdefault("revise_errors", []).append(str(exc)[:200])
            continue
        report["calls"] = report.get("calls", 0) + 1
        for r in reply.revisions or []:
            if not 1 <= r.sentence <= len(batch):
                continue
            old = batch[r.sentence - 1]
            if r.starts_with and _first_words(r.starts_with, 2) != _first_words(
                old.text, 2
            ):
                continue
            text = " ".join(str(r.text or "").split())
            if text and text != old.text:
                new[id(old)] = Written(
                    text, list(old.notes), old.section, old.para, revised=old.text
                )
    fresh = list(new.values())
    code_check(fresh, notes, name)
    model_check(fresh, notes, name, ask, report)
    report["revised"] = len(fresh)
    report["revised_kept"] = sum(1 for w in fresh if not w.dropped)
    out: List[Written] = []
    for w in written:
        out.append(w)
        if id(w) in new:
            out.append(new[id(w)])
    return out
