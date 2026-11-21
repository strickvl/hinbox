"""Second chances for a written page: a dropped sentence is rewritten once
(``revise()``), and planned notes no kept sentence uses are written into
their section (``fill_unused()``).

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

from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.engine.written_checks import _first_words, code_check, model_check
from src.engine.written_page import (
    CHECK_BATCH,
    CHECK_NOTE_CHARS,
    Ask,
    Note,
    Written,
    strip_refs,
)

REVISE = """You fix sentences of an encyclopedia entry about "{name}". \
Each sentence below was taken out because a check found a problem with it, \
given after it, against the notes it was written from (sentences from news \
articles, each with its article's date).

Rewrite each sentence so it says only what its notes say: take out or \
correct the detail the check named, keep the rest, and keep its place in \
the entry (it must still read well after the sentence before it). Keep \
who said what, and never drop what changes the meaning (a retraction, a \
denial, an "alleged", "later withdrawn"): if the fix would need that, give \
the sentence up. Never add a detail the notes don't give. If nothing in the \
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
    if w.summary:  # an opening summary about the page, or with a stray detail
        return w.dropped in ("talks about the page",) or w.dropped.startswith(
            "not in the sections"
        )
    return bool(w.notes) and (
        w.dropped.startswith("not in its notes") or w.dropped.startswith("check")
    )


_PROBLEMS = {
    "talks about the page": "it talks about the entry itself; state the facts "
    'instead, as a sentence about the subject (local Qwen wrote "The entry '
    'covers his capture ...")',
}


def _swollen(old: str, new: str) -> bool:
    """An opening summary's rewrite that grew or quotes: on 21 Nov Luna
    replaced "The sections also describe ..." with 75 words copying the
    page's sentences, a review board's quotes among them."""
    return len(new.split()) > len(old.split()) or any(q in new for q in '"“”')


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
        problem = _PROBLEMS.get(w.dropped, w.dropped.removeprefix("check: "))
        parts.append(f"Sentence {i}: {w.text}\nProblem: {problem}\nIts notes:\n{lines}")
    return [
        {"role": "system", "content": REVISE.format(name=name)},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def revise(
    written: List[Written], notes: Sequence[Note], name: str, ask: Ask,
    report: Dict[str, Any], pick: Optional[Callable[[Written], bool]] = None,
    tag: str = "revised",
) -> List[Written]:  # fmt: skip
    """``written`` with a rewrite after each dropped sentence the model could
    fix (those ``pick`` accepts, when given); the rewrites are checked like
    any sentence."""
    todo = [w for w in written if revisable(w) and (pick is None or pick(w))]
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
            text = strip_refs(" ".join(str(r.text or "").split()))[0]
            if old.summary and _swollen(old.text, text):
                continue
            if text and text != old.text:
                new[id(old)] = Written(
                    text, list(old.notes), old.section, old.para,
                    summary=old.summary, revised=old.text,
                )  # fmt: skip
    fresh = list(new.values())
    # A rewritten opening summary is checked by ``summary_check()`` after this
    code_check([w for w in fresh if not w.summary], notes, name)
    model_check(fresh, notes, name, ask, report)
    report[tag] = len(fresh)
    report[f"{tag}_kept"] = sum(1 for w in fresh if not w.dropped)
    out: List[Written] = []
    for w in written:
        out.append(w)
        if id(w) in new:
            out.append(new[id(w)])
    return out


FILL = """You add to one section of an encyclopedia entry about the {what} \
"{name}". Below is the section "{heading}" as written, then notes planned \
for it that it does not use yet (sentences from news articles, each with \
its article's date).

Write a sentence or two for each of those notes, so the section tells \
what they say; each goes into the section at its place in date order, \
after what happened before it, so it must read well there on its own \
(name people in full rather than "he" or "his"). Use only what \
the notes say, keep who said what, say when the event or statement was \
(month and year; never "a 2016 report said" or "a news account \
described"), and don't repeat what the section already says. Leave out a note that only comments or judges and \
gives no fact ("this highlights ...", "the city represents ..."). A note \
about a different major event (a killing, an attack, a trial) gets its \
own short section: give its sentences a short plain heading in \
"heading"; leave "heading" empty for sentences that belong here. Every sentence lists the numbers of the notes it \
uses in its notes field; never write note numbers into the sentence \
itself. If nothing should be added, give an empty list."""


class FillSentence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str
    notes: List[int]
    heading: str = Field(default="", description="A new section's heading, or ''")


class Fill(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentences: List[FillSentence]


def fill_unused(
    written: List[Written], notes: Sequence[Note], name: str, what: str,
    parts: Sequence[Dict[str, Any]], ask: Ask, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with sentences added for each part's planned notes that
    no kept sentence uses, checked like any sentence. Why: on 21 Nov the
    plan put Daniel Pearl's killing into an arrests part in 3 runs of 4 and
    the section writer left it out, with nothing to say so."""
    from src.engine.written_checks import drop_repeats
    from src.engine.written_groups import place_after
    from src.engine.written_page import (
        close_quotes,
        notes_block,
        split_sentences,
        strip_refs,
    )

    by_n = {n.n: n for n in notes}
    # The opening's notes don't count: its first sentence cites the note
    # with Bin Atef's 2009 clearance, which no section then told
    used = {n for w in written if not w.dropped and w.section for n in w.notes}
    # A note whose people no kept sentence names is not used yet: on 21 Nov
    # the dispute sentence cited bin Attash's notes, which also gave Ammar al
    # Baluchi's arrest, and two pages of three never named al Baluchi
    from src.engine.written_plan import _surnames

    page = set().union(*(_surnames(w.text, name) for w in written if not w.dropped))
    used = {n for n in used if n not in by_n or _surnames(by_n[n].text, name) <= page}
    out = list(written)
    fresh: List[Written] = []
    for part in parts:
        heading = part.get("heading") or ""
        unused = [
            by_n[n] for n in part.get("notes") or [] if n in by_n and n not in used
        ]
        if not unused:
            continue
        mine = [w for w in out if w.section == heading]
        body = "\n".join(f"- {w.text}" for w in mine if not w.dropped) or "(empty)"
        msgs = [
            {"role": "system", "content": FILL.format(what=what, name=name, heading=heading)},
            {"role": "user", "content": f"The section:\n{body}\n\nNotes it does not use yet:\n{notes_block(unused[:8])}"},
        ]  # fmt: skip
        try:
            reply = ask(msgs, Fill, 2048)
        except Exception as exc:
            report.setdefault("fill_errors", []).append(str(exc)[:200])
            continue
        report["calls"] = report.get("calls", 0) + 1
        added = []
        for s in reply.sentences or []:
            text, refs = strip_refs(" ".join(str(s.text or "").split()))
            ns = [n for n in dict.fromkeys(list(s.notes or []) + refs) if n in by_n]
            own = " ".join(str(s.heading or "").strip("# ").split())
            # One item can hold several sentences (local Qwen pasted three
            # notes as one): cut it, so the repeat check sees each
            for one in split_sentences(text) if text else []:
                one = close_quotes(one)
                added.append(Written(one, ns, own or heading, 0, filled=True))
        # Each into its place in the section's date order, after the sentence
        # telling the note before it; new sections at the section's end
        order = {n: i for i, n in enumerate(part.get("notes") or [])}
        end = max((w.para for w in mine), default=0) + 1000
        for w in [a for a in added if a.section == heading]:
            at = place_after(out, heading, w.notes, order)
            w.para = (
                out[at].para
                if at >= 0 and out[at].section == heading
                else (out[at + 1].para if at + 1 < len(out) else end)
            )
            out.insert(at + 1, w)
        rest = [a for a in added if a.section != heading]
        for w in rest:
            w.para = end
        at = max(
            (i for i, w in enumerate(out) if w.section == heading), default=len(out) - 1
        )
        out[at + 1 : at + 1] = rest
        fresh += added
    code_check(fresh, notes, name)
    model_check(fresh, notes, name, ask, report)
    drop_repeats(out)
    report["filled"] = report.get("filled", 0) + len(fresh)
    report["filled_kept"] = report.get("filled_kept", 0) + sum(
        1 for w in fresh if not w.dropped
    )
    return out


def unwritten(
    written: List[Written], notes: Sequence[Note], name: str,
    parts: Sequence[Dict[str, Any]],
) -> List[int]:  # fmt: skip
    """Planned notes no kept section sentence uses (or whose people no kept
    sentence names), by number."""
    from src.engine.written_plan import _surnames

    by_n = {n.n: n for n in notes}
    used = {n for w in written if not w.dropped and w.section for n in w.notes}
    page = set().union(*(_surnames(w.text, name) for w in written if not w.dropped))
    used = {n for n in used if n not in by_n or _surnames(by_n[n].text, name) <= page}
    return [
        n for p in parts for n in p.get("notes") or [] if n in by_n and n not in used
    ]


def recite(
    written: List[Written], notes: Sequence[Note], name: str,
    parts: Sequence[Dict[str, Any]],
) -> None:  # fmt: skip
    """A sentence the code check dropped for details its notes lack gets the
    notes of its section that hold them, when some do (up to 4, those
    sharing most words with it), but only when its own notes share under
    40% of its words, so they are plainly the wrong ones. On 21 Nov one Luna
    run's Bin Atef section
    came back citing the wrong note numbers; every sentence then failed
    both checks, though its planned notes said exactly what they said."""
    from src.engine.written_checks import _content, missing_details

    by_n = {n.n: n for n in notes}
    planned = {p.get("heading") or "": p.get("notes") or [] for p in parts}
    for w in written:
        if not w.dropped.startswith("not in its notes") or w.section not in planned:
            continue
        words = _content(w.text)
        own = {x for n in w.notes if n in by_n for x in _content(by_n[n].text)}
        if words and len(words & own) >= 0.4 * len(words):
            # Its notes are about what it says: the gap is a detail it added
            # ("Pakistani authorities arrested Majid Khan"), for the rewrite
            continue
        ranked = sorted(
            (n for n in planned[w.section] if n in by_n),
            key=lambda n: -len(words & _content(by_n[n].text)),
        )
        for k in range(1, 5):
            pick = ranked[:k]
            srcs = [f"{by_n[n].text} {by_n[n].date}" for n in pick]
            if pick and not missing_details(w.text, srcs, name):
                w.notes, w.dropped = pick, ""
                break
