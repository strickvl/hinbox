"""A last stage that reads the whole finished page once: facts told twice in
different sections, and sentences in different sections that contradict
each other.

Why: on 21 Nov (round 19) a full read of the long pages found what no
earlier stage could see, because every earlier stage looks at one section
or one plan part at a time. The CIA page gave Khalid Sheik Mohammed's 183
waterboardings four times and said in one section that Majid Khan was "not
subjected to waterboarding" and in another that CIA agents waterboarded him
twice, with nothing to say the accounts differ. So one call is shown the
page's kept sentences, numbered under their headings, and lists:

- repeats: a sentence that says nothing an earlier one doesn't. Code drops
  it only when its numbers and the people it names are all in that earlier
  sentence too, so a sentence adding a date or a name is never lost.
- contradictions: two sentences that cannot both be true of the same
  thing. One more call rewrites the pair from both sentences' notes as one
  passage telling both accounts, checked like any sentence; when it is kept
  it takes the earlier sentence's place and both old sentences are dropped.
"""

from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.engine.written_page import Ask, Note, Written

REVIEW = """You read a finished encyclopedia entry about the {what} \
"{name}", sentence by sentence (numbered, under their headings), and list \
two kinds of fault.

1. repeats: a sentence that tells again a fact an earlier sentence \
already told (in any section) and adds nothing to it: no new date, number, \
name, place or detail. Give its number and the earlier sentence's number.
2. contradictions: two sentences that cannot both be true about the same \
person and the same thing (one says he was not waterboarded, another that \
he was; one says a fine was waived, another that it stood), with neither \
saying the accounts differ. Two different events, two different people, a \
state that changed over time, or a later decision are not contradictions. \
Give both numbers and, in a few words, what they disagree about.

List only clear cases. If there are none, give empty lists."""


class Repeat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sentence: int
    repeats: int = Field(description="The earlier sentence it repeats")


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    first: int
    second: int
    about: str = ""


class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    repeats: List[Repeat] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)


BOTH = """You fix two sentences of an encyclopedia entry about the {what} \
"{name}" that contradict each other ({about}). Below are the two \
sentences, then the notes they were written from (sentences from news \
articles, each with its article's date).

Write one or two sentences to replace both. Keep every other fact the two \
sentences give. Tell the disputed point as two accounts, each with its \
source and date as the notes give them ("A 2014 Senate report says ...; \
an earlier account says ..."), without judging which is right. If the \
notes show the two are about different things or different times, say \
that plainly instead. Use only what the notes say. Every sentence lists \
the numbers of the notes it uses in its notes field; never write note \
numbers into the sentence itself."""


def review_messages(
    kept: Sequence[Written], name: str, what: str
) -> List[Dict[str, str]]:
    lines, section = [], None
    for i, w in enumerate(kept, 1):
        if w.section != section:
            section = w.section
            lines.append(f"## {section or 'Opening'}")
        lines.append(f"S{i}: {w.text}")
    return [
        {"role": "system", "content": REVIEW.format(what=what, name=name)},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _covered(later: Written, earlier: Written, name: str) -> bool:
    """Every number and person of ``later`` is in ``earlier``."""
    from src.engine.written_checks import _numbers
    from src.engine.written_plan import _surnames

    return _numbers(later.text) <= _numbers(earlier.text) and _surnames(
        later.text, name
    ) <= _surnames(earlier.text, name)


def review_page(
    written: List[Written], notes: Sequence[Note], name: str, what: str,
    ask: Ask, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with repeats across sections dropped and each
    contradiction told as two accounts, or left as it was when the rewrite
    fails its checks."""
    from src.engine.written_checks import code_check, model_check
    from src.engine.written_page import (
        close_quotes,
        notes_block,
        split_sentences,
        strip_refs,
    )
    from src.engine.written_revise import Fill

    kept = [w for w in written if not w.dropped]
    if len(kept) < 4:
        return written
    try:
        reply = ask(review_messages(kept, name, what), Review, 2048)
    except Exception as exc:
        report.setdefault("review_errors", []).append(str(exc)[:200])
        return written
    report["calls"] = report.get("calls", 0) + 1
    num = lambda k: kept[k - 1] if 1 <= k <= len(kept) else None  # noqa: E731
    dropped = 0
    for r in reply.repeats or []:
        later, earlier = num(r.sentence), num(r.repeats)
        if later is None or earlier is None or r.repeats >= r.sentence:
            continue
        if later.dropped or earlier.dropped or not later.section:
            continue
        if _covered(later, earlier, name):
            later.dropped = "repeats an earlier sentence of the page"
            dropped += 1
    by_n = {n.n: n for n in notes}
    out = list(written)
    told, fresh = 0, []
    for c in (reply.contradictions or [])[:4]:
        a, b = num(min(c.first, c.second)), num(max(c.first, c.second))
        if a is None or b is None or a is b or a.dropped or b.dropped:
            continue
        ns = [n for n in dict.fromkeys(a.notes + b.notes) if n in by_n]
        if not ns:
            continue
        msgs = [
            {"role": "system", "content": BOTH.format(what=what, name=name, about=c.about or "see the sentences")},
            {"role": "user", "content": f"Sentence A: {a.text}\nSentence B: {b.text}\n\nTheir notes:\n{notes_block([by_n[n] for n in ns])}"},
        ]  # fmt: skip
        try:
            fix = ask(msgs, Fill, 1024)
        except Exception as exc:
            report.setdefault("review_errors", []).append(str(exc)[:200])
            continue
        report["calls"] += 1
        new = []
        for s in fix.sentences or []:
            text, refs = strip_refs(" ".join(str(s.text or "").split()))
            cited = [n for n in dict.fromkeys(list(s.notes or []) + refs) if n in by_n]
            for one in split_sentences(text) if text else []:
                new.append(Written(close_quotes(one), cited or ns, a.section, a.para))
        code_check(new, notes, name)
        model_check(new, notes, name, ask, report)
        if not new or any(w.dropped for w in new):
            continue  # both stay as they were
        a.dropped = b.dropped = "told again with both accounts"
        at = out.index(a)
        out[at + 1 : at + 1] = new
        fresh += new
        told += 1
    report["review"] = {"repeats": dropped, "contradictions": told,
                        "listed": len(reply.contradictions or [])}  # fmt: skip
    return out
