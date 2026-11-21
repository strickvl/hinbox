"""Last stages of a written page: each disagreement the plan found is told,
and sentences that narrate sources where nothing is disputed are rewritten.

Why a stage of its own: on 21 Nov the plan listed bin Attash's two capture
dates (Sept. 11, 2002, against April 29, 2003) in all three Karachi runs,
but in two of them the page never told it. Once the section writer mixed
his raid into another man's sentence, the check rightly dropped that half
and the rewrite cut it; the other time the fill stage wrote only the April
date. Nothing looked at the finished page to see that each listed
disagreement was still there. ``tell_conflicts()`` does: a disagreement no
kept sentence tells (citing notes from both sides) gets one call that
writes it, checked like any sentence, placed after the first kept sentence
citing one of its notes.

``unnarrate()``: GPT-6 Luna still wrote "a news account described ..." and
"another account says ..." where no two sources disagree. Such sentences
get one rewrite, as the revise stage does for any dropped sentence.
"""

from typing import Any, Dict, List, Sequence

from src.engine.written_checks import NARRATES, code_check, model_check
from src.engine.written_page import Ask, Note, Written

TELL = """You add one passage to an encyclopedia entry about the {what} \
"{name}". The notes below (sentences from news articles, each with its \
article's date) disagree: {conflict}.

Write one or two sentences that tell it as one event with two accounts \
and cite both sides: say how many and which sources give each ("One \
account dates his capture to Sept. 11, 2002; three later reports, among \
them the Red Cross, give April 29, 2003."). Use only what the notes say; \
don't judge which is right. Every sentence lists the numbers of the notes \
it uses in its notes field; never write note numbers into the sentence \
itself. If the notes do not really disagree (two different people or \
events, or both can be true), give an empty list."""


def tells(w: Written, c: Dict[str, Any]) -> bool:
    return any(n in w.notes for n in c.get("one_notes") or []) and any(
        n in w.notes for n in c.get("other_notes") or []
    )


def _describe(c: Dict[str, Any]) -> str:
    nums = lambda ns: ", ".join(str(n) for n in ns)  # noqa: E731
    return (
        f"{c.get('about', '')}: {c.get('one_side', '')} (notes "
        f"{nums(c.get('one_notes') or [])}) against {c.get('other_side', '')} "
        f"(notes {nums(c.get('other_notes') or [])})"
    )


def tell_conflicts(
    written: List[Written], notes: Sequence[Note], name: str, what: str,
    conflicts: Sequence[Dict[str, Any]], ask: Ask, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with a passage for each conflict no kept sentence tells."""
    from src.engine.written_page import notes_block, split_sentences, strip_refs
    from src.engine.written_revise import Fill

    by_n = {n.n: n for n in notes}
    out = list(written)
    fresh: List[Written] = []
    for c in conflicts:
        ns = [n for n in (c.get("one_notes") or []) + (c.get("other_notes") or [])]
        ns = [n for n in dict.fromkeys(ns) if n in by_n]
        kept = [w for w in out if not w.dropped and w.section]
        if not ns or any(tells(w, c) for w in kept):
            continue
        msgs = [
            {"role": "system", "content": TELL.format(what=what, name=name, conflict=_describe(c))},
            {"role": "user", "content": notes_block([by_n[n] for n in ns])},
        ]  # fmt: skip
        try:
            reply = ask(msgs, Fill, 1024)
        except Exception as exc:
            report.setdefault("tell_errors", []).append(str(exc)[:200])
            continue
        report["calls"] = report.get("calls", 0) + 1
        near = [i for i, w in enumerate(out) if w in kept and set(w.notes) & set(ns)]
        if near:
            at, section, para = near[0], out[near[0]].section, out[near[0]].para
        else:  # the section holding most of the planned notes, at its end
            sections = [w.section for w in kept]
            section = max(set(sections), key=sections.count) if sections else ""
            at = max(
                (i for i, w in enumerate(out) if w.section == section),
                default=len(out) - 1,
            )
            para = out[at].para if out else 0
        added = []
        for s in reply.sentences or []:
            text, refs = strip_refs(" ".join(str(s.text or "").split()))
            cited = [n for n in dict.fromkeys(list(s.notes or []) + refs) if n in by_n]
            for one in split_sentences(text) if text else []:
                added.append(Written(one, cited, section, para, filled=True))
        out[at + 1 : at + 1] = added
        fresh += added
    code_check(fresh, notes, name)
    model_check(fresh, notes, name, ask, report)
    report["conflicts_told"] = sum(1 for w in fresh if not w.dropped)
    report["conflicts_untold"] = sum(
        1 for c in conflicts if not any(tells(w, c) for w in out if not w.dropped)
    )
    return out


_SOURCE_WORDS = {"a", "an", "another", "one", "the", "in", "on", "by", "according"}


def lost_or_added(old: str, new: str) -> bool:
    """The rewrite dropped a capitalised name the sentence had ("the
    International Red Cross reported" became a bare date, 21 Nov), added one
    it lacked ("CIA agents captured him", where no captor was named), or put
    in a day of the month it lacked (a 2014 description of feeding became
    "On November 19, 2014, Army guards shackled ...")."""
    from src.engine.written_checks import _CAP, _COMMON, _DAY

    def caps(t: str) -> set:
        return (
            {w.lower().strip("'’") for w in _CAP.findall(t)} - _SOURCE_WORDS - _COMMON
        )

    days = lambda t: {m.group(0).lower() for m in _DAY.finditer(t)}  # noqa: E731
    return bool(caps(old) ^ caps(new)) or bool(days(new) - days(old))


def drop_one_sided(written: List[Written], conflicts: Sequence[Dict[str, Any]]) -> None:
    """Drop a kept section sentence that states one side of a disagreement
    as plain fact in a section where a kept sentence tells it, when it cites
    only that disagreement's notes, gives one of the disputed dates, and
    names no person the telling sentence leaves out. On 21 Nov three runs of
    four told bin Attash's two capture dates and then also said, as fact,
    "a Pakistani-U.S. raid on September 11 captured Walid bin Attash"."""
    from src.engine.written_checks import _numbers
    from src.engine.written_plan import _surnames

    for c in conflicts:
        sides = set((c.get("one_notes") or []) + (c.get("other_notes") or []))
        values = _numbers(f"{c.get('one_side')} {c.get('other_side')}")
        told = [w for w in written if not w.dropped and w.section and tells(w, c)]
        for w in written:
            if w.dropped or not w.section or w in told or not set(w.notes) <= sides:
                continue
            mine = [t for t in told if t.section == w.section]
            if not mine or not values & _numbers(w.text):
                continue
            people = _surnames(w.text, "")
            if people and people <= set().union(*(_surnames(t.text, "") for t in mine)):
                w.dropped = "states one side of a disagreement the section tells"


def unnarrate(
    written: List[Written], notes: Sequence[Note], name: str,
    conflicts: Sequence[Dict[str, Any]], ask: Ask, report: Dict[str, Any],
) -> List[Written]:  # fmt: skip
    """``written`` with each kept section sentence that narrates sources
    ("a news account described") but tells no conflict rewritten once."""
    from src.engine.written_revise import revise

    marked = []
    for w in written:
        if w.dropped or not w.section or not NARRATES.search(w.text):
            continue
        if any(tells(w, c) for c in conflicts):
            continue
        w.dropped = (
            'check: it narrates the sources ("a news account described", '
            '"another account says") though nothing is disputed. State the '
            'fact itself. Keep a named source as the source ("according to '
            'the International Red Cross", "a Senate report said"), keep a '
            "description of how things were done a description, keep two "
            "descriptions of one thing as one thing, and add no date the "
            "sentence does not give"
        )
        marked.append(w)
    report["narrating"] = len(marked)
    if not marked:
        return written
    ids = {id(w) for w in marked}
    out = revise(
        written, notes, name, ask, report, pick=lambda w: id(w) in ids, tag="unnarrated"
    )
    # A sentence whose rewrite failed stays as it was: narrated is better
    # than lost (the June 2017 sentence held Bin Atef's 2009 clearance)
    for i, w in enumerate(out):
        if id(w) in ids:
            nxt = out[i + 1] if i + 1 < len(out) else None
            if nxt is None or nxt.revised != w.text or nxt.dropped:
                w.dropped = ""
            elif lost_or_added(w.text, nxt.text):
                nxt.dropped, w.dropped = "rewrite lost a source or added a day", ""
    return out
