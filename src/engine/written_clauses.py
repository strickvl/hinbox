"""Repeats smaller than a sentence, found by code after the page is put
together: a fact told plainly once and again as a clause of a longer
sentence, and a fact told again as what a later report said.

Why: in round 28 (22 Nov) the chosen sections still had 16 repeated pairs.
Half were a clause: "On November 3, Rishikof suspended Brig. Gen. John
Baker's 21-day contempt sentence" and, two sentences on, "In February
2018, Rishikof supported the war court judge's view ..., and suspended
Baker's contempt sentence." Others were a repeat wrapped as a report under
its article's date, which the one-time-event check can't see: "In May
2018, reports confirmed that Brown was dismissed alongside Rishikof."
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from src.engine.written_page import Written

# "In May 2018, reports confirmed that X", "An August 2018 report said X",
# "X was reported to have been resettled"
_REPORT = re.compile(
    r"^(?:(?:In|By|On)\s+[^,]{3,30},\s+)?(?:an?\s+|the\s+|later\s+)?"
    r"(?:[A-Z][a-z]+\s+)?(?:\d{4}\s+)?(?:news\s+)?(?:reports?|accounts?|articles?)\s+"
    r"(?:said|confirmed|noted|stated|reported|added|showed)\s+(?:that\s+)?",
    re.I,
)
_REPORTED = re.compile(r"\b(?:was|were) reported to have (?:been )?", re.I)


_NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|dozens?|"
    r"hundreds?)\b",
    re.I,
)


def _facts(text: str, name: str) -> Tuple[set, set, set]:
    """Words, numbers (number words too: round 30 cut "two were
    hospitalized" as told by a sentence about three) and people."""
    from src.engine.written_checks import _content, _numbers
    from src.engine.written_plan import _surnames

    nums = _numbers(text) | {m.lower() for m in _NUMBER_WORDS.findall(text)}
    return _content(text), nums, _surnames(text, name)


def _within(part: str, earlier: List[str], name: str, share: float) -> bool:
    """Every number and person of ``part``, and ``share`` of its words, in
    one earlier sentence."""
    words, nums, people = _facts(part, name)
    if len(words) < 3:
        return False
    for old in earlier:
        w, n, p = _facts(old, name)
        if nums <= n and people <= p and len(words & w) >= share * len(words):
            return True
    return False


def unwrap_report(text: str) -> str:
    """``text`` without "In May 2018, reports confirmed that" or "was
    reported to have been"; ``text`` itself when it has neither."""
    m = _REPORT.match(text)
    out = text[m.end() :] if m else text
    return _REPORTED.sub("was " if "was" in text else "were ", out)


def drop_report_repeats(written: List[Written], name: str) -> List[str]:
    """Drop a section sentence that only tells, as a later report, what an
    earlier kept sentence of the page already tells; the dropped texts."""
    dropped: List[str] = []
    earlier: List[str] = []
    for w in written:
        if w.dropped:
            continue
        inner = unwrap_report(w.text)
        # the report's own date went with its wrapper: it is not a fact of
        # what the report says
        if w.section and inner != w.text and _within(inner, earlier, name, 0.6):
            w.dropped = "tells again, as a later report, an earlier sentence"
            dropped.append(w.text)
            continue
        earlier.append(w.text)
    return dropped


def _clauses(text: str) -> List[Tuple[str, str]]:
    """``text`` as (join, clause) pairs, the first join "". A clause starts
    lower case after ";" or ", and"; the last item of a list ("upheld X,
    suspended Y, and struck down Z") stays with its list, since round 30 cut
    "upheld Baker's conviction, suspended confinement" and left "and struck
    down the $1,000 fine" on the sentence before."""
    out: List[Tuple[str, str]] = []
    for k, part in enumerate(re.split(r";\s+(?=[a-z])", text)):
        join = "; " if k else ""
        bits = re.split(r"(,\s+and\s+)(?=[a-z])", part)
        piece = bits[0]
        for sep, rest in zip(bits[1::2], bits[2::2]):
            if re.search(r",\s+[a-z]", piece):  # a list: keep it whole
                piece += sep + rest
                continue
            out.append((join, piece))
            join, piece = sep, rest
        out.append((join, piece))
    return out


def drop_repeat_clauses(written: List[Written], name: str) -> List[List[str]]:
    """Cut from a section sentence a clause that an earlier kept sentence of
    its section tells (its numbers, people and 60% of its words); each cut
    as [before, after]."""
    cut: List[List[str]] = []
    said: dict = {}
    for w in written:
        if w.dropped or not w.section:
            continue
        before = said.setdefault(w.section, [])
        parts = _clauses(w.text)
        if len(parts) > 1 and before:
            keep = [parts[0]] + [
                (j, p) for j, p in parts[1:] if not _within(p, before, name, 0.6)
            ]
            if len(keep) < len(parts):
                text = "".join(j + p for j, p in keep).rstrip(" ,;")
                if not text.endswith((".", "!", "?", '"', "”")):
                    text += "."
                if text != w.text:
                    cut.append([w.text, text])
                    w.text = text
        before.append(w.text)
    return cut


def _told_by(a: str, b: str, name: str) -> bool:
    """Everything ``a`` tells is in ``b``: no person or number ``b`` lacks,
    60% of its words in ``b`` and at most two words more."""
    (wa, na, pa), (wb, nb, pb) = _facts(a, name), _facts(b, name)
    return (
        len(wa) >= 3 and na <= nb and pa <= pb
        and len(wa & wb) >= 0.6 * len(wa) and len(wa - wb) <= 2
    )  # fmt: skip


def drop_section_repeats(written: List[Written], name: str) -> List[str]:
    """Drop a section sentence whose whole content another sentence of its
    section tells: the later one when the earlier tells it (round 29 kept
    "In April 2013, most of the men awaiting transfer were Yemenis,
    including Hamiri." right after a sentence saying so), the earlier one
    when the later, fuller one tells it (round 30). A sentence that adds a
    fact (three words or more) is left for the review's trim."""
    dropped: List[str] = []
    said: Dict[str, List[Written]] = {}
    for w in written:
        if w.dropped or not w.section:
            continue
        before = said.setdefault(w.section, [])
        old = next((o for o in before if _told_by(w.text, o.text, name)), None)
        if old is not None:
            w.dropped = "repeats an earlier sentence of its section"
            dropped.append(w.text)
            continue
        for o in [o for o in before if _told_by(o.text, w.text, name)]:
            o.dropped = "told again, more fully, later in its section"
            dropped.append(o.text)
            w.notes = list(dict.fromkeys(o.notes + w.notes))
            before.remove(o)
        before.append(w)
    return dropped


def tidy_page(
    written: List[Written], notes: Sequence[Any], name: str, report: Dict[str, Any]
) -> None:
    """The code passes run on a finished page and again on a picked one;
    what each changed is added to the report, so nothing is lost from view."""
    from src.engine.written_dates import own_dates

    for key, found in (
        ("own_dates", own_dates(written, notes)),
        ("section_repeats", drop_section_repeats(written, name)),
        ("report_repeats", drop_report_repeats(written, name)),
        ("clauses_cut", drop_repeat_clauses(written, name)),
    ):
        report[key] = list(report.get(key) or []) + found
