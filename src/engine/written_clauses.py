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
from typing import List, Tuple

from src.engine.written_page import Written

# "..., and suspended Baker's contempt sentence" / "...; the board ...": a
# clause starting lower case, so a list of names ("Roe, Moss, and Vale")
# is never cut
_CLAUSE = re.compile(r"(?:,\s+and|;)\s+(?=[a-z])")
# "In May 2018, reports confirmed that X", "An August 2018 report said X",
# "X was reported to have been resettled"
_REPORT = re.compile(
    r"^(?:(?:In|By|On)\s+[^,]{3,30},\s+)?(?:an?\s+|the\s+|later\s+)?"
    r"(?:[A-Z][a-z]+\s+)?(?:\d{4}\s+)?(?:news\s+)?(?:reports?|accounts?|articles?)\s+"
    r"(?:said|confirmed|noted|stated|reported|added|showed)\s+(?:that\s+)?",
    re.I,
)
_REPORTED = re.compile(r"\b(?:was|were) reported to have (?:been )?", re.I)


def _facts(text: str, name: str) -> Tuple[set, set, set]:
    from src.engine.written_checks import _content, _numbers
    from src.engine.written_plan import _surnames

    return _content(text), _numbers(text), _surnames(text, name)


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


def drop_report_repeats(written: List[Written], name: str) -> int:
    """Drop a section sentence that only tells, as a later report, what an
    earlier kept sentence of the page already tells."""
    dropped, earlier = 0, []
    for w in written:
        if w.dropped:
            continue
        inner = unwrap_report(w.text)
        # the report's own date went with its wrapper: it is not a fact of
        # what the report says
        if w.section and inner != w.text and _within(inner, earlier, name, 0.6):
            w.dropped = "tells again, as a later report, an earlier sentence"
            dropped += 1
            continue
        earlier.append(w.text)
    return dropped


def drop_repeat_clauses(written: List[Written], name: str) -> int:
    """Cut from a section sentence a clause that an earlier kept sentence of
    its section tells (its numbers, people and 60% of its words); the
    number cut."""
    cut = 0
    said: dict = {}
    for w in written:
        if w.dropped or not w.section:
            continue
        before = said.setdefault(w.section, [])
        pieces = _CLAUSE.split(w.text)
        if len(pieces) > 1 and before:
            keep = [pieces[0]] + [
                p for p in pieces[1:] if not _within(p, before, name, 0.6)
            ]
            if len(keep) < len(pieces):
                text = _join(w.text, keep)
                if text and text != w.text:
                    w.text, cut = text, cut + len(pieces) - len(keep)
        before.append(w.text)
    return cut


def _join(original: str, keep: List[str]) -> str:
    """The kept pieces put back with the joins they had."""
    joins = [m.group(0) for m in _CLAUSE.finditer(original)]
    pieces = _CLAUSE.split(original)
    out = pieces[0]
    for j, p in zip(joins, pieces[1:]):
        if p in keep:
            out += j + p
    out = out.rstrip(" ,;")
    return out if out.endswith((".", "!", "?", '"', "”")) else out + "."
