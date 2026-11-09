"""Where each cited claim of a profile sits in its text.

A claim is the text since the previous citation marker, cut the way the
grounding check cuts it (``_extract_cited_claims()`` in
``src/utils/grounding.py``), so a verdict, a reader's mark and a historian's
correction all name the same words. The reader (``profile_render.py``) and
the pipeline's corrections pass (``src/engine/corrections_pass.py``) both
find claims here.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Tuple

from src.utils.grounding import _HEADING_LINES
from src.utils.grounding import _LEADING_PUNCT as _GROUNDING_LEAD
from src.utils.quality_controls import CITATION_RE, citation_ids

# No "^": it is matched at an offset with pattern.match(text, pos)
_LEADING_PUNCT = re.compile(r"[\s.,;:!?)\]]+")
_HAS_WORD = re.compile(r"\w")
# Lines that start a new Markdown block: a highlight span must not cross one
_BLOCK_START = re.compile(r"\n[ \t]*(?:\n|#|[-*+][ \t]|\d+[.)][ \t])")
_SKIP_LINE = re.compile(r"[ \t]*(?:#[^\n]*)?\n")  # blank or heading line
_LIST_MARK = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+")


@dataclass
class Span:
    """One cited claim: its text, where it starts, and where its first
    citation marker starts."""

    index: int  # 0-based, in text order
    text: str
    start: int
    end: int
    articles: List[str] = field(default_factory=list)


@dataclass
class Marker:
    """One ``^[...]`` marker: where it sits and which claim it closes."""

    start: int
    end: int
    articles: List[str]
    claim: int  # index into the claims list


def claim_text(text: str, last_end: int, marker_start: int) -> str:
    """The claim exactly as ``_extract_cited_claims()`` cuts it: everything
    since the previous marker, heading lines dropped, leading punctuation
    stripped. This text is what the stored verdict is keyed on."""
    span = _HEADING_LINES.sub("", text[last_end:marker_start])
    return _GROUNDING_LEAD.sub("", span.strip()).strip()


def claim_start(text: str, last_end: int, marker_start: int) -> int:
    """Where a claim's visible words start: after the previous marker, but
    never before the start of the block (paragraph, heading, list item) the
    marker sits in, so the span stays inside one HTML element. A sentence
    wrapped over two lines of one paragraph is taken whole."""
    start = last_end
    for m in _BLOCK_START.finditer(text, last_end, marker_start):
        start = m.start() + 1  # the line that opens the block
    while True:  # step over blank and heading lines
        skip = _SKIP_LINE.match(text, start)
        if not skip or skip.end() > marker_start:
            break
        start = skip.end()
    for pat in (_LIST_MARK, _LEADING_PUNCT):
        m = pat.match(text, start)
        if m and m.end() <= marker_start:
            start = m.end()
    return min(start, marker_start)


def parse_claims(
    text: str, make: Callable[[int, str, int, int], Any] = Span
) -> Tuple[List[Any], List[Marker]]:
    """Every cited claim and marker in a profile, in text order. ``make``
    builds each claim from (index, text, start, end); it needs an
    ``articles`` list and an ``index``, as ``Span`` has."""
    claims: List[Any] = []
    markers: List[Marker] = []
    last_end = 0
    for match in CITATION_RE.finditer(text):
        ids = citation_ids(match.group(1))
        if not ids:
            continue
        span = claim_text(text, last_end, match.start())
        if _HAS_WORD.search(span) or not claims:
            start = claim_start(text, last_end, match.start())
            claims.append(make(len(claims), span, start, match.start()))
        claim = claims[-1]
        for aid in ids:
            if aid not in claim.articles:
                claim.articles.append(aid)
        markers.append(Marker(match.start(), match.end(), ids, claim.index))
        last_end = match.end()
    return claims, markers


def claim_end(markers: List[Marker], index: int) -> int:
    """Where claim ``index``'s last citation marker ends."""
    return max(m.end for m in markers if m.claim == index)
