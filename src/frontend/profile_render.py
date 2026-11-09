"""Turn a stored profile into reader HTML: citations, checker verdicts, contents.

A profile is Markdown whose sentences end in citation markers
(``fact.^[article-id]`` or ``fact^[a1, a2].``). The grounding pass stored one
verdict per (claim, article) in ``entity["profile_grounding"]["verifications"]``,
where a claim is the text between the previous marker and this one
(``_extract_cited_claims()`` in ``src/utils/grounding.py``). Here we walk the
same markers, find each claim's verdict again, and render:

- each claim wrapped in ``<span class="s v-<verdict>">`` so the page can mark
  the exceptions (partly supported, not supported, not checked) and highlight
  the sentence whose source is open;
- each marker as numbered ``<sup>`` links, numbered by first appearance,
  pointing at the in-app article page (the reader script opens them in the
  source panel instead);
- the headings as a contents list, with claim and problem counts per section.

Nothing here calls a model; it only reads saved fields.
"""

import html
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import markdown

from src.utils.grounding import _claim_key
from src.utils.quality_controls import CITATION_RE, citation_ids

# Verdicts in order of how much they need the reader's attention
SEVERITY = {
    "supported": 0,
    "unchecked": 1,
    "unclear": 1,
    "partial": 2,
    "not_supported": 3,
}
VERDICT_LABELS = {
    "supported": "Supported",
    "partial": "Partly supported",
    "not_supported": "Not supported",
    "unclear": "Checker unsure",
    "unchecked": "Not checked yet",
    "missing_source": "Source text missing",
}
# No "^": it is matched at an offset with pattern.match(text, pos)
_LEADING_PUNCT = re.compile(r"[\s.,;:!?)\]]+")
_HAS_WORD = re.compile(r"\w")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
# Sentences where the profile says two sources differ ("Sources disagree on…",
# "…2,974 people, rather than 2,976"), collected in one place on the page
_DISPUTE = re.compile(
    r"sources disagree|disagree (?:on|about|over)|reports? differ|conflicting (?:report|account|with)|"
    r"(?:another|a later|an earlier|a conflicting) report|"
    r"rather than the (?:previously|earlier|other)|differs? from (?:the |an? )?(?:earlier|previous|other)",
    re.IGNORECASE,
)
# Private-use characters mark where claim spans start and end; they pass
# through Markdown untouched and are swapped for HTML afterwards
_OPEN, _MID, _CLOSE, _CITE = "", "", "", ""


@dataclass
class Claim:
    """One cited stretch of profile text and the verdict on each source."""

    index: int  # 0-based, in text order
    text: str
    start: int  # where the visible claim starts in the profile text
    end: int  # where its first citation marker starts
    articles: List[str] = field(default_factory=list)
    verdicts: Dict[str, str] = field(default_factory=dict)  # article id -> level
    reasons: Dict[str, str] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """The verdict needing most attention across its sources."""
        if not self.verdicts:
            return "unchecked"
        return max(self.verdicts.values(), key=lambda v: SEVERITY.get(v, 1))

    @property
    def is_dispute(self) -> bool:
        return bool(_DISPUTE.search(self.text))


@dataclass
class Marker:
    """One ``^[...]`` marker: where it sits and which claim it closes."""

    start: int
    end: int
    articles: List[str]
    claim: int  # index into the claims list


def _claim_start(text: str, last_end: int, marker_start: int) -> int:
    """Where the visible claim before a marker starts: after the previous
    marker, but never before the start of the marker's own line, so a heading
    or an uncited paragraph is not swept into it."""
    line_start = text.rfind("\n", last_end, marker_start) + 1
    start = max(last_end, line_start)
    lead = _LEADING_PUNCT.match(text, start)
    if lead:
        start = lead.end()
    return min(start, marker_start)


def parse_claims(text: str) -> Tuple[List[Claim], List[Marker]]:
    """Every cited claim and marker in a profile, in text order."""
    claims: List[Claim] = []
    markers: List[Marker] = []
    last_end = 0
    for match in CITATION_RE.finditer(text):
        ids = citation_ids(match.group(1))
        if not ids:
            continue
        start = _claim_start(text, last_end, match.start())
        span = text[start : match.start()].strip()
        if _HAS_WORD.search(span) or not claims:
            claims.append(Claim(len(claims), span, start, match.start()))
        claim = claims[-1]
        for aid in ids:
            if aid not in claim.articles:
                claim.articles.append(aid)
        markers.append(Marker(match.start(), match.end(), ids, claim.index))
        last_end = match.end()
    return claims, markers


def attach_verdicts(claims: List[Claim], report: Optional[Dict[str, Any]]) -> bool:
    """Fill each claim's verdicts from a stored grounding report.

    Returns False when there is no usable report (nothing was checked), so
    the page can say so once instead of greying every sentence.
    """
    verifications = (report or {}).get("verifications") or []
    if not verifications:
        return False
    known: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_article: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for v in verifications:
        key = _claim_key(str(v.get("claim", "")))
        aid = str(v.get("article_id", ""))
        known[(aid, key)] = v
        by_article.setdefault(aid, []).append((key, v))
    for claim in claims:
        key = _claim_key(claim.text)
        for aid in claim.articles:
            v = known.get((aid, key))
            if v is None:  # stored claims are cut at the judge's limit
                v = next(
                    (
                        sv
                        for sk, sv in by_article.get(aid, [])
                        if sk and (key.startswith(sk) or sk.startswith(key))
                    ),
                    None,
                )
            if v is None:
                claim.verdicts[aid] = "unchecked"
                continue
            claim.verdicts[aid] = str(v.get("support_level", "unchecked"))
            if v.get("reasoning"):
                claim.reasons[aid] = str(v["reasoning"])
    return True


def shift_headings(text: str) -> str:
    """Make the profile's top heading level an ``h2`` (the page's ``h1`` is
    the entity name); profiles often start at ``###``."""
    levels = [len(m.group(1)) for m in _HEADING.finditer(text)]
    if not levels:
        return text
    shift = min(levels) - 2

    def fix(m: "re.Match[str]") -> str:
        level = max(2, min(6, len(m.group(1)) - shift))
        return f"{'#' * level} {m.group(2)}"

    return _HEADING.sub(fix, text)


@dataclass
class Section:
    """A contents entry: heading, depth, anchor, claim and problem counts."""

    title: str
    level: int
    anchor: str
    claims: int = 0
    flags: Dict[str, int] = field(default_factory=dict)


@dataclass
class RenderedProfile:
    html: str
    claims: List[Claim]
    sections: List[Section]
    order: List[str]  # article ids by first citation
    checked: bool  # a grounding report exists

    def counts(self) -> Dict[str, int]:
        """Sentences per verdict; a sentence citing two sources counts once,
        under the verdict needing most attention."""
        out: Dict[str, int] = {}
        for c in self.claims:
            out[c.verdict] = out.get(c.verdict, 0) + 1
        return out

    def citation_total(self) -> int:
        return sum(max(1, len(c.articles)) for c in self.claims)


def _slug(title: str, used: Dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"
    n = used.get(base, 0)
    used[base] = n + 1
    return base if n == 0 else f"{base}-{n + 1}"


def _sections(text: str, claims: List[Claim], checked: bool) -> List[Section]:
    heads = list(_HEADING.finditer(text))
    used: Dict[str, int] = {}
    out = [
        Section(m.group(2).strip(), len(m.group(1)), _slug(m.group(2).strip(), used))
        for m in heads
    ]
    for claim in claims:
        idx = -1
        for i, m in enumerate(heads):
            if m.start() <= claim.start:
                idx = i
        if idx < 0:
            continue
        sec = out[idx]
        sec.claims += 1
        if checked and claim.verdict != "supported":
            sec.flags[claim.verdict] = sec.flags.get(claim.verdict, 0) + 1
    # A parent heading's counts include its sub-sections
    for i, sec in enumerate(out):
        for sub in out[i + 1 :]:
            if sub.level <= sec.level:
                break
            sec.claims += sub.claims
            for k, n in sub.flags.items():
                sec.flags[k] = sec.flags.get(k, 0) + n
    return out


def _sup(
    articles: List[str], claim: Claim, order: List[str], checked: bool, link: str
) -> str:
    parts = []
    for aid in articles:
        n = order.index(aid) + 1
        v = claim.verdicts.get(aid, "unchecked") if checked else "supported"
        mark = ""
        if v != "supported":
            sym = {"partial": "?", "not_supported": "✕"}.get(v, "·")
            mark = f'<span class="vm v-{v}" aria-hidden="true">{sym}</span>'
        label = html.escape(f"Source {n}: {VERDICT_LABELS.get(v, v)}")
        href = html.escape(link.format(article=aid, claim=claim.index), quote=True)
        parts.append(
            f'<a class="cite" href="{href}" data-article="{html.escape(aid)}" '
            f'data-claim="{claim.index}" aria-label="{label}">{n}</a>{mark}'
        )
    return f'<sup class="c">{",".join(parts)}</sup>'


def render_profile(
    text: str,
    report: Optional[Dict[str, Any]] = None,
    link: str = "/sources/{article}",
    show_verdicts: bool = True,
) -> RenderedProfile:
    """Render a profile's Markdown with numbered citations and verdict marks.

    ``link`` is the citation's fallback URL (``{article}`` and ``{claim}``
    are filled in). ``show_verdicts=False`` renders an older version, whose
    text the stored report doesn't describe.
    """
    text = shift_headings(text or "")
    claims, markers = parse_claims(text)
    checked = show_verdicts and attach_verdicts(claims, report)
    order: List[str] = []
    for m in markers:
        for aid in m.articles:
            if aid not in order:
                order.append(aid)
    sections = _sections(text, claims, checked)

    # Edits as (position, rank, chars replaced, new text), applied back to
    # front so earlier offsets stay valid. At one position the result reads
    # left to right: a claim's close, the next claim's open, then a marker.
    edits: List[Tuple[int, int, int, str]] = []
    for c in claims:
        if c.end > c.start:
            v = c.verdict if checked else "supported"
            edits.append((c.start, 1, 0, f"{_OPEN}{c.index}|{v}{_MID}"))
            edits.append((c.end, 0, 0, _CLOSE))
    subs: List[str] = []
    for m in markers:
        edits.append((m.start, 2, m.end - m.start, f"{_CITE}{len(subs)}{_CITE}"))
        subs.append(_sup(m.articles, claims[m.claim], order, checked, link))
    out = text
    for pos, _, n, token in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        out = out[:pos] + token + out[pos + n :]

    body = markdown.markdown(
        html.escape(out, quote=False), extensions=["sane_lists"], output_format="html"
    )
    body = re.sub(f"{_CITE}(\\d+){_CITE}", lambda m: subs[int(m.group(1))], body)
    body = re.sub(
        f"{_OPEN}(\\d+)\\|(\\w+){_MID}",
        lambda m: f'<span class="s v-{m.group(2)}" id="claim-{m.group(1)}">',
        body,
    )
    body = body.replace(_CLOSE, "</span>")
    # Give headings the anchors the contents list links to
    it = iter(sections)

    def anchor(m: "re.Match[str]") -> str:
        sec = next(it, None)
        return f'<h{m.group(1)} id="{sec.anchor}">' if sec else m.group(0)

    body = re.sub(r"<h([2-6])>", anchor, body)
    return RenderedProfile(body, claims, sections, order, checked)


def plain_claim(text: str) -> str:
    """A claim as a reader would quote it: no markers, single spaces."""
    return " ".join(CITATION_RE.sub("", text).split()).strip()
