"""A shortened copy of a long profile, for prompts. The stored profile is untouched.

A hub entity such as "United States" is cited by dozens of articles and its
profile grows to ~50k characters. Every update then sent all of it to the
model to add one sentence, and every merge check against it sent all of it
again, though most of that text had nothing to do with the new article.

``profile_view()`` builds what the model sees instead when a profile is over
its budget:

1. the opening text before the first heading (at most a quarter of the budget),
2. the heading lines (up to a third of the budget), so the model can still
   add to the end of a section,
3. the passages that share the most words, names and years with the new
   article, best first, until the budget is used up,

in their original order, with ``[… N passages not shown …]`` where text was
left out. A passage is a paragraph, a list item, or a run of sentences from a
very long paragraph.

The edit-list updater applies the model's edits to the full stored text: an
edit points at a sentence the model copied from the view, and code finds that
sentence in the full profile. So nothing is cut from storage, and citations
stay where they are.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from src.constants import PROFILE_VIEW_CHARS

_HEADING = re.compile(r"^#{1,6} ")
# After ". ", or after a citation, when a capital letter starts the next sentence.
# A rare wrong cut ("U.S. Army") does no harm: the pieces are shown together.
_SENTENCE_GAP = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z\"“(])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d{4}")
_CITATION = re.compile(r"\^\[[^\]]*\]")
_YEAR = re.compile(r"^(1[89]|20)\d\d$")
_STOPWORDS = {
    "that", "this", "with", "from", "have", "were", "been", "they", "their",
    "which", "when", "would", "about", "after", "also", "into", "more", "than",
    "said", "says", "will", "other", "there", "these", "those", "such", "while",
    "where", "before", "under", "over", "some", "only", "then", "them", "being",
}  # fmt: skip
NOT_SHOWN = "[… {n} {what} not shown …]"


@dataclass
class _Passage:
    text: str
    paragraph: int  # passages cut from one long paragraph share this
    heading: bool = False
    lead: bool = False


def _terms(text: str) -> Dict[str, float]:
    """Content words of ``text`` with a weight: years 3, capitalised names 2, else 1."""
    terms: Dict[str, float] = {}
    for word in _WORD.findall(_CITATION.sub(" ", text)):
        key = word.lower()
        if _YEAR.match(word):
            weight = 3.0
        elif len(word) < 4 or key in _STOPWORDS:
            continue
        else:
            weight = 2.0 if word[0].isupper() else 1.0
        terms[key] = max(terms.get(key, 0.0), weight)
    return terms


def _blocks(text: str) -> List[List[str]]:
    """Split into blocks: a heading line, a list item, or a paragraph's lines."""
    blocks: List[List[str]] = []
    current: List[str] = []

    def close() -> None:
        if current:
            blocks.append(list(current))
            current.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            close()
        elif _HEADING.match(stripped):
            close()
            blocks.append([line])
        elif stripped[:2] in ("- ", "* ") or re.match(r"\d+[.)] ", stripped):
            close()
            current.append(line)
        else:
            current.append(line)
    close()
    return blocks


def _passages(text: str, max_chars: int) -> List[_Passage]:
    """The profile as passages; a block over ``max_chars`` is cut into sentence runs."""
    passages: List[_Passage] = []
    seen_heading = False
    for index, block in enumerate(_blocks(text)):
        joined = "\n".join(block)
        if _HEADING.match(joined.strip()):
            # a heading on the first line ("### Overview") still opens the lead
            seen_heading = seen_heading or any(not p.heading for p in passages)
            passages.append(_Passage(joined, index, heading=True))
            continue
        lead = not seen_heading
        if len(joined) <= max_chars:
            passages.append(_Passage(joined, index, lead=lead))
            continue
        run: List[str] = []
        for sentence in _SENTENCE_GAP.split(joined):
            if run and len(" ".join(run + [sentence])) > max_chars:
                passages.append(_Passage(" ".join(run), index, lead=lead))
                run = []
            run.append(sentence)
        if run:
            passages.append(_Passage(" ".join(run), index, lead=lead))
    return passages


def _scores(passages: List[_Passage], article_text: str) -> List[float]:
    """How much each passage shares with the article, rarer shared words counting more."""
    wanted = _terms(article_text)
    passage_terms = [set(_terms(p.text)) for p in passages]
    counts: Dict[str, int] = {}
    for terms in passage_terms:
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
    total = max(1, len(passages))
    scores = []
    for passage, terms in zip(passages, passage_terms):
        shared: Set[str] = terms & set(wanted)
        raw = sum(wanted[t] * math.log(1 + total / counts[t]) for t in shared)
        scores.append(raw / math.sqrt(max(len(passage.text), 100)))
    return scores


def _render(passages: List[_Passage], keep: List[bool], what: str) -> str:
    parts: List[str] = []
    hidden = 0
    previous = -1  # index of the last kept passage
    for i, (passage, kept) in enumerate(zip(passages, keep)):
        if not kept:
            hidden += 1
            continue
        if hidden:
            parts.append(
                NOT_SHOWN.format(n=hidden, what=what if hidden > 1 else "passage")
            )
            hidden = 0
            parts.append(passage.text)
        elif (
            0 <= previous == i - 1 and passages[previous].paragraph == passage.paragraph
        ):
            parts[-1] += " " + passage.text  # the next sentences of the same paragraph
        else:
            parts.append(passage.text)
        previous = i
    if hidden:
        parts.append(NOT_SHOWN.format(n=hidden, what=what if hidden > 1 else "passage"))
    return "\n\n".join(parts)


def profile_view(text: str, article_text: str, budget: int) -> str:
    """``text`` cut to about ``budget`` characters, keeping what ``article_text`` is about.

    Returns ``text`` itself when it fits or ``budget`` is 0 or less (off).
    """
    if budget <= 0 or len(text) <= budget:
        return text
    passages = _passages(text, max(200, budget // 8))
    marker = len(NOT_SHOWN.format(n=99, what="passages")) + 2
    # Headings first, each with room for the gap marker after it, up to a
    # third of the budget (a profile with 40 short sections would otherwise
    # be all headings and markers)
    keep = [False] * len(passages)
    used = 0
    for i, passage in enumerate(passages):
        cost = len(passage.text) + 2 + marker
        if passage.heading and used + cost <= budget // 3:
            keep[i] = True
            used += cost
    # The opening text says what the entity is: keep its start
    lead_used = 0
    for i, passage in enumerate(passages):
        if passage.heading and not lead_used:
            continue
        if not passage.lead or lead_used + len(passage.text) > budget // 4:
            break
        keep[i] = True
        lead_used += len(passage.text) + 2
    used += lead_used
    scores = _scores(passages, article_text)
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    for i in order:
        cost = len(passages[i].text) + 2 + marker
        if keep[i] or used + cost > budget:
            continue
        keep[i] = True
        used += cost
    return _render(passages, keep, "passages")


def trimmed_note(full_text: str, shown_text: str) -> str:
    """One line telling the model that part of the profile was left out, or ''."""
    if shown_text == full_text:
        return ""
    return (
        f"(This profile is long: {len(full_text):,} characters. Only its opening, "
        "its headings and the parts closest to the new article are shown; "
        "'not shown' lines mark the gaps. The rest is kept as it is. Anchor "
        "edits only on sentences or headings you can see.)\n\n"
    )


def embedding_text(text: str) -> str:
    """What to embed for a profile: a hub's opening, headings and first passages.

    The embedding model only reads the first ~8k tokens anyway; this stops
    sending the rest of a 50k-character profile on every update.
    """
    return profile_view(text, "", PROFILE_VIEW_CHARS)
