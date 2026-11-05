"""Code checks on the edit-list updater's edits, before they are applied.

The edit-list updater (``profile_edits.py``) finds the article's main fact far
more often than the fact-list updater (dev claim labels, Luna: 34 of 38 cases
against 21-25). Its weak spots were on long profiles: in a 60-article Qwen run
it added 320 sentences that restated one already in the profile, and 312
"sources disagree" sentences, nearly all about two different things. Both can
be checked without a model, so each new sentence of an ``add`` edit is checked
here:

- **Repeat**: nearly all its content words (``REPEAT_OVERLAP``) are in one
  existing sentence, and it gives no year, month or number the profile lacks.
  Dropped.
- **Invented dispute**: it says sources or reports disagree, but no existing
  sentence that shares a topic word gives a different year, or a different
  month or count for the same year (``_disputes_value()``). Dropped.

``replace`` edits are left alone: they rewrite one sentence and keep its
citations. An edit whose sentences are all dropped is removed.
"""

import re
from typing import Any, Dict, List, Tuple

from src.engine.date_disputes import _PUBLISHED
from src.engine.fact_list import _stems, _values, adds_values
from src.engine.profile_edits import ProfileEdit
from src.evals.profile_cases import _sentences

REPEAT_OVERLAP = 0.8  # share of the new sentence's word stems in one old sentence
MIN_STEMS = 3  # shorter lines (headings, "Tags:") aren't checked
_CITATION = re.compile(r"\^\[[^\]]*\]")
# The same wording the profile-update eval counts as "sources disagree"
_DISPUTE = re.compile(
    r"\b(disagree\w*|differs?\b|contradict\w*|discrepanc\w*|inconsistent with"
    r"|conflicts? with|conflicting (?:accounts|reports|dates|figures|sources))",
    re.I,
)
_SOURCES = re.compile(r"\b(sources?|reports?|articles?|accounts?|records?)\b", re.I)


def _claim(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


def is_repeat(sentence: str, known: List[str], profile_text: str) -> bool:
    """Nearly every content word is in one known sentence, and it adds no value."""
    stems = _stems(_claim(sentence))
    if len(stems) < MIN_STEMS:
        return False
    for other in known:
        shared = len(stems & _stems(_claim(other))) / len(stems)
        if shared >= REPEAT_OVERLAP and not adds_values(
            _claim(sentence), _claim(other), profile_text
        ):
            return True
    return False


def _disputes_value(claim: str, other: str) -> bool:
    """``claim`` gives a different year, or a different month or count for
    the same year, than ``other`` does, and they share a topic word.

    A count that changes between articles (59 hunger strikers, then 100,
    then 106) or an age a year later is a later state, not a dispute: in the
    60-article Qwen run those were most of the "sources disagree" sentences
    a plain "adds a value" test let through. A planted dispute changes one
    year ("July 2001" against "a July 2002 firefight"), and shares as little
    as one topic word, so one is enough.
    """
    (years, months, counts), (o_years, o_months, o_counts) = (
        _values(claim),
        _values(other),
    )
    if not _stems(claim) & _stems(other):
        return False
    return bool(
        (years - o_years and o_years)
        or ((months - o_months and o_months) or (counts - o_counts and o_counts))
        and years & o_years
    )


def is_invented_dispute(sentence: str, known: List[str]) -> bool:
    """Says the sources disagree, but no known sentence gives a different year
    (or a different month or count for the same year) for something it mentions."""
    claim = _claim(sentence)
    if not (_DISPUTE.search(claim) and _SOURCES.search(claim)):
        return False
    # The dispute wording itself says nothing about what is disputed, and a
    # source's own date ("the August 14, 2009 article describes Khadr as 22")
    # is when it was written: on dev, Luna's two kept false disputes differed
    # from the profile only in that year
    topic = _SOURCES.sub(" ", _DISPUTE.sub(" ", _PUBLISHED.sub(" ", claim)))
    return not any(
        _disputes_value(topic, _PUBLISHED.sub(" ", _claim(o))) for o in known
    )


def guard_edits(
    text: str, edits: List[ProfileEdit]
) -> Tuple[List[ProfileEdit], Dict[str, Any]]:
    """The edits with repeated and invented-dispute sentences taken out."""
    known = [s for s in _sentences(text) if not s.startswith("#")]
    report: Dict[str, Any] = {"repeats": [], "disputes": []}
    kept: List[ProfileEdit] = []
    for edit in edits:
        if edit.op != "add":
            kept.append(edit)
            continue
        new_text, facts_left = edit.text, 0
        for sentence in _sentences(edit.text):
            if sentence.startswith("#"):
                continue
            if is_repeat(sentence, known, text):
                report["repeats"].append(sentence[:200])
            elif is_invented_dispute(sentence, known):
                report["disputes"].append(sentence[:200])
            else:
                facts_left += 1
                known.append(sentence)  # a later edit can't add it again
                continue
            new_text = new_text.replace(sentence, "", 1)
        if new_text == edit.text:
            kept.append(edit)
        elif facts_left:
            tidy = re.sub(r"[ \t]{2,}", " ", new_text).strip()
            kept.append(edit.model_copy(update={"text": tidy}))
    return kept, report
