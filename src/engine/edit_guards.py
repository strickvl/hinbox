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
  sentence is about the same thing (``same_subject()``) with a value the new
  sentence doesn't share. Dropped.

``replace`` edits are left alone: they rewrite one sentence and keep its
citations. An edit whose sentences are all dropped is removed.
"""

import re
from typing import Any, Dict, List, Tuple

from src.engine.fact_list import _stems, adds_values, same_subject
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


def is_invented_dispute(sentence: str, known: List[str]) -> bool:
    """Says the sources disagree, but no known sentence on the same thing has
    a value this one adds ("opened in 2003" against "opened in 2002")."""
    claim = _claim(sentence)
    if not (_DISPUTE.search(claim) and _SOURCES.search(claim)):
        return False
    # The dispute wording itself says nothing about what is disputed
    topic = _SOURCES.sub(" ", _DISPUTE.sub(" ", claim))
    return not any(
        same_subject(topic, _claim(o)) and adds_values(claim, _claim(o)) for o in known
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
