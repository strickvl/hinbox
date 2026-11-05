"""Code checks on the edit-list updater's edits, before they are applied.

The edit-list updater (``profile_edits.py``) finds the article's main fact far
more often than the fact-list updater (dev claim labels, Luna: 34 of 38 cases
against 21-25). Its weak spots were on long profiles: in a 60-article Qwen run
it added 320 sentences that restated one already in the profile, and 312
"sources disagree" sentences, nearly all about two different things. Both can
be checked without a model, so each new sentence of an ``add`` edit is checked
here:

- **Repeat**: every word of it (filler such as "the article states" aside)
  is in one existing sentence, it gives no year, month or number that sentence lacks, and
  it has as many negations ("not", "never") as that sentence.
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

MIN_STEMS = 3  # shorter lines (headings, "Tags:") aren't checked
_CITATION = re.compile(r"\^\[[^\]]*\]")
# The same wording the profile-update eval counts as "sources disagree"
_DISPUTE = re.compile(
    r"\b(disagree\w*|differs?\b|contradict\w*|discrepanc\w*|inconsistent with"
    r"|conflicts? with|conflicting (?:accounts|reports|dates|figures|sources))",
    re.I,
)
_SOURCES = re.compile(r"\b(sources?|reports?|articles?|accounts?|records?)\b", re.I)
# Only the profile's own voice comparing sources is checked: "sources
# disagree", "conflicting reports", "differs from the earlier report". A
# disagreement the article reports ("his lawyers contradicted reports that he
# confessed") is news and is never dropped as a dispute
_SRC = r"(?:sources?|reports?|accounts?|articles?|records?|profile)"
_NARRATOR = re.compile(
    rf"\b{_SRC}\s+(?:disagree|differ|conflict|vary|contradict)\w*"
    r"|\bconflicting (?:accounts|reports|dates|figures|sources)"
    r"|\bdiscrepanc\w*"
    rf"|\b(?:differs?|disagrees?|contradicts?|contradicting|inconsistent|conflicts?)"
    rf"\s+(?:with|from)?\s*(?:the|an?|this)?\s*(?:earlier|previous|prior|other|"
    rf"another|existing|new)\s+{_SRC}",
    re.I,
)
# Capitalised words after the first: names. "consulted Judge Spath" isn't a
# repeat of "consulted Judge Pohl", and the stems drop short names
_NAME = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")


def _claim(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


# Words that carry no fact, and the wrapping models put round a restated one
# ("The article states that ...")
_FILLER = set(
    "the a an of in on at to for and or by with as is was were be been being "
    "has had have that this these those from its it his her their they he she "
    "which who whom also there then than into about after before while "
    "article articles report reports reported states stated notes noted "
    "according describes described says said new profile".split()
)


def _words(text: str) -> set:
    """Every word of the sentence but filler, cut to 5 letters ("detainees"
    and "detained" match). Short words count: "won" and "lost", "Pohl" and
    "Spath" are what separate two otherwise identical sentences."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:5] for w in words if w not in _FILLER}


def is_repeat(sentence: str, known: List[str]) -> bool:
    """Every word of the sentence is in one known sentence (filler aside), it
    adds no value, and it has as many negations.

    Not "most words": in review, "lost the appeal" against "won the appeal",
    "consulted Judge Spath" against "Judge Pohl" and "could not attend"
    against "could attend" all shared over 80% of their long words.
    """
    words = _words(_claim(sentence))
    if len(_stems(_claim(sentence))) < MIN_STEMS:
        return False
    negations, names = _negations(sentence), set(_NAME.findall(_claim(sentence)))
    for other in known:
        if (
            words <= _words(_claim(other))
            and _negations(other) == negations
            # Any capitalised word of the other counts, its first one too
            and names <= set(re.findall(r"\b[A-Z][a-z]{2,}\b", _claim(other)))
            # Against this sentence only: a year the profile gives elsewhere
            # ("filed an appeal in 2014") still makes "in 2014 the judge
            # scheduled" a change to "in 2013 the judge scheduled"
            and not adds_values(_claim(sentence), _claim(other))
        ):
            return True
    return False


# Short words the stems leave out, so "could not attend" and "could attend"
# would otherwise look the same
_NEGATION = re.compile(r"\b(not|no|never|neither|nor|without|\w+n't)\b", re.I)


def _negations(sentence: str) -> int:
    return len(_NEGATION.findall(_claim(sentence)))


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
    if not _NARRATOR.search(claim):
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
            if is_repeat(sentence, known):
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
