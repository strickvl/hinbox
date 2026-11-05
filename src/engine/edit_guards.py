"""Code checks on the edit-list updater's edits, before they are applied.

The edit-list updater (``profile_edits.py``) finds the article's main fact far
more often than the fact-list updater (dev claim labels, Luna: 34 of 38 cases
against 21-25). Its weak spots were on long profiles: in a 60-article Qwen run
it added 320 sentences that restated one already in the profile, and 312
"sources disagree" sentences, nearly all about two different things. Both can
be checked without a model, so each new sentence of an ``add`` edit is checked
here:

- **Repeat**: every word of it (filler such as "the article states" aside)
  is in one existing sentence, in the same order, it gives no year, month or number that sentence lacks, and
  it has as many negations ("not", "never") and the same hedges ("could",
  "expected", "allegedly") as that sentence.
  Dropped.
- **Invented dispute**: it says sources or reports disagree, but no existing
  sentence that shares a topic word gives a different year, or a different
  month or count for the same year (``_disputes_value()``). Dropped.

``replace`` edits are left alone: they rewrite one sentence and keep its
citations. New sentences are checked against the profile as the
replacements leave it. An edit whose sentences are all dropped is removed.
"""

import re
from typing import Any, Dict, List, Tuple

from src.engine.date_disputes import _PUBLISHED
from src.engine.fact_list import _MONTH, _stems, _values, adds_values
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
    "which who whom also there "
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
    hedges = _hedges(_claim(sentence))
    for other in known:
        if (
            words <= _words(_claim(other))
            and _negations(other) == negations
            and _hedges(_claim(other)) == hedges
            # Any capitalised word of the other counts, its first one too
            and names <= set(re.findall(r"\b[A-Z][a-z]{2,}\b", _claim(other)))
            # Against this sentence only: a year the profile gives elsewhere
            # ("filed an appeal in 2014") still makes "in 2014 the judge
            # scheduled" a change to "in 2013 the judge scheduled"
            and not adds_values(_claim(sentence), _claim(other))
            and _same_order(_claim(sentence), _claim(other))
        ):
            return True
    return False


# A year in front ("In 2006, Camp 6 opened ...") is the same fact as one at
# the end ("Camp 6 opened in 2006"), so it is left out of the order check
_LEADING_YEAR = re.compile(r"^\s*(?:in|by|since|during)\s+(?:\w+\s+)?\d{4},?\s+", re.I)


def _order(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", _LEADING_YEAR.sub("", text).lower())
    return [w[:5] for w in words if w not in _FILLER]


def _same_order(claim: str, other: str) -> bool:
    """The words come in the same order as in the old sentence, numbers
    included, so "Nashiri questioned the judge" isn't a repeat of "the judge
    questioned Nashiri", nor "17 in 2013 and 18 in 2014" of "18 in 2013 and
    17 in 2014". A year moved to the front still matches. A passive sentence
    ("the appeal was won by his lawyers") can't be checked this way and is
    let through. A repeat with its clauses swapped is kept: a repeat left in
    costs less than a new fact dropped."""
    if re.search(r"\b(?:was|were|been|being|is|are)\s+\w+\s+by\b", claim, re.I):
        return True
    words, others = _order(claim), iter(_order(other))
    return all(w in others for w in dict.fromkeys(words))


# Short words the stems leave out, so "could not attend" and "could attend"
# would otherwise look the same
_NEGATION = re.compile(r"\b(not|no|never|neither|nor|without|\w+n't)\b", re.I)


def _negations(sentence: str) -> int:
    return len(_NEGATION.findall(_claim(sentence)))


# Words that make a fact tentative: "the hearing could resume in 2014" isn't
# a repeat of "the hearing resumed in 2014" (the 5-letter cut makes "resume"
# and "resumed" one word), it's the news that it happened
_HEDGE = re.compile(
    r"\b(could|may|might|would|will|shall|should|can|cannot|expected|expects?"
    r"|plans?|planned|intends?|intended|likely|unlikely|possibly|probably"
    r"|allegedly|reportedly|proposed|scheduled|set to|due to|hopes?|hoped)\b",
    re.I,
)


def _hedges(text: str) -> List[str]:
    return sorted(m.lower() for m in _HEDGE.findall(text))


# Capitalised words in the dispute wording itself, not names
_SKIP_NAMES = {"Sources", "Source", "Reports", "Another", "The", "This", "However"}
_SKIP_NAMES |= set("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())


# A name right after the same word on both sides: "Judge Spath" in place of
# "Judge Pohl". Just "each side has a capitalised word the other lacks" kept
# 48 false disputes in the 60-article Qwen run, e.g. "Dec. 4" against "does
# not specify", or "Judge Pohl denied" against "ruled for the defense"
_TITLED = re.compile(r"\b(\w+)\s+([A-Z][a-z]{2,})\b")


def _titled(text: str) -> set:
    return {
        (w.lower(), name)
        for w, name in _TITLED.findall(text)
        if not _MONTH.fullmatch(name) and name not in _SKIP_NAMES
    }


def _swaps_name(claim: str, other: str) -> bool:
    """Some word is followed by one name in the claim and by another, which
    the claim never gives, in ``other``."""
    mine, theirs = _titled(claim), _titled(other)
    my_names = {n for _, n in mine}
    their_names = {n for _, n in theirs}
    return any(
        w == v and a not in their_names and b not in my_names
        for w, a in mine
        for v, b in theirs
    )


def _disputes_value(claim: str, other: str) -> bool:
    """``claim`` gives a different year, or a different month or count for
    the same year, than ``other`` does, and they share a topic word; or,
    sharing two, a name in place of another or another undated count.

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
    shared = _stems(claim) & _stems(other)
    if not shared:
        return False
    if (years - o_years and o_years) or (
        ((months - o_months and o_months) or (counts - o_counts and o_counts))
        and years & o_years
    ):
        return True
    # On the same thing (two topic words): another name ("Judge Spath
    # presided" against "Judge Pohl presided"), or another count where
    # neither gives a year ("18 sailors killed" against "killed 17 sailors")
    if len(shared) < 2:
        return False
    swapped = _swaps_name(claim, other)
    undated = not years and not o_years
    return swapped or bool(undated and counts - o_counts and o_counts)


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
    # Checked against the profile as the replacements leave it: when an edit
    # replaces "captured in 2002 and charged in 2008" with the capture alone
    # and adds the charge separately, the added charge isn't a repeat
    for edit in edits:
        if edit.op == "replace":
            old = _claim(edit.anchor)
            known = [k for k in known if not old or old not in _claim(k)]
            known += [s for s in _sentences(edit.text) if not s.startswith("#")]
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
