"""A sentence about the profile itself never goes into a profile.

Asked for edits, a model sometimes writes its comparison of the article with
the profile as a new sentence: "The article states there are 104 captives,
conflicting with the profile's count of 93 captives." In the 12 Nov full
local run 1,701 of 78,436 cited sentences mention "the profile" this way, and
the citation check marked two thirds of them partly or not supported (an
article can't back what "the profile" says). Read by hand, 30 of 30 sampled
ones were such notes, most with a real fact from the article in one half.
Many compare a count with an earlier count (93 captives, then 104), which is
a later state, not a disagreement.

``edit_guards.py`` already tells real disputes from later states and
different events for sentences worded "sources disagree"; these notes are
worded differently ("conflicting with the profile's count", "whereas the
profile states") and skipped every check. So a new sentence that mentions the
profile is handled as that kind of sentence (``guard_edits()``):

- a count, schedule, post or status from an article of another date becomes
  the article's own dated sentence (``later_update()``);
- otherwise the dispute check's model call decides: a later state or another
  event is replaced by the article's fact alone, a restatement is dropped,
  and a real conflict is kept with "the profile" reworded as "another report"
  (``reworded()``);
- without that call, the article's half is kept as a plain sentence
  (``plain_side()``); a note that says outright that sources disagree, or
  has no half that stands alone, stays, reworded;
- a note with no fact from the article ("The article does not mention his
  age, whereas the profile ...") is dropped (``is_empty_note()``).
"""

import datetime as dt
import re
from typing import Optional

from src.engine.later_updates import article_side, as_update, is_time_varying

_CITATION = re.compile(r"\^\[[^\]]*\]")
# "the profile of the detainees" and "a high-profile case" are not notes
_ABOUT_PROFILE = re.compile(
    r"\bthe\s+(?:existing\s+|current\s+|earlier\s+)?profile(?P<s>['’]s)?\b(?!\s+of\b)",
    re.I,
)
# The article's half says how things stand, not what happened
_NOW = re.compile(
    r"\b(?:is|are|has|have|holds|houses|remains?|commands|consists|leads|heads)\b"
)


# The half opens with a word that pointed at the half now gone
_LEANS = re.compile(r"^(?:it|this|that|they|these|those|both|such)\b", re.I)
# The note says the article has nothing to add, or is the model thinking
_EMPTY = re.compile(
    r"\b(?:article|report|story|source)\s+(?:does\s+not|doesn't|did\s+not|never)\s+"
    r"(?:mention|state|specify|say|give|address|confirm|include|provide|report)"
    r"|\bno (?:conflict|edit|change)(?:\s+is)?(?:\s+needed)?\b[,)]"
    r"|\(no conflict|\bno edit needed\b",
    re.I,
)
# It says outright that sources disagree: kept as a dispute, not as a fact
_SAYS_DISAGREE = re.compile(
    r"\b(?:sources?|reports?|accounts?|records?)\s+(?:disagree|differ|conflict|vary)"
    r"|\bconflicting (?:accounts|reports|dates|figures|sources)\b"
    r"|\bdiscrepanc|\bdisagree(?:ment)?\b.{0,40}\b(?:exists?|on|about|regarding)\b",
    re.I,
)


def is_profile_note(claim: str) -> bool:
    """The sentence talks about the profile it is being added to."""
    return bool(_ABOUT_PROFILE.search(claim))


def is_empty_note(claim: str) -> bool:
    """A note with no fact from the article in it: "The article does not
    mention his age, whereas the profile ...", "(no conflict, ...)"."""
    return bool(_EMPTY.search(claim))


def says_disagree(claim: str) -> bool:
    return bool(_SAYS_DISAGREE.search(claim))


def reworded(sentence: str) -> str:
    """A real conflict, kept, without the profile as a source of its own:
    "the profile states" -> "another report states"."""

    def other(m: "re.Match[str]") -> str:
        start = sentence[: m.start()].rstrip()
        first = not start or start[-1] in ".!?:;" and m.group(0)[0] == "T"
        return ("Another" if first else "another") + " report" + (m.group("s") or "")

    return _ABOUT_PROFILE.sub(other, sentence)


def plain_side(sentence: str, article_date: Optional[dt.date] = None) -> Optional[str]:
    """The article's half of a note as a sentence of its own, with the
    note's citations, or None when no half stands alone. A count, schedule,
    post or status, or anything in the present tense, is dated by the article
    when its date is known ("As of March 2016, there are 104 captives."),
    since it held then, not always; so is a half that needs a source in
    front ("An April 2013 report notes 41 captives at the camp.")."""
    found = article_side(_CITATION.sub("", sentence).strip())
    if found is None:
        return None
    verb, side, dated = found
    if _LEANS.match(side):
        return None
    cites = "".join(dict.fromkeys(_CITATION.findall(sentence)))
    held_then = verb == "notes" or is_time_varying(side) or _NOW.search(side)
    if article_date and held_then:
        return as_update(verb, side, dated, article_date, cites)
    if verb != "says" or dated:
        month = f"{dated} " if dated and dated.lower() not in side.lower() else ""
        article = "An" if re.match(r"[AEIOU]|8|1[18]\b", month) else "A"
        says = "states that" if verb == "says" else verb
        text = f"{article} {month}report {says} {side}"
    else:
        text = side[0].upper() + side[1:]
    return f"{text.rstrip('.')}.{cites}"
