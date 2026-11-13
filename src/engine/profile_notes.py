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
_PROFILE = (
    r"the\s+(?:existing\s+|current\s+|earlier\s+)?profile(?P<s>['’]s)?\b(?!\s+of\b)"
)
_ABOUT_PROFILE = re.compile(rf"\b{_PROFILE}", re.I)
# ... and neither is a sentence about a document called a profile ("The
# profile also acknowledged that he answered the debriefers", of a review
# board's file on a captive). A note sets the profile against the article:
# "conflicting with the profile's count", "aligns with the profile", ...
_AGAINST = re.compile(
    r"\b(?:conflict\w*|contradict\w*|differ\w*|disagree\w*|inconsistent|at odds"
    r"|unlike|contrary to|in contrast (?:to|with)|rather than|instead of"
    r"|consistent with|in line with|match\w*|agree\w*|align\w*|distinct from"
    r"|compared (?:to|with)|versus)\b[^.;]{0,160}?\b" + _PROFILE.replace("?P<s>", "?:"),
    re.I,
)
# ... or names it beside a source in a contrast: "the article states X,
# whereas the profile states Y"
_SOURCE = re.compile(
    r"\b(?:the|this|an?|new|another)\s+(?:[\w.,]+\s+){0,3}?(?:article|report|story)\b"
    r"|\bsources?\b",
    re.I,
)
# ... or says what it lacks: "The profile omits that Reed led the camp"
_LACKS = re.compile(
    r"\b" + _PROFILE.replace("?P<s>", "?:") + r"\s+(?:omits|lacks|leaves\s+out"
    r"|is\s+missing|(?:does\s+not|doesn't|did\s+not|fails\s+to)\s+(?:mention"
    r"|include|state|note|say|specify|give|reflect))\s+(?P<that>that\s+)?(?P<side>.+)$",
    re.I | re.S,
)
_CONTRAST = re.compile(r"\b(?:while|whereas|but|although|though|however)\b|[;:]", re.I)
# The article's half says how things stand, not what happened
_NOW = re.compile(
    r"\b(?:is|are|has|have|holds|houses|remains?|commands|consists|leads|heads)\b"
)


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
    """The sentence talks about the profile it is being added to, setting
    it against the article."""
    if not _ABOUT_PROFILE.search(claim):
        return False
    if _AGAINST.search(claim) or _LACKS.search(claim):
        return True
    return bool(_SOURCE.search(claim) and _CONTRAST.search(claim))


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
    claim = _CITATION.sub("", sentence).strip()
    found = article_side(claim)
    if found is None and (m := _LACKS.search(claim)):
        # What the profile lacks is the article's fact
        side = m.group("side").rstrip(".;, ")
        found = ("says" if m.group("that") else "notes", side, None)
    if found is None:
        return None
    verb, side, dated = found
    cites = "".join(dict.fromkeys(_CITATION.findall(sentence)))
    held_then = verb == "notes" or is_time_varying(side) or _NOW.search(side)
    if article_date and held_then:
        return as_update(verb, side, dated, article_date, cites)
    said = bool(dated) and dated.lower() not in side.lower()
    if verb != "says" or said:
        month = f"{dated} " if said else ""
        article = "An" if re.match(r"[AEIOU]|8|1[18]\b", month) else "A"
        says = "states that" if verb == "says" else verb
        text = f"{article} {month}report {says} {side}"
    else:
        text = side[0].upper() + side[1:]
    return f"{text.rstrip('.')}.{cites}"
