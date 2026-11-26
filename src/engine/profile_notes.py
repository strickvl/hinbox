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
from typing import List, Optional

from src.engine.later_updates import (
    article_side,
    as_update,
    is_time_varying,
    relative_time,
)

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
# ... and "discrepancy with the profile's age of 38" is a note too. Any
# other mention is one as well, unless the profile is a document someone
# wrote about the entity (``_DOCUMENT``)
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
# A document called a profile, in the past tense or with its own name: "the
# profile also assessed him as ...", "an intelligence profile prepared for
# his file review", "the board released the profile, which called him ..."
_DOCUMENT = re.compile(
    r"\bintelligence\s+profile\b|\bprofile\s+prepared\b"
    r"|\b(?:released|issued|published|prepared|wrote|read)\s+the\s+profile\b"
    r"|\bprofile,?\s+(?:(?:also|that|which)\s+)?(?:acknowledged|assessed|dismissed"
    r"|described|judged|concluded|cast|characteri[sz]ed|portrayed|depicted"
    r"|determined|recommended|called)\b"
    r"|\bprofile\s+also\s+(?:states|notes|says|adds)\b",
    re.I,
)
# The profile as the sentence's subject: "The profile states that ...",
# "The profile's claim that ..."
_OURS_FIRST = re.compile(
    r"^(?:(?:however|but|yet)[,\s]+)?" + _PROFILE.replace("?P<s>", "?:"), re.I
)
_OURS_SAYS = re.compile(
    _OURS_FIRST.pattern + r"\s+(?:states|says|notes|claims|mentions|reports"
    r"|indicates)\s+(?:that\s+)?(?P<side>.+)$",
    re.I | re.S,
)
_OTHER_SOURCE = re.compile(r"\b(?:article|report|story|sources?)\b", re.I)
_OTHER_HALF = re.compile(r"\b(?:article|report|story|source|profile)\b", re.I)
# The article's half says how things stand, not what happened
# ("have been closed once" is something that happened)
_NOW = re.compile(
    r"\b(?:is|are|holds|houses|remains?|commands|consists|leads|heads)\b"
    r"|\b(?:has|have)\b(?!\s+(?:been|since)\b)"
)


# The note says the article has nothing to add, or is the model thinking
_EMPTY = re.compile(
    r"\b(?:article|report|story|source)\s+(?:does\s+not|doesn't|did\s+not|never)\s+"
    r"(?:mention|state|specify|say|give|address|confirm|include|provide|report)"
    r"|\bno (?:conflict|edit|change)(?:\s+is)?(?:\s+needed)?\b[,)]"
    r"|\(no conflict|\bno edit needed\b"
    # A remark on the sentence before: "This contradicts the profile's claim"
    r"|^(?:this|that|it)\s+(?:\w+\s+)?(?:contradicts|conflicts|differs|disagrees"
    r"|is\s+(?:inconsistent|at\s+odds|consistent)|aligns|matches|confirms)\b"
    # The model thinking aloud: "Wait, the profile sentence *is* the
    # article's sentence.", "The sentence with 154 is already in the profile?"
    r"|^wait\b|\?\s*$|\*\w+\*|\balready\s+in\s+the\s+profile\b",
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
    if _SOURCE.search(claim) and _CONTRAST.search(claim):
        return True
    return not _DOCUMENT.search(claim)


def is_empty_note(claim: str, cited: bool = True) -> bool:
    """A note with no fact from the article in it: "The article does not
    mention his age, whereas the profile ...", "(no conflict, ...)", "This
    contradicts the profile's claim that ...", the model thinking aloud, or
    the profile alone with no article ("The profile has 116 and 122."; with
    no citation, also "The profile states that Reed led the camp.")."""
    claim = _CITATION.sub("", claim).strip()
    if _EMPTY.search(claim):
        return True
    # "the profile states there are 172 captives, while the article states
    # there are 172 captives": both halves say the same
    if (found := article_side(claim)) and claim.lower().count(found[1].lower()) > 1:
        return True
    if not _OURS_FIRST.match(claim) or _OTHER_SOURCE.search(claim):
        return False
    if _DOCUMENT.search(claim):
        return False
    return not cited or _own_fact(claim) is None


def _own_fact(claim: str) -> Optional[str]:
    """The fact in a note that names the profile and no article: what the
    profile is said to lack or to state ("The profile states that Reed led
    the camp"), or what follows a semicolon ("The profile wrongly calls it
    a firm; Reed was employed by the airline")."""
    if (m := _LACKS.search(claim)) and m.group("that"):
        # What the profile lacks is the article's fact ("omits that Reed
        # led the camp"; "does not specify the month" names no fact)
        side = m.group("side")
    elif m := _OURS_SAYS.match(claim):
        side = m.group("side")
    elif _OURS_FIRST.match(claim) and ";" in claim:
        side = claim.rsplit(";", 1)[1]
    else:
        return None
    side = side.strip().rstrip(".;, ")
    if len(side.split()) < 4 or _OTHER_HALF.search(side):
        return None
    return side


# All the note says of the profile is what it lacks: "while the profile
# does not specify a cost, the article states ...". Nothing disagrees
_ONLY_LACKS = re.compile(
    r"\bprofile\s+(?:does\s+not|doesn't|did\s+not|fails\s+to)\s+(?:mention"
    r"|include|state|note|say|specify|give|reflect|address)\b",
    re.I,
)
_PROFILE_HAS = re.compile(
    r"\bprofile(?:['’]s\b|\s+(?:\w+ly\s+)?(?:states|says|notes|claims|cites"
    r"|reports|gives|lists|describes|identifies|mentions|indicates|puts|has"
    r"|asserts|implies|records)\b)",
    re.I,
)


def says_disagree(claim: str) -> bool:
    """The note says outright that sources disagree, and gives the profile
    a side of its own."""
    if _ONLY_LACKS.search(claim) and not _PROFILE_HAS.search(claim):
        return False
    return bool(_SAYS_DISAGREE.search(claim))


def reworded(sentence: str) -> str:
    """A real conflict, kept, without the profile as a source of its own:
    "the profile states" -> "another report states"."""

    def other(m: "re.Match[str]") -> str:
        start = sentence[: m.start()].rstrip()
        first = not start or start[-1] in ".!?:;" and m.group(0)[0] == "T"
        return ("Another" if first else "another") + " report" + (m.group("s") or "")

    new = _ABOUT_PROFILE.sub(other, sentence)
    # "the profile notes the profile also notes this fact"
    return re.sub(
        r"\b(another report\s+\w+)\s+another report\s+(?:also\s+)?\w+\b",
        r"\1",
        new,
        flags=re.I,
    )


# The side still has the article as its subject
_NARRATOR = re.compile(
    r"^the\s+(?:new\s+|latest\s+|current\s+)?(?:article|story)\s+(?P<verb>\w+s)\s+",
    re.I,
)
# "... overturned the conviction^[old].^[new]": the model copied the old
# sentence's citation inside and put its own after the full stop
_OWN_CITES = re.compile(r"\^\[[^\]]*\]\.((?:\s*\^\[[^\]]*\])+)\s*$")


def _own_citations(sentence: str) -> List[str]:
    own = _OWN_CITES.search(sentence)
    return _CITATION.findall(own.group(1) if own else sentence)


def plain_side(sentence: str, article_date: Optional[dt.date] = None) -> Optional[str]:
    """The article's half of a note as a sentence of its own, with the
    note's citations, or None when no half stands alone. A count, schedule,
    post or status, or anything in the present tense, is dated by the article
    when its date is known ("As of March 2016, there are 104 captives."),
    since it held then, not always; so is a half that needs a source in
    front ("An April 2013 report notes 41 captives at the camp.")."""
    claim = _CITATION.sub("", sentence).strip()
    found = article_side(claim)
    if found is None and (side := _own_fact(claim)):
        found = ("says", side, None)
    if found is None:
        return None
    verb, side, dated = found
    if verb == "says" and (m := _NARRATOR.match(side)):
        # "The new article refers to her as ...": "A report refers to ..."
        verb, side = m.group("verb"), side[m.end() :]
    cites = "".join(dict.fromkeys(_own_citations(sentence)))
    held_then = verb == "notes" or is_time_varying(side) or _NOW.search(side)
    if article_date and (held_then or relative_time(side)):
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
