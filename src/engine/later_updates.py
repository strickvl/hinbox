"""Tell a later update from a real dispute in a "sources disagree" sentence.

In the 60-article Qwen run, the "sources disagree" sentences that survived
``edit_guards.py`` were nearly all changes over time, not two sources
disagreeing. The military counted 24 hunger strikers on March 19, 2013 and 28
on March 25; the next article was written up as "Sources disagree on the
number of hunger strikers". Judge Pohl set the Cole trial for Sept. 2, 2014,
then moved it to Dec. 4; that became "Sources disagree on the trial date". A
reader of the profile then thinks the record is unsure about something that
simply changed.

A real dispute is two sources on the same point in time: the year Khadr
pleaded guilty, the date of a bombing. A later update is a newer article on a
thing that changes: a count of an ongoing state (hunger strikers, men fed by
tube), a schedule (a trial date, the next hearing), who holds a post (the
judge, the chief prosecutor), a status. So a dispute sentence is read as a
later update when both hold:

1. **It is about a thing that changes** (``is_time_varying()``): it names a
   count of an ongoing state or a schedule, or it names a post or a status
   and no one-off past event (born, captured, pleaded, killed...).
2. **The article is newer** than every source cited by the old sentences it
   disputes (``guard_edits()`` passes those in).

Such a sentence is rewritten to the article's side alone, dated by the
article ("As of April 2013, 59 captives were hunger strikers.^[id]"), and the
old sentence stays as it is, so the profile reads as a timeline with both
sources cited. ``article_side()`` finds the article's side in the wordings the
models use; when it can't, the sentence is left as it is, since dropping it
would lose the newer value.
"""

import datetime as dt
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_CITATION = re.compile(r"\^\[[^\]]*\]")
_CITED = re.compile(r"\^\[([^\]]+)\]")

# An ongoing state (hunger strikes, tube feedings, lockdown) ...
_STATE = re.compile(
    r"hunger[- ]?strik|\bon strike\b|tube[- ]?fe|force[- ]?fe|forced[- ]?feed"
    r"|nasogastric|\bfeedings?\b|lockdown|single[- ]cell|\bpopulation\b",
    re.I,
)
# ... a count of one ("the number of hunger strikers", "59 captives") ...
_COUNT = re.compile(
    r"\b(?:number|count|total|tally|figure)\b"
    r"|\b\d+\s+(?:of\s+(?:the\s+)?\d+\s+)?(?:captives|detainees|prisoners|men"
    r"|inmates|hunger strikers|strikers|people|were|are)\b",
    re.I,
)
# ... or a schedule: all change from week to week, so a newer value is news
_SCHEDULE = re.compile(
    r"\btrial (?:date|timeline|schedule|start)|\bschedul\w*|\bpostpon\w*"
    r"|\bpushed (?:back|to|the)|\bdelay\w*|\bresched\w*|\btimeline\b"
    r"|\bdeadline\b|\bnext (?:hearing|session)\b|\bset (?:for|to (?:begin|start))"
    r"|\bexpected to (?:begin|start|open|close|end|resume)|\bearliest\b",
    re.I,
)
# Who holds a post, or where something stands: changing too, unless the
# sentence is about a one-off past event
_POST = re.compile(
    r"\b(?:judge|commander|prosecutor|spokes(?:man|woman|person)|chief|director"
    r"|warden|secretary|envoy|counsel|presiding|rank)\b.*?"
    r"\b(?:identity|replac\w*|succeed\w*|took over|assigned|appointed|named"
    r"|current\w*|now|new)\b"
    r"|\b(?:identity|replac\w*|succeed\w*|took over|assigned|appointed)\b.*?"
    r"\b(?:judge|commander|prosecutor|spokes(?:man|woman|person)|chief|director"
    r"|warden|secretary|envoy|counsel)\b"
    r"|\bstatus\b|\bstill\b|\bcurrently\b|\bas of\b",
    re.I,
)
# One-off past events: a different year for one of these is a real dispute
_ONE_OFF = re.compile(
    r"\b(?:born|birth|captur\w*|arrest\w*|killed|death toll|died|bomb(?:ing|ed)"
    r"|attack\w*|plead\w*|pleaded|convict\w*|sentenced|founded|nationality"
    r"|charged|moved to|transferred|arrived|opened|built|released|began|begun"
    r"|started|ended|launched|first|start of|end of)\b",
    re.I,
)


def is_time_varying(text: str) -> bool:
    """The sentence is about a count, schedule, post or status that changes.

    When it names a one-off event ("the hunger strike began on March 11",
    "captured in 2002"), only a count of an ongoing state or a schedule
    still counts: the day a strike began can't change, the number on strike
    can, and a death toll is about the attack."""
    text = _CITATION.sub("", text)
    if _ONE_OFF.search(text):
        state_count = _STATE.search(text) and _COUNT.search(text)
        return bool(state_count or _SCHEDULE.search(text))
    return bool(_STATE.search(text) or _SCHEDULE.search(text) or _POST.search(text))


# Verbs that introduce what the source says ("reports that B") ...
_SAYS = (
    r"states?|stated|stating|reports?|reported|reporting|indicates?|indicated"
    r"|indicating|says|said|saying|notes?|noted|noting|specifies|specified"
    r"|specifying|shows?|showed|showing"
)
# ... and verbs whose object is the claim ("identifies Pohl as the judge")
_NAMES = {
    "identifies": "identifies",
    "identified": "identifies",
    "identifying": "identifies",
    "describes": "describes",
    "described": "describes",
    "describing": "describes",
    "dates": "dates",
    "dated": "dates",
    "dating": "dates",
    "gives": "gives",
    "gave": "gives",
    "giving": "gives",
    "puts": "puts",
    "putting": "puts",
    "lists": "lists",
    "listed": "lists",
    "listing": "lists",
    "names": "names",
    "named": "names",
    "naming": "names",
    "cites": "cites",
    "cited": "cites",
    "citing": "cites",
}
_VERB = (
    rf"(?:\w+ly\s+)?(?:instead\s+)?(?P<verb>{_SAYS}|{'|'.join(_NAMES)})(?:\s+instead)?"
)
_DATE_WORDS = (
    r"(?:(?:from|on|dated|published)\s+)?(?:[A-Z][a-z]+\.?\s+(?:\d{1,2},\s+)?"
    r"\d{4}|\d{4}|[A-Z][a-z]+\.?\s+\d{1,2})"
)
_NEWER = r"(?:new|later|subsequent|newer|more recent|recent|latest)"
_SOURCE = r"(?:article|report|source|account|story)"
_SUBJECT = (
    rf"(?:the|this|a|an|another)?\s*(?:{_NEWER}\s+)?(?:(?P<d1>{_DATE_WORDS})\s+)?"
    rf"{_SOURCE}s?,?\s+(?:(?P<d2>{_DATE_WORDS}),?\s+)?{_VERB}\s+(?:that\s+)?"
)
# "..., while the profile states A, the article reports that B." The article's
# side follows its last "the article reports (that)"; searched from every
# start, since the first match would swallow the rest of the sentence
_TAIL = re.compile(
    rf"(?:[,;:]|\bwhile\b|\bwhereas\b|\bbut\b|\band\b)\s*{_SUBJECT}", re.I
)
# "Sources disagree on X, with the article stating that B, while the profile
# states A."
_WITH = re.compile(
    rf"\bwith\s+the\s+(?:{_NEWER}\s+)?{_SOURCE}\s+{_VERB}\s+(?:that\s+)?", re.I
)
# "B; the sources disagree on Y." The article's side comes first, often with
# the source as its subject ("The article identifies Pohl as the judge")
_LEAD = re.compile(
    r"^(?P<side>.+?)[;,]\s*(?:so\s+|these\s+|and\s+)?(?:the\s+)?sources?\s+"
    r"(?:disagree|differ|conflict)\w*\b.*$",
    re.I | re.S,
)
_LEAD_SUBJECT = re.compile(rf"^{_SUBJECT}", re.I)
# Where the profile's side starts after the article's ("B, while the profile
# states A")
_PROFILE_SIDE = re.compile(
    r",?\s+(?:while|whereas|but)\s+(?:the\s+|a\s+|an\s+)?(?:profile|earlier|previous"
    r"|prior|existing|other|older|another|one|some)\b.*$",
    re.I | re.S,
)
# The profile's side as a trailing clause without "while": "B, conflicting
# with the profile's count of 93", "B, contradicting the profile's claim that
# ...", "B, rather than the unspecified timing in the profile"
_PROFILE_TAIL = re.compile(
    r",?\s*(?:\b(?:thereby|thus|and|but)\s+)?"
    # "..., a figure that differs from the profile's count", "which is at odds"
    r"(?:\b(?:which|this|that|an?\s+(?:\w+\s+){1,2}(?:that|which))\s+)?"
    r"(?:\b(?:is|are|was|were|also|directly|therefore)\s+)*"
    r"\b(?:conflict\w*|contradict\w*|differ\w*|disagree\w*|inconsistent|at odds"
    r"|unlike|contrary to|in contrast (?:to|with)|rather than|instead of|whereas"
    # "..., which is consistent with the profile's X but conflicts with ..."
    r"|while|consistent with|in line with|match\w*|agree\w*|confirm\w*)\b"
    r"(?:(?!\b(?:whereas|while)\b)[^.;]){0,80}?"
    r"\bthe\s+(?:existing\s+|current\s+|earlier\s+)?profile\b.*$",
    re.I | re.S,
)
_OPENER = re.compile(r"^(?:however|but|yet|conversely|in contrast)[,\s]+", re.I)
# A side that still compares the two versions isn't the article's alone
_COMPARES = re.compile(
    r"\b(?:the profile|earlier report|previous report|prior report|rather than"
    r"|instead of|contradict\w*|disagree\w*|differs?|compared to)\b",
    re.I,
)


def _subject_date(m: "re.Match[str]") -> Optional[str]:
    """The date the sentence gives its source ("a May 21, 2013 report"), if it
    has a year and more than the year: the count is as of then, not as of the
    article."""
    for group in ("d1", "d2"):
        text = m.groupdict().get(group) or ""
        text = re.sub(r"^(?:from|on|dated|published)\s+", "", text, flags=re.I)
        if re.search(r"\b(?:19|20)\d\d\b", text) and not re.fullmatch(r"\d{4}", text):
            return text
    return None


_CURRENT = re.compile(
    rf"\b(?:the|this)\s+(?:{_NEWER}\s+)?(?:article|story)\b|\b{_NEWER}\b", re.I
)


def _current_clause(
    claim: str, clauses: List["re.Match[str]"]
) -> Optional["re.Match[str]"]:
    """The clause that gives the new article's side. With one source clause,
    that one; with several ("the article reports 59, while a March 25 report
    states 28"), the last one naming "the article" or a newer report, or
    None when none does, since picking by position could give the old value
    the new citation."""
    if len(clauses) == 1:
        return clauses[0]
    current = [c for c in clauses if _CURRENT.search(claim[c.start() : c.end()])]
    return current[-1] if current else None


def article_side(claim: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """What the article says, without the profile's side or the dispute
    wording: ("says", "the trial was pushed to Dec. 4", None) or
    ("identifies", "Pohl as the judge", None), by the verb that introduced
    it, with the date the sentence gives the source, if any. The verb is
    "notes" when the side may not be a sentence of its own (``_object_only()``).

    ``claim`` has no citations. None when no known wording fits or what is
    left still compares the two versions.
    """
    found = None
    subject = None  # the "the article reports (that)" in front of the side
    if m := _WITH.search(claim):
        # The side ends where another source's clause starts
        nxt = _TAIL.search(claim, m.end())
        found = (m.group("verb"), claim[m.end() : nxt.start() if nxt else None], None)
        subject = m
    elif tails := [m for i in range(len(claim)) if (m := _TAIL.match(claim, i))]:
        # One per "reports that", the longest, so "a March 4, 2014 report"
        # keeps its whole date
        clauses = [
            next(t for t in tails if t.end() == end)
            for end in sorted({t.end() for t in tails})
        ]
        m = _current_clause(claim, clauses)
        if m is None:
            return None
        # The side ends where another source's clause starts
        stop = min((c.start() for c in clauses if c.start() > m.end()), default=None)
        found = (m.group("verb"), claim[m.end() : stop], _subject_date(m))
        subject = m
    elif m := _LEAD.match(claim):
        side = m.group("side").strip()
        if s := _LEAD_SUBJECT.match(side):
            found = (s.group("verb"), side[s.end() :], _subject_date(s))
            subject = s
        else:
            found = ("says", side, None)
    elif _PROFILE_TAIL.search(claim):
        # "The article states B, conflicting with the profile's count of A"
        side = _OPENER.sub("", _PROFILE_TAIL.sub("", claim).strip())
        if s := _LEAD_SUBJECT.match(side):
            found = (s.group("verb"), side[s.end() :], _subject_date(s))
            subject = s
        else:
            found = ("says", side, None)
    if not found:
        return None
    verb, side = found[0].lower(), _PROFILE_SIDE.sub("", found[1]).strip()
    side = _PROFILE_TAIL.sub("", side).strip()
    side = side.rstrip(".;, ")
    if subject and _took_that(subject):
        # "reports that A and that B": the second "that" has lost its verb
        side = re.sub(r"\b(and|but)\s+that\b", r"\1", side)
    if len(side.split()) < 4 or _COMPARES.search(side):
        return None
    if verb in _NAMES:
        return _NAMES[verb], side, found[2]
    return (
        ("notes" if subject and _object_only(subject, side) else "says"),
        side,
        found[2],
    )


# A word that makes the side a sentence of its own, outside a "who ..." clause
_FINITE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|would|can|could|did|does|must|may"
    r"|might|should)\b",
    re.I,
)
_RELATIVE = re.compile(r"\b(?:who|whom|whose|which|that)\b.*$", re.I | re.S)


def _took_that(subject: "re.Match[str]") -> bool:
    return subject.group(0).rstrip().lower().endswith("that")


def _object_only(subject: "re.Match[str]", side: str) -> bool:
    """The side may be only what the verb reports ("The article reports 41
    captives at the camp"), not a sentence: no "that" before it and no
    "is/was/has/will" in it. Such a side keeps a source in front ("A report
    notes 41 captives at the camp"), which reads right either way."""
    if _took_that(subject):
        return False
    return not _FINITE.search(_RELATIVE.sub("", side))


def to_date(value: Any) -> Optional[dt.date]:
    """A published date as stored in Parquet (Unix seconds, ISO text, a date)."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, (int, float)):
        # Milliseconds if it's too big to be seconds
        seconds = value / 1000 if value > 1e11 else value
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).date()
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def cited_ids(sentence: str) -> set:
    return {i.strip() for g in _CITED.findall(sentence) for i in g.split(",")}


# Full month names and their short forms ("Mar.", "Sept"), any case
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b\.?"
)
# The side already says when: it starts with a date ("On May 16, 2013, ...",
# "In February 2014, ..."), or gives one as when the value held ("... as of
# July 17, 2013", "... in February 2014"). A date it is about ("postponed
# until February 2015", "scheduled for December 4, 2014", "to begin in
# December 2014") doesn't count
_STARTS_DATED = re.compile(
    rf"^(?:(?:on|in|by|as of|since|during|prior to|before|after)\s+(?:early |late "
    rf"|mid-)?(?:the\s+\w+\s+\d{{4}}\s+\w+|{_MONTH}\s+(?:\d{{1,2}},?\s+)?\d{{4}})"
    rf"|{_MONTH}\s+\d{{1,2}},\s+\d{{4}}"
    rf"|(?:in|by|during|since)\s+(?:early |late |mid-)?(?:19|20)\d\d\b)",
    re.I,
)
_AS_OF = re.compile(r"\bas of\b", re.I)
_ENDS_DATED = re.compile(
    rf"\b(?:on|in|by|during)\s+(?:early |late |mid-)?{_MONTH}\s+"
    rf"(?:\d{{1,2}},?\s+)?\d{{4}}\s*$",
    re.I,
)

_TARGET = re.compile(
    rf"\b(?:for|to|until|till|begin|start|open|resume|before|after)\s+"
    rf"(?:at least\s+|no earlier than\s+)?(?:on\s+|in\s+)?(?:early |late |mid-)?"
    rf"{_MONTH}\s+(?:\d{{1,2}},?\s+)?\d{{4}}\s*$",
    re.I,
)


def _says_when(side: str) -> bool:
    """The side already says when its value held."""
    if _STARTS_DATED.match(side) or _AS_OF.search(side):
        return True
    return bool(_ENDS_DATED.search(side)) and not _TARGET.search(side)


def as_update(
    verb: str,
    side: str,
    dated: Optional[str],
    article_date: dt.date,
    citations: str,
) -> str:
    """The article's side as a dated timeline entry, with its citations:
    "As of April 2013, 59 captives were hunger strikers.^[id]", or "A June
    2018 report identifies Pohl as the judge.^[id]". Dated by the article's
    month unless the sentence dated the source itself (``dated``)."""
    month = dated or f"{article_date:%B %Y}"
    if verb != "says":
        # "A report notes ... in October 2017", not "An October 2017 report"
        month = "" if month.lower() in side.lower() else f"{month} "
        article = "An" if re.match(r"[AEIOU]|8|1[18]\b", month) else "A"
        text = f"{article} {month}report {verb} {side}"
    elif _says_when(side) or month.lower() in side.lower():
        text = side[0].upper() + side[1:]
    else:
        first, _, rest = side.partition(" ")
        # "the trial was ..." stays lower case; a name keeps its capital
        text = f"As of {month}, {first} {rest}"
    unique = "".join(dict.fromkeys(_CITATION.findall(citations)))
    return f"{text.rstrip('.')}.{unique}"


_DAY = re.compile(rf"\b({_MONTH})\s+(\d{{1,2}})\b(?:,?\s+((?:19|20)\d\d)\b)?", re.I)


def _days(text: str) -> set:
    """Days the text names ("March 25", "Mar. 25, 2013"), as ("mar", "25",
    "2013"), the year None when not given."""
    return {
        (m.group(1).lower()[:3], m.group(2), m.group(3)) for m in _DAY.finditer(text)
    }


def _same_day(a: tuple, b: tuple) -> bool:
    # Sept. 2, 2014 and Sept. 2, 2015 are a year apart; a day without a year
    # could be either
    return a[:2] == b[:2] and (a[2] is None or b[2] is None or a[2] == b[2])


def same_observation_day(side: str, dated: Optional[str], olds: Iterable[str]) -> bool:
    """The article's value and an old sentence are for the same named day:
    28 hunger strikers on March 25 in one, 30 on March 25 in the other is a
    real dispute, whenever each article was published."""
    mine = _days(side) | _days(dated or "")
    theirs = {d for o in olds for d in _days(_CITATION.sub("", o))}
    return any(_same_day(a, b) for a in mine for b in theirs)


def own_citations(sentence: str, olds: Iterable[str]) -> str:
    """The sentence's citations without those of the sentences it disputes,
    which back the profile's side, not the article's (all of them if that
    would leave none)."""
    theirs = {i for o in olds for i in cited_ids(o)}
    mine = [i for i in dict.fromkeys(_ids_in_order(sentence)) if i not in theirs]
    if not mine:
        return "".join(_CITATION.findall(sentence))
    return "".join(f"^[{i}]" for i in mine)


def _ids_in_order(sentence: str) -> List[str]:
    return [i.strip() for g in _CITED.findall(sentence) for i in g.split(",")]


def dated_apart(
    article_date: Optional[dt.date],
    old_sentences: Iterable[str],
    dates: Dict[str, dt.date],
) -> bool:
    """The old sentences cite at least one dated source, and none from the
    article's own day.

    Usually the article is newer. When it is older (a run that doesn't go in
    date order: the pilot added a June 2018 article naming Pohl as judge to a
    profile already saying Parrella took over in September), its value is an
    earlier state, so it is a timeline entry too, dated by the article.
    """
    if article_date is None:
        return False
    cited = [dates[i] for s in old_sentences for i in cited_ids(s) if i in dates]
    return bool(cited) and article_date not in cited


def article_dates(
    entity: Dict[str, Any], article_id: str, published: Any
) -> Dict[str, Any]:
    """Published date by article id: the entity's linked articles and the new
    one (``merge_entities()`` passes this to ``update_profile()``)."""
    linked = entity.get("articles") or []
    dates = {a.get("article_id"): a.get("article_published_date") for a in linked}
    return {**dates, article_id: published}
