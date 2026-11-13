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
  "expected", "allegedly") as that sentence, and that sentence doesn't
  frame it as someone's claim or denial ("his lawyers denied that ...").
  Dropped.
- **Invented dispute**: it says sources or reports disagree, but no existing
  sentence that shares a topic word gives a different year, or a different
  month or count for the same year (``_disputes_value()``). Dropped.
- **Later update**: it says sources disagree, but about something that
  changes over time (a hunger-strike count, a trial date, who is judge), and
  the sentence it disputes cites only articles from other days. Rewritten as
  the article's own value, dated ("As of April 2013, 59 captives were hunger
  strikers."), next to the old sentence (``later_update()``,
  ``later_updates.py``). Needs the articles' published dates.

- **Note about the profile**: it compares the article with "the profile"
  ("..., conflicting with the profile's count of 93"). Never kept as written:
  handled like a "sources disagree" sentence, see ``profile_notes.py``.

``replace`` edits are left alone: they rewrite one sentence and keep its
citations. New sentences are checked against the profile as the
replacements leave it. An edit whose sentences are all dropped is removed.
"""

import datetime as dt
import re
from typing import Any, Dict, List, Optional, Tuple

from src.engine.date_disputes import _PUBLISHED
from src.engine.dispute_check import Ask, check_disputes, replacement
from src.engine.fact_list import _MONTH, _stems, _values, adds_values
from src.engine.later_updates import (
    article_side,
    as_update,
    dated_apart,
    is_time_varying,
    own_citations,
    same_observation_day,
    to_date,
)
from src.engine.profile_edits import ProfileEdit, apply_edits
from src.engine.profile_notes import (
    is_empty_note,
    is_profile_note,
    plain_side,
    reworded,
    says_disagree,
)
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
# Anything up to the sentence's end, letting through an abbreviation's period
# ("dates the conviction to Oct. 31, 2017, instead")
_NOT_END = r"(?:[^.;]|\.(?=\S|\s+\d))"
_SRC = r"(?:sources?|reports?|accounts?|articles?|records?|profile)"
_NARRATOR = re.compile(
    rf"\b{_SRC}\s+(?:disagree|differ|conflict|vary|contradict)\w*"
    r"|\bconflicting (?:accounts|reports|dates|figures|sources)"
    r"|\bdiscrepanc\w*"
    rf"|\b(?:differs?|disagrees?|contradicts?|contradicting|inconsistent|conflicts?)"
    rf"\s+(?:with|from)?\s*(?:the|an?|this)?\s*(?:earlier|previous|prior|other|"
    rf"another|existing|new|later|newer|older|original|reported)\s+{_SRC}"
    # "these sources give different timelines", "this differs from the
    # reported May transfer date": wordings the 7 Nov Luna run used
    rf"|\b{_SRC}\s+(?:\w+\s+)?(?:give|gives|offer|offers)\s+(?:a\s+)?different"
    r"|\b(?:this|which)\s+(?:differs?|conflicts?|contradicts?)\b"
    # "The October 2010 article dates the attack to July 2001 instead."
    # Only with the article as subject: "A Senate report recommended
    # Colorado instead" is news
    r"|^\s*(?:the|an?|this)\s+(?:[\w.,]+\s+){0,5}?(?:article|source|account)s?\b"
    rf"{_NOT_END}*\binstead\b",
    re.I,
)
# Softer comparison wordings with no "disagree": "The article reports that the
# attacks killed 2,974 people, rather than 2,976." and the prompt's own example
# "A later report gives the year as 2003 instead." On the 7 Nov replay these
# skipped every guard and the dispute check. They can also be news ("the
# article says he chose prison rather than exile"), so they are never dropped
# as invented disputes; they only go to the dispute check, with the sentences
# ``_disputed_olds()`` finds (the anchor's, else ones giving another value)
_SOFT_DISPUTE = re.compile(
    r"^\s*(?:the|an?|this)\s+(?:[\w.,]+\s+){0,5}?(?:article|source|account)s?\b"
    rf"{_NOT_END}*\brather than\s+(?:[\w.]+\s+){{0,2}}?(?:\d|(?:Jan|Feb|Mar|Apr|May|Jun|Jul"
    r"|Aug|Sep|Oct|Nov|Dec)[a-z]*\b)"
    r"|\b(?:later|earlier|newer|older|another|other|second|new|separate)\s+"
    r"(?:report|article|account|source)s?\b"
    rf"{_NOT_END}*\b(?:instead|rather than)\b",
    re.I,
)
_NUMBERS = re.compile(r"\b\d[\d,]*\d\b|\b\d\b")
# Capitalised words after the first: names. "consulted Judge Spath" isn't a
# repeat of "consulted Judge Pohl", and the stems drop short names
_NAME = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")


def _claim(sentence: str) -> str:
    return _CITATION.sub("", sentence).strip()


# Words that carry no fact, and the wrapping models put round a restated one
# ("The article states that ...")
_FILLER = set(
    "the a an of in on at to for by with as is was were be been being "
    "has had have that this these those from its it his her their they he she "
    "which who whom also there "
    "article articles report reports reported states stated notes noted "
    "according describes described says said new profile".split()
)


def _stem(word: str) -> str:
    """The word without an ending: "detainees"/"detained", "resume"/"resumed"
    match, "convicted"/"convinced" don't (a 5-letter cut made both "convi")."""
    for end in ("ing", "ed", "es", "s"):
        if len(word) > len(end) + 2 and word.endswith(end):
            word = word[: -len(end)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def _words(text: str) -> set:
    """Every word of the sentence but filler, without its ending (``_stem()``). Short words count: "won" and "lost", "Pohl" and
    "Spath" are what separate two otherwise identical sentences."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if w not in _FILLER}


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
    framing = _framing(_claim(sentence))
    for other in known:
        if (
            words <= _words(_claim(other))
            and _negations(other) == negations
            and _hedges(_claim(other)) == hedges
            and _framing(_claim(other)) <= framing
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
    return [_stem(w) for w in words if w not in _FILLER]


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
# a repeat of "the hearing resumed in 2014" (the word check makes "resume"
# and "resumed" one word), it's the news that it happened
_HEDGE = re.compile(
    r"\b(could|may|might|would|will|shall|should|can|cannot|expected|expects?"
    r"|plans?|planned|intends?|intended|likely|unlikely|possibly|probably"
    r"|allegedly|reportedly|proposed|scheduled|set to|due to|hopes?|hoped)\b",
    re.I,
)


def _hedges(text: str) -> List[str]:
    return sorted(m.lower() for m in _HEDGE.findall(text))


# Words that put an old sentence's claim in someone's mouth or deny it: "Nashiri
# confessed" isn't a repeat of "his lawyers denied that Nashiri confessed",
# though every word of it is there. A new sentence that drops one of these
# from the old one is kept
_FRAMING = re.compile(
    r"\b(den(?:y|ies|ied|ial)|disput\w*|reject\w*|refut\w*|contest\w*|false\w*"
    r"|claim\w*|alleg\w*|accus\w*|insist\w*|argu\w*|assert\w*|suggest\w*"
    r"|rumou?r\w*|whether|if)\b",
    re.I,
)


def _framing(text: str) -> set:
    return {_stem(m.lower()) for m in _FRAMING.findall(text)}


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


def _compares(claim: str) -> bool:
    """The sentence sets two sources against each other in the profile's own
    voice: "sources disagree", or a note about what "the profile" says."""
    return bool(_NARRATOR.search(claim)) or is_profile_note(claim)


def disputed_sentences(
    sentence: str, known: List[str], soft: bool = False
) -> List[str]:
    """The known sentences a "sources disagree" sentence could be disputing:
    those giving a different year (or month or count for the same year), or
    another name, for something it mentions. Empty for any other sentence;
    with ``soft``, a ``_SOFT_DISPUTE`` wording ("rather than 2,976") counts too."""
    claim = _claim(sentence)
    if not (_compares(claim) or (soft and _SOFT_DISPUTE.search(claim))):
        return []
    # The dispute wording itself says nothing about what is disputed, and a
    # source's own date ("the August 14, 2009 article describes Khadr as 22")
    # is when it was written: on dev, Luna's two kept false disputes differed
    # from the profile only in that year
    # A note about the profile quotes the profile's value too ("... 104
    # captives, conflicting with the profile's count of 93"), so an old
    # sentence may also differ from the article's half alone
    side = article_side(claim) if is_profile_note(claim) else None
    topic = _SOURCES.sub(" ", _DISPUTE.sub(" ", _PUBLISHED.sub(" ", claim)))
    # "... 2,974 people, rather than 2,976": the second value is the old one
    rather = re.search(r"\brather than\b(.*)$", topic, flags=re.I)
    topic = topic[: rather.start()] if rather else topic
    topics = [topic] + ([_PUBLISHED.sub(" ", side[1])] if side else [])
    found = [
        o
        for o in known
        if any(_disputes_value(t, _PUBLISHED.sub(" ", _claim(o))) for t in topics)
    ]
    if found or not rather:
        return found
    # The value topic words can't link ("death toll" against "killed"): the
    # old sentence holding the value the claim names after "rather than"
    named = set(_NUMBERS.findall(rather.group(1))) - set(_NUMBERS.findall(topic))
    return [o for o in known if named & set(_NUMBERS.findall(_claim(o)))]


def is_invented_dispute(sentence: str, known: List[str]) -> bool:
    """Says the sources disagree, but no known sentence gives a different year
    (or a different month or count for the same year) for something it mentions."""
    return bool(_NARRATOR.search(_claim(sentence))) and not disputed_sentences(
        sentence, known
    )


def later_update(
    sentence: str,
    profile: List[str],
    article_date: Optional[dt.date],
    dates: Dict[str, dt.date],
    anchor: str = "",
) -> Optional[str]:
    """A "sources disagree" sentence rewritten as the article's own value, when
    it is about something that changes over time and the article is from
    another date than the sentences it disputes (``later_updates.py``);
    otherwise None.

    ``profile`` is the profile's own sentences, not ones this update adds
    (those cite the article, so they share its date). The disputed sentences
    are the profile sentences giving another value (``disputed_sentences()``),
    narrowed to the edit's anchor when the anchor is one of them (the prompt
    asks for the dispute right after it).
    """
    claim = _claim(sentence)
    if not _compares(claim):
        return None
    if article_date is None or not is_time_varying(claim):
        return None
    candidates = disputed_sentences(sentence, profile)
    # The anchor only says where the edit goes; it is the disputed sentence
    # only when it is one of the sentences giving another value
    olds = [anchor] if anchor in candidates else candidates
    if not olds or not dated_apart(article_date, olds, dates):
        return None
    found = article_side(claim)
    if found is None:
        return None
    verb, side, dated = found
    if same_observation_day(side, dated, olds):
        return None
    return as_update(verb, side, dated, article_date, own_citations(sentence, olds))


def paragraph_citations(text: str) -> Dict[str, str]:
    """Each uncited sentence with the citations of the next cited sentence in
    its paragraph appended: profiles cite once per paragraph, so the dispute
    check can see when "... killed 2,976 people." was reported."""
    out: Dict[str, str] = {}
    for paragraph in re.split(r"\n\s*\n", text):
        pending: List[str] = []
        for sentence in _sentences(paragraph):
            if _CITATION.search(sentence):
                cites = "".join(_CITATION.findall(sentence))
                out.update({p: f"{p}{cites}" for p in pending})
                pending = []
            elif not sentence.startswith("#"):
                pending.append(sentence)
    return out


def _disputed_olds(sentence: str, profile: List[str], anchor: str) -> List[str]:
    """The profile sentences a kept "sources disagree" sentence disputes.

    The prompt puts the dispute right after the sentence it disputes, but the
    model often anchors on a whole paragraph, and the disputed value is
    usually not in its last sentence ("... killed 2,976 people. It gives no
    further details."). So: the anchor's sentences giving another value; else
    all of them (at most three); else the profile's sentences giving another
    value, at most two."""
    anchors = [s for s in profile if s and s in anchor]
    valued = disputed_sentences(sentence, anchors, soft=True)
    if valued:
        return valued
    return anchors[:3] or disputed_sentences(sentence, profile, soft=True)[:2]


def _anchor_differs(sentence: str, profile: List[str], anchor: str) -> bool:
    """A sentence of the edit's anchor gives another count or year than the
    dispute does. Topic words can miss the link ("a later article gives the
    death toll as 2,974 instead" against "... attacks, which killed 2,976
    people"), so such a dispute goes to the model check instead of being
    dropped as invented."""
    topic = _PUBLISHED.sub(" ", _claim(sentence))
    years, _, counts = _values(topic)
    for old in (s for s in profile if s and s in anchor):
        o_years, _, o_counts = _values(_PUBLISHED.sub(" ", _claim(old)))
        if (counts - o_counts and o_counts) or (years - o_years and o_years):
            return True
    return False


def _to_check(sentence: str, profile: List[str], anchor: str) -> bool:
    """Whether a kept sentence goes to the dispute check: it says sources
    disagree, or it compares them more softly ("rather than 2,976", "a later
    report ... instead") and there is an old sentence to set it against."""
    claim = _claim(sentence)
    if _compares(claim):
        return True
    if not _SOFT_DISPUTE.search(claim):
        return False
    return bool(_disputed_olds(sentence, profile, anchor))


def guard_edits(
    text: str,
    edits: List[ProfileEdit],
    article_date: Any = None,
    source_dates: Optional[Dict[str, Any]] = None,
    ask: Optional[Ask] = None,
) -> Tuple[List[ProfileEdit], Dict[str, Any]]:
    """The edits with repeated and invented-dispute sentences taken out, and
    "sources disagree" sentences that are later updates rewritten as one.

    ``article_date`` is the new article's published date and
    ``source_dates`` the published date of each article the profile cites
    (by id); without them no sentence is read as a later update. With
    ``ask`` (a model call, ``dispute_check.py``), every "sources disagree"
    sentence still left is checked by the model and kept, rewritten or
    dropped by its verdict.
    """
    when = to_date(article_date)
    dates = {k: d for k, v in (source_dates or {}).items() if (d := to_date(v))}
    # Checked against the profile as the replacements leave it: when an edit
    # replaces "captured in 2002 and charged in 2008" with the capture alone
    # and adds the charge separately, the added charge isn't a repeat. The
    # replacements are applied the way ``apply_edits()`` will apply them, so
    # one whose anchor isn't found changes nothing here either
    replaced, _ = apply_edits(text, [e for e in edits if e.op == "replace"])
    known = [s for s in _sentences(replaced) if not s.startswith("#")]
    profile = list(known)
    report: Dict[str, Any] = {"repeats": [], "disputes": [], "later_updates": []}
    report["profile_notes"] = []
    cited = paragraph_citations(replaced) if ask else {}
    # Per edit: (span, new text) swaps and how many facts stay
    swaps: List[List[Tuple[Tuple[int, int], str]]] = []
    facts_left: List[int] = []
    pending: List[Dict[str, Any]] = []  # disputes for the model check
    # Sentences this call rewrote: the same one again in the reply is a repeat
    rewritten: List[str] = []
    for n, edit in enumerate(edits):
        swaps.append([])
        facts_left.append(0)
        if edit.op != "add":
            continue
        cursor = 0
        for sentence in _sentences(edit.text):
            # Each sentence's own place in the text, so a dropped sentence
            # that also ends an earlier kept one is cut from the right spot
            start = edit.text.find(sentence, cursor)
            span = (start, start + len(sentence)) if start >= 0 else None
            cursor = span[1] if span else cursor
            if sentence.startswith("#"):
                continue
            new = ""
            claim = _claim(sentence)
            note = is_profile_note(claim)
            if is_repeat(sentence, known) or is_repeat(sentence, rewritten):
                report["repeats"].append(sentence[:200])
            elif note and is_empty_note(claim):
                report["profile_notes"].append([sentence[:200], "empty", ""])
            elif is_invented_dispute(sentence, known) and not (
                ask and _anchor_differs(sentence, profile, edit.anchor)
            ):
                report["disputes"].append(sentence[:200])
            elif new := later_update(sentence, profile, when, dates, edit.anchor) or "":
                rewritten.append(sentence)
                if _has(known, new):  # two notes with one later state
                    new = ""
                report["later_updates"].append([sentence[:200], new[:200]])
                if new:
                    facts_left[n] += 1
                    known.append(new)
            elif note and not (ask and span):
                rewritten.append(sentence)
                how, new = _unjudged_note(sentence, when, known)
                report["profile_notes"].append([sentence[:200], how, new[:200]])
                if new:
                    facts_left[n] += 1
                    known.append(new)
            else:
                facts_left[n] += 1
                known.append(sentence)  # a later edit can't add it again
                if ask and span and _to_check(sentence, profile, edit.anchor):
                    olds = [
                        cited.get(o, o)
                        for o in _disputed_olds(sentence, profile, edit.anchor)
                    ]
                    pending.append(
                        {"edit": n, "span": span, "sentence": sentence, "olds": olds}
                    )
                continue
            if span:
                swaps[n].append((span, new))
    if ask and pending:
        _apply_dispute_check(
            pending, ask, when, dates, known, swaps, facts_left, report
        )
    kept: List[ProfileEdit] = []
    for n, edit in enumerate(edits):
        if edit.op != "add" or not swaps[n]:
            kept.append(edit)
            continue
        new_text = edit.text
        for (start, end), new in sorted(swaps[n], reverse=True):
            new_text = new_text[:start] + new + new_text[end:]
        if facts_left[n]:
            tidy = re.sub(r"[ \t]{2,}", " ", new_text).strip()
            kept.append(edit.model_copy(update={"text": tidy}))
    return kept, report


def _apply_dispute_check(
    pending: List[Dict[str, Any]],
    ask: Ask,
    when: Optional[dt.date],
    dates: Dict[str, dt.date],
    known: List[str],
    swaps: List[List[Tuple[Tuple[int, int], str]]],
    facts_left: List[int],
    report: Dict[str, Any],
) -> None:
    """Keep, rewrite or drop each pending dispute by the model's verdict."""
    verdicts = check_disputes(pending, ask, when, dates)
    report["dispute_check"] = []
    for i, item in enumerate(pending):
        verdict = verdicts.get(i)
        if verdict is None or verdict.verdict == "conflict":
            how = verdict.verdict if verdict else "unchecked"
            report["dispute_check"].append([item["sentence"][:200], how])
            if is_profile_note(_claim(item["sentence"])):
                _keep_note(item, how, when, known, swaps, facts_left, report)
            continue
        sentence = item["sentence"]
        known.remove(sentence)
        cites = own_citations(sentence, item["olds"])
        new = replacement(verdict, sentence, cites)
        if new and _has(known, new):
            new = None
        report["dispute_check"].append(
            [sentence[:200], verdict.verdict, (new or "")[:200]]
        )
        if new:
            known.append(new)
        else:
            facts_left[item["edit"]] -= 1
        swaps[item["edit"]].append((item["span"], new or ""))


def _keep_note(
    item: Dict[str, Any],
    how: str,
    when: Optional[dt.date],
    known: List[str],
    swaps: List[List[Tuple[Tuple[int, int], str]]],
    facts_left: List[int],
    report: Dict[str, Any],
) -> None:
    """A note about the profile that stays: a real conflict with "the
    profile" reworded as "another report"; one the model never judged (the
    call failed) as the article's half alone, else reworded too."""
    sentence = item["sentence"]
    known.remove(sentence)
    if how != "conflict":
        how, new = _unjudged_note(sentence, when, known)
    elif _has(known, new := reworded(sentence)):
        how, new = "repeat", ""  # the same conflict twice in one reply
    if new:
        known.append(new)
    else:
        facts_left[item["edit"]] -= 1
    report["profile_notes"].append([sentence[:200], how, new[:200]])
    swaps[item["edit"]].append((item["span"], new))


def _has(known: List[str], new: str) -> bool:
    """A rewritten sentence is already there: a repeat, or the same words (a
    short one, "There are 104 captives.", is never a repeat by
    ``is_repeat()``'s rules)."""
    same = " ".join(_claim(new).lower().split())
    return is_repeat(new, known) or any(
        " ".join(_claim(k).lower().split()) == same for k in known
    )


def _unjudged_note(
    sentence: str, when: Optional[dt.date], known: List[str]
) -> Tuple[str, str]:
    """(how, new text) for a note about the profile no model has judged.

    One that says outright that sources disagree stays a dispute, with "the
    profile" reworded as "another report". Any other becomes the article's
    half alone ("" when the profile already has it); with no half that
    stands alone it is reworded too."""
    plain = None if says_disagree(_claim(sentence)) else plain_side(sentence, when)
    if plain is None:
        new = reworded(sentence)
        return ("repeat", "") if _has(known, new) else ("reworded", new)
    return ("repeat", "") if _has(known, plain) else ("plain", plain)
