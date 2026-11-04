"""Catch an article that gives a past event a different date than the profile.

A step of the fact-list updater (``fact_list.py``), run after the claims are
compared and before new facts are written.

Why a separate step: the extract step lists what the article says *about the
entity*, so dates in background sentences never reach the compare step.
Scenario: the profile says "Spath was assigned to the case against Nashiri,
accused of the October 2000 USS Cole bombing"; an article about a hearing says
in passing "17 sailors died in the bombing of the USS Cole on Oct. 12, 1999".
Luna and Qwen both listed only the hearing's news, so the 1999 never got
compared and the profile kept one side. Loosening the compare step to catch it
brought back invented "sources disagree" sentences, so this step works the
other way round:

1. **Pair** (code): every article sentence with a year, beside the cited
   profile sentences that give a *different* year (or the same year with a
   different month) and share at least ``DATE_PAIR_OVERLAP`` of their words.
   Most articles give no pair at all; the rest give a handful
   (``MAX_DATE_PAIRS`` at most), so the model never sees the whole profile.
2. **Judge** (model, one call): for each pair, is it the same completed event
   given two dates (``conflict``), two different events (``different``), or a
   plan, schedule or count that changed over time (``later``)?
3. **Write** (code): for a ``conflict``, the article's version goes right
   after the old sentence, cited to the article, so both dates stay with their
   own sources. The sentence must state the article's differing year or
   month, or code uses the article's own sentence instead.
"""

import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from src.evals.profile_cases import _sentences

DATE_PAIR_OVERLAP = 0.25  # share of the shorter sentence's words both must use
MIN_DATE_SHARED = 2  # and at least this many words
MAX_DATE_PAIRS = 8  # pairs shown to the model per article
# Rows one article sentence may be paired with: a profile often states one
# event in two sections, and the closest by words isn't always the one dated
# ("the Pentagon had been preparing the case for years" beat "withdrew the
# 2008 war-crimes case" for "the Pentagon charged Mohammed in 2009")
PAIRS_PER_SENTENCE = 2
DISPUTE_PREFIX = "Another report differs: "
DATE_CHECK_ENV = "HINBOX_FACT_DATE_CHECK"

_CITATION = re.compile(r"\^\[([^\]]+)\]")
_YEAR = re.compile(r"\b(?:1[89]|20)\d\d\b")
_MONTH = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|March|April|May|June|July|Aug(?:ust)?"
    r"|Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b\.?"
)
_DATE = rf"(?:{_MONTH.pattern}\s+(?:\d{{1,2}},?\s+)?)?(?:1[89]|20)\d\d"
# "The May 1, 2010 article says...", "an article published April 28, 2012":
# when a source was written, not when its event happened
_PUBLISHED = re.compile(
    rf"\b(?:the|an?|this|that)\s+{_DATE}\s+(?:article|report|story)"
    rf"|\b(?:article|report|story),?\s+(?:was\s+)?(?:published|dated)\s+"
    rf"(?:on\s+)?{_DATE}",
    re.IGNORECASE,
)
# Words of 4+ letters that say nothing about which event a sentence is on
_STOP = set(
    "that with from were have said they this been when into after their there "
    "would which about while where what also than then them more most other "
    "some such only over under before since during until again could should "
    "being these those does just made like year years month later earlier "
    "says told report article reported according".split()
)


class DateVerdict(BaseModel):
    pair: int = Field(description="The pair's number")
    verdict: Literal["conflict", "different", "later"]
    sentence: str = Field(
        default="",
        description="For conflict: the article's version as one sentence, no citation",
    )


class DateVerdicts(BaseModel):
    verdicts: List[DateVerdict] = Field(default_factory=list)

    @field_validator("verdicts", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        from src.engine.profile_edits import _parse_list_string

        if isinstance(value, str):  # Bolt sometimes sends a list as a string
            parsed = _parse_list_string(value)
            return value if parsed is None else parsed
        return value


def date_check_enabled(flag: Optional[bool] = None) -> bool:
    """On unless turned off by argument or ``HINBOX_FACT_DATE_CHECK=off``."""
    if flag is not None:
        return flag
    return os.getenv(DATE_CHECK_ENV, "on").strip().lower() not in {"off", "0", "no"}


def _words(text: str) -> Set[str]:
    text = _CITATION.sub("", text).lower()
    return {w[:5] for w in re.findall(r"[a-z]{4,}", text) if w not in _STOP}


def _dates(text: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Years in the text, and for each year the months written just before it.
    Publication dates ("the May 1, 2010 article") are left out."""
    text = _PUBLISHED.sub("the article", _CITATION.sub("", text))
    years = set(_YEAR.findall(text))
    months: Dict[str, Set[str]] = {}
    for year in years:
        for m in re.finditer(rf"{_MONTH.pattern}\s+(?:\d{{1,2}},?\s+)?{year}\b", text):
            months.setdefault(year, set()).add(m.group(1).lower()[:3])
    return years, months


def dates_differ(article_sentence: str, profile_sentence: str) -> bool:
    """Each sentence has a year the other lacks ("2005" against "2006"), or both
    give the same year with different months ("October 2002" / "November 2002").
    Extra dates on one side only ("in 2006, and again in 2008") don't count."""
    (a_years, a_months), (p_years, p_months) = (
        _dates(article_sentence),
        _dates(profile_sentence),
    )
    if a_years - p_years and p_years - a_years:
        return True
    for year in a_years & p_years:
        mine, theirs = a_months.get(year, set()), p_months.get(year, set())
        if mine and theirs and not mine & theirs:
            return True
    return False


def overlap(a: str, b: str) -> Tuple[int, float]:
    """(shared words, their share of the shorter sentence's words)."""
    mine, theirs = _words(a), _words(b)
    shared = len(mine & theirs)
    return shared, shared / (min(len(mine), len(theirs)) or 1)


def date_pairs(
    rows: List[Dict[str, Any]], article_text: str, skip: Set[str]
) -> List[Dict[str, Any]]:
    """Article sentences beside the profile rows they may give another date for.

    Best overlap first; each row is used once and each article sentence at
    most ``PAIRS_PER_SENTENCE`` times, so one long background paragraph
    can't fill every slot.
    """
    sentences = _sentences(article_text)
    found = []
    for i, sentence in enumerate(sentences):
        if not _YEAR.search(sentence):
            continue
        for row in rows:
            if row["id"] in skip:
                continue
            if not dates_differ(sentence, row["claim"]):
                continue
            shared, share = overlap(sentence, row["claim"])
            if shared >= MIN_DATE_SHARED and share >= DATE_PAIR_OVERLAP:
                context = sentences[i - 1] if i else ""
                found.append((share, shared, i, row, sentence, context))
    found.sort(key=lambda f: (-f[0], -f[1], f[2]))
    pairs: List[Dict[str, Any]] = []
    used_rows: Set[str] = set()
    per_sentence: Dict[int, int] = {}
    for share, _, i, row, sentence, context in found:
        if row["id"] in used_rows or per_sentence.get(i, 0) >= PAIRS_PER_SENTENCE:
            continue
        used_rows.add(row["id"])
        per_sentence[i] = per_sentence.get(i, 0) + 1
        pairs.append(
            {"row": row, "sentence": sentence, "context": context, "overlap": share}
        )
        if len(pairs) == MAX_DATE_PAIRS:
            break
    return pairs


def pairs_prompt(pairs: List[Dict[str, Any]]) -> str:
    blocks = []
    for n, pair in enumerate(pairs, 1):
        context = f"(Before it: {pair['context']})\n" if pair["context"] else ""
        blocks.append(
            f"Pair {n}\nKnown fact: {pair['row']['claim']}\n"
            f"{context}Article sentence: {pair['sentence']}"
        )
    return "\n\n".join(blocks)


def dispute_text(verdict_sentence: str, pair: Dict[str, Any], article_id: str) -> str:
    """The sentence code writes after the old one, cited to the article.

    The model's sentence is used only when it states the article's differing
    date and no year the article sentence doesn't give; otherwise (empty, it
    copied the profile's date, or it made up a third one) the article's own
    sentence is written.
    """
    body = _CITATION.sub("", verdict_sentence).strip()
    article_years = _dates(pair["sentence"])[0]
    if (
        not body
        or not dates_differ(body, pair["row"]["claim"])
        or not _dates(body)[0] <= article_years
    ):
        body = _CITATION.sub("", pair["sentence"]).strip()
    body = body.removeprefix(DISPUTE_PREFIX.strip()).strip()
    return f"{DISPUTE_PREFIX}{body}^[{article_id}]"


def find_date_disputes(
    ask: Callable[..., Any],
    rows: List[Dict[str, Any]],
    article_text: str,
    article_id: str,
    skip: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Steps 1-3. Returns the disputes to write and a report of what happened.

    Each dispute is ``{"row", "text", "quote"}``: the old row, the sentence to
    add after it, and the article sentence it rests on.
    """
    # Old dispute sentences and rows already disputed are left alone, so the
    # next article repeating the other date doesn't add a second "differs"
    skip = skip | {r["disputes"] for r in rows if r.get("disputes")}
    candidates = [
        r
        for r in rows
        if not r.get("disputes") and not r["text"].startswith(DISPUTE_PREFIX)
    ]
    pairs = date_pairs(candidates, article_text, skip)
    report: Dict[str, Any] = {"date_pairs": len(pairs), "date_disputes": 0}
    if not pairs:
        return [], report
    answer = ask("date_check", pairs_prompt(pairs), DateVerdicts)
    verdicts = {v.pair: v for v in answer.verdicts}
    disputes = []
    for n, pair in enumerate(pairs, 1):
        verdict = verdicts.get(n)
        if not verdict or verdict.verdict != "conflict":
            continue
        disputes.append(
            {
                "row": pair["row"],
                "text": dispute_text(verdict.sentence, pair, article_id),
                "quote": pair["sentence"],
            }
        )
    report["date_disputes"] = len(disputes)
    report["date_verdicts"] = [
        v.verdict if (v := verdicts.get(n)) else "missing"
        for n in range(1, len(pairs) + 1)
    ]
    return disputes, report
