"""Ask the model whether a "sources disagree" sentence is a real dispute.

The code guards (``edit_guards.py``) only drop a "sources disagree" sentence
when no profile sentence gives another year, month or count. Almost every
profile sentence has some date, so on the 7 Nov 150-article Luna run 14 such
sentences got through, and only 2 were real (2,974 against 2,976 dead on
9/11; Baker convicted Oct. 31 or Nov. 1). The rest were:

- **Different steps**: "the appeals court stayed the case Nov. 7" written as
  disputing "the court will hear arguments Jan. 22"; "the fire started
  Wednesday" against "it jumped the fence Thursday".
- **The same fact in other words**: "Spath had no contempt authority" against
  "Spath had no authority to convict him alone".
- **An earlier state from an older article**: the run doesn't go in date
  order, so a Feb. 2018 article saying al Darbi "is slated to go home Feb. 20"
  reached a profile that already said he went in May. The date guard
  (``later_update()``) missed these because the old sentence had no citation,
  or the wording ("slated", "this differs from") wasn't one it knows.

Telling these apart takes reading, so each such sentence left after the code
guards goes to one model call (all of an update's together), with the old
sentence it disputes and both sources' dates. The model answers:

- ``conflict``: both can't be true. The sentence is kept as written.
- ``later``: a plan, count, post or status at another time. Replaced by the
  model's dated sentence with the article's fact alone.
- ``different``: another event or step, a fact the profile lacks. Replaced
  by the model's plain sentence of that fact.
- ``same``: what the profile already says, in other words. Dropped.

A replacement that still says sources disagree, or that repeats a profile
sentence, is dropped. If the call fails, the sentences are kept as they are.
``HINBOX_DISPUTE_CHECK=off`` turns the check off.
"""

import datetime as dt
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from src.logging_config import get_logger

logger = get_logger("engine.dispute_check")

DISPUTE_CHECK_ENV = "HINBOX_DISPUTE_CHECK"
_CITATION = re.compile(r"\^\[[^\]]*\]")
_ABOUT_SOURCE = re.compile(
    r"^(?:the|an?|this)\s+(?:[\w.,]+\s+){0,4}?(?:article|report|story)\s+"
    r"(?:says|said|states|stated|reports|reported|notes|noted)\s+(?:that\s+)?",
    re.I,
)

INSTRUCTIONS = """You check a profile of the {entity_type} "{entity_name}" \
that is built from news articles, one article at a time. Articles are not \
added in date order, so a new article can be older than the profile's sources.

Each item below is a sentence an editor wants to add, saying that the new \
article disagrees with a sentence already in the profile. Most such sentences \
are wrong: the two sentences are about different things, or both are true. \
Decide what each item really is:

- `conflict`: both describe the very same thing at the same time and give it \
different values (a date, a number, a name, an outcome), so both cannot be \
true. Example: one says the bombing killed 17 sailors, the other 19.
- `later`: a plan, schedule, count, post or status that was true at one time \
and changed at another, so both were true when written. Example: "he is \
slated to go home Feb. 20" in a February article, against "he was sent home \
in May".
- `different`: two different events, steps or details, even if related. \
Example: a court stayed the case on Nov. 7, and later set arguments for Jan. \
22; a fire started on Wednesday and crossed the fence on Thursday; two \
separate interviews.
- `same`: the same fact, in other words or with more detail. Example: "the \
judge had no contempt authority" and "the judge could not convict him alone".

Harder cases, from real articles:
- A count from a May article (10 charged) against a count from a July \
article (9 charged): `later`. Counts of people held, charged or on staff \
change between dates; different source dates with different counts are \
`later` unless both counts are given for the same day.
- Two reasons given for the same ruling, one fuller or worded differently \
("the defense could not probe the CIA years" against "protective orders \
put the defense at an unfair disadvantage"): `same`.
- Reporters interviewed someone "this week" in one article and on June 19 \
in another: `different` (two interviews), unless both name the same day.
- The same event on two different days, or the same death toll with two \
different numbers: `conflict`.

`conflict` needs the very same thing at the very same time. Do not answer \
`conflict` because the values would clash "if" both were about the same \
thing: decide whether they are. When in doubt, it is not a conflict.

For each item, first write `reasoning`: what the profile sentence says, what \
the article says, and whether both can be true. Then the verdict. For \
`later` and `different`, also write `sentence`: the article's fact as one \
plain sentence about {entity_name}, without comparing it to the profile and \
without words such as "instead", "disagree", "differ" or "rather than", and \
stating the fact itself rather than what "the article says". For \
`later`, say when it held, from the article's own date or wording ("In \
February 2018, al Darbi was slated to return to Saudi Arabia on Feb. 20."). \
No citations. For `conflict` and `same`, leave `sentence` empty."""


class DisputeVerdict(BaseModel):
    item: int = Field(description="The item's number")
    # Before the verdict, so the model reads both sides first
    reasoning: str = Field(default="")
    verdict: Literal["conflict", "later", "different", "same"]
    sentence: str = Field(
        default="",
        description="For later/different: the article's fact alone, no citation",
    )


class DisputeVerdicts(BaseModel):
    verdicts: List[DisputeVerdict] = Field(default_factory=list)

    @field_validator("verdicts", mode="before")
    @classmethod
    def _parse(cls, value: Any) -> Any:
        from src.engine.profile_edits import _parse_list_string

        if isinstance(value, str):  # a list sent as a string
            parsed = _parse_list_string(value)
            return value if parsed is None else parsed
        return value


def dispute_check_enabled() -> bool:
    return os.getenv(DISPUTE_CHECK_ENV, "on").strip().lower() not in (
        "off",
        "0",
        "false",
        "no",
    )


def _when(date: Optional[dt.date]) -> str:
    return f"{date:%B %d, %Y}".replace(" 0", " ") if date else "unknown date"


def items_prompt(items: List[Dict[str, Any]]) -> str:
    """One block per item: the profile sentence(s) with their sources' dates,
    and the sentence the editor wants to add with the article's date."""
    blocks = []
    for n, item in enumerate(items, 1):
        olds = "\n".join(
            f"- {_CITATION.sub('', o).strip()} (source dated {when})"
            for o, when in item["olds"]
        )
        blocks.append(
            f"Item {n}\nIn the profile:\n{olds or '- (no matching sentence)'}\n"
            f"New article, dated {_when(item['article_date'])}, the editor's "
            f"sentence: {_CITATION.sub('', item['sentence']).strip()}"
        )
    return "\n\n".join(blocks)


def _old_dates(olds: List[str], dates: Dict[str, dt.date]) -> List[Tuple[str, str]]:
    from src.engine.later_updates import cited_ids

    out = []
    for old in olds:
        known = sorted(dates[i] for i in cited_ids(old) if i in dates)
        out.append((old, ", ".join(_when(d) for d in known) or "unknown date"))
    return out


Ask = Callable[[str, str], DisputeVerdicts]


def check_disputes(
    found: List[Dict[str, Any]],
    ask: Ask,
    article_date: Optional[dt.date],
    dates: Dict[str, dt.date],
) -> Dict[int, DisputeVerdict]:
    """Verdicts by index into ``found`` (each ``{"sentence", "olds"}``). One
    call for all. Empty when the call fails, so the sentences stay."""
    if not found:
        return {}
    items = [
        {
            "sentence": f["sentence"],
            "olds": _old_dates(f["olds"], dates),
            "article_date": article_date,
        }
        for f in found
    ]
    try:
        answer = ask(INSTRUCTIONS, items_prompt(items))
    except Exception as e:  # keep the update; the sentences stay as written
        logger.warning(f"Dispute check failed, keeping sentences: {e}")
        return {}
    return {v.item - 1: v for v in answer.verdicts if 0 < v.item <= len(found)}


def replacement(
    verdict: DisputeVerdict, sentence: str, citations: str
) -> Optional[str]:
    """What goes in place of the dispute sentence: itself for a conflict,
    nothing for ``same``, else the model's sentence with the article's
    citations. None means drop."""
    from src.engine.edit_guards import _compares

    if verdict.verdict == "conflict":
        return sentence
    body = _CITATION.sub("", verdict.sentence).strip().rstrip(".")
    # "The April 17, 2018 article says Ring became commander" -> the fact
    body = _ABOUT_SOURCE.sub("", body)
    if verdict.verdict == "same" or len(body.split()) < 4 or _compares(body):
        return None
    if re.search(r"\b(?:instead|rather than|disagree\w*)\b", body, re.I):
        return None
    return f"{body[0].upper()}{body[1:]}.{citations}"
