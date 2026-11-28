"""The citation check with the last call taken from the model
(``HINBOX_GROUNDING_DECIDE=differences``, off by default).

On the citation test set (28 Nov) the tuned and untuned Inkling checkers
often found the exact detail a wrong sentence gets wrong and then passed it
anyway: "'leading to' is a slight causal gloss", "'was' vs. 'will be' is a
minor tense shift", "a fair paraphrase". The model was making two decisions
in one breath: what differs, and whether that matters.

This splits them. The judge first lists, in ``differences``, every detail of
the claim the article doesn't state the same way, however small, without
judging it; then writes its reasoning and verdict as before. Code then
decides: a claim with any difference listed is never "supported" (it
becomes "partial"); a claim with none keeps the model's verdict. So a
"supported" verdict now means the model found nothing to list.

The cost is that a model listing differences of wording alone will doubt
good claims. Measure both rates before turning it on, on sentences that
weren't used to write it (the citation test set was read to find the
pattern).
"""

import os
import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from src.utils.grounding import SupportLevel  # grounding imports this module lazily

GROUNDING_DECIDE = os.getenv("HINBOX_GROUNDING_DECIDE", "model").strip().lower()

DIFFERENCES_INSTRUCTION = """For each claim, first fill differences: list every detail of the claim that the article does not state in the same way, each in a few words. Count: a fact, number, date, name, title, place or person the article does not give or gives differently; a cause, consequence, motive, "first" or judgment the article does not make; more certainty than the article has (a dropped "allegedly" or "expected", something planned or scheduled written as done, "will be" written as "was"); a dropped attribution; a detail the article gives for someone else. List each one however small it seems, and do not decide here whether it matters. The same fact in other words is not a difference. Leave differences empty when there are none. Then write the reasoning: quote or name the article passage you compared the claim with (at most 40 words). Then give the support_level. A claim with any difference listed is not "supported". Return one verdict for every claim, with its claim_number. Do not repeat the claim text."""

_REPLY_START = "For each claim, first write the reasoning"


def decides() -> bool:
    return GROUNDING_DECIDE == "differences"


def with_differences(prompt: str) -> str:
    """``prompt`` with its reply paragraph (the last one, starting with
    ``_REPLY_START``) replaced by ``DIFFERENCES_INSTRUCTION``."""
    head, sep, _ = prompt.rpartition(_REPLY_START)
    if not sep:
        raise ValueError("judge prompt has no reply paragraph to replace")
    return head + DIFFERENCES_INSTRUCTION


class ClaimVerdictWithDifferences(BaseModel):
    """A verdict whose differences are listed first, before the reasoning
    and the verdict, so the list is written before any decision."""

    claim_number: int
    differences: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    support_level: SupportLevel


class ClaimVerdictWithDifferencesAndPassage(ClaimVerdictWithDifferences):
    """The same, plus the copied passage when ``HINBOX_GROUNDING_PASSAGE`` asks
    for one, after the verdict as in ``ClaimVerdictWithPassage``."""

    passage: Optional[str] = None


def verdict_model(with_passage: bool, plain: Any, with_copy: Any) -> Any:
    """The reply schema for one verdict: ``plain`` or ``with_copy`` (the
    passage schema) normally, their differences versions in this mode."""
    if decides():
        return (
            ClaimVerdictWithDifferencesAndPassage
            if with_passage
            else ClaimVerdictWithDifferences
        )
    return with_copy if with_passage else plain


_NONE = re.compile(r"^(none|no differences?( found)?|n/?a|nothing|-|\[\])\W*$", re.I)


def real_differences(items: Any) -> List[str]:
    """The listed differences, less blanks and "none"."""
    if isinstance(items, str):
        items = [items]
    out = []
    for d in items or []:
        text = str(d).strip(" \n-*•.;")
        if text and not _NONE.match(text):
            out.append(text)
    return out


def decided_level(level: Any, differences: Any) -> Any:
    """``level``, except that "supported" with a difference listed becomes
    "partial" (as a ``SupportLevel`` when given one, else as text)."""
    value = getattr(level, "value", level)
    if value == "supported" and real_differences(differences):
        return SupportLevel.PARTIAL if isinstance(level, SupportLevel) else "partial"
    return level


def reasoning_with(differences: Any, reasoning: Optional[str]) -> Optional[str]:
    """The reasoning stored with a verdict, led by the differences listed, so
    the reader shows what the verdict rests on."""
    found = real_differences(differences)
    if not found:
        return reasoning
    lead = "Differences: " + "; ".join(found) + "."
    return f"{lead} {reasoning}" if reasoning else lead


# A "Differences:" header at the start of a line (after a claim number, if
# any: "1. Differences: ..."), so a reasoning that only
# mentions "the difference" in an older reply is never read as a list
_PLAIN_DIFF = re.compile(
    r"(?:^|\n)[\W_]*(?:claim\s*)?\d*[\W_]*differences?[*_]*\s*:"
    r"(.*?)(?=\n[\W_]*(?:claim\s*)?\d*[\W_]*(?:reasoning|support[ _]level)|$)",
    re.I | re.S,
)


def plain_differences(segment: str) -> List[str]:
    """Differences from one claim's part of a reply written as text
    ("Differences: - a  - b  Reasoning: ...")."""
    m = _PLAIN_DIFF.search(segment or "")
    if not m:
        return []
    lines = re.split(r"\n|;|(?:^|\s)[-*•]\s", m.group(1))
    return real_differences(lines)
