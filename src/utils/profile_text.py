"""Mend two formatting faults in stored profile text, without a model.

1. **Escaped newlines.** A model reply sometimes carries ``\\n`` as the two
   characters backslash and n, and the text was stored that way. In the full
   run of 12 Nov, 81 sentences of the long people pages held them, so a
   heading showed mid-paragraph as "\\n\\n### Professional Duties" and the
   section never appeared (3 of the 11 pages hand-checked on 21 Nov).
2. **Doubled markers.** The same citation twice in a row (``fact^[a].^[a]``,
   823 sentences in the same pages) shows as two footnotes, "2.2". The second
   marker adds no claim: grounding reuses the claim before it
   (``_extract_cited_claims()``), so dropping it changes no claim text.

``repair_text()`` mends one text. ``load_existing_entities()`` mends every
saved profile as it loads a run's tables, so the run saves the mended text
and grounding checks it; profile QC mends each new or updated profile.
"""

import re
from typing import Any, Dict, Tuple

# The two characters "\" and "n", and the same for "\t"
_ESCAPED_NEWLINE = re.compile(r"\\n")
_ESCAPED_TAB = re.compile(r"\\t")
# A marker, optional punctuation, then the very same marker again
_DOUBLED = re.compile(r"(\^\[([^\]]+)\])([.,;:]?)[ \t]*\^\[\2\]")


def repair_text(text: str) -> str:
    """``text`` with escaped newlines made real and doubled markers dropped."""
    if not text:
        return text
    out = _ESCAPED_TAB.sub(" ", _ESCAPED_NEWLINE.sub("\n", text))
    while True:
        new = _DOUBLED.sub(r"\1\3", out)
        if new == out:
            return out
        out = new


def repair_profile(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """The profile with its text mended, and whether anything changed."""
    text = profile.get("text")
    if not isinstance(text, str):
        return profile, False
    fixed = repair_text(text)
    if fixed == text:
        return profile, False
    return {**profile, "text": fixed}, True
