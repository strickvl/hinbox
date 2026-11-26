"""Length limits on the edit reply a local model writes (``profile_edits.py``).

A local model sometimes never closes one string of its edit reply. In the
12 Nov full run, 255 of 24,942 edit replies ran to the 16,384-token limit:
a citation ID repeated without end inside a copied profile sentence
("-a999-9999-a999-…"), or the model arguing with itself inside
``article_says`` ("Wait, the profile says…"). The reply is then not valid
JSON, the update fails, and at temperature 0 every retry writes the same
reply, so the article never gets into that profile.

A local reply is constrained to a JSON schema, and the server (llama.cpp)
turns ``maxLength`` and ``maxItems`` into the grammar it samples from: when a
string reaches its limit the only thing the model can write next is the
closing quote. So the limits go into the schema the local model is given
(``LimitedEditList``), and the reply always ends.

A string cut this way stops in the middle of a word and has exactly its
limit's length, which is how ``without_cut_off()`` finds one. What it does:

- ``disagreements`` and ``new_facts`` are notes the model makes before its
  edits; they are never written into the profile, so a cut one is only
  counted.
- An edit whose ``anchor`` was cut is dropped: the model was looping there.
- An edit whose ``text`` was cut keeps its whole cited sentences (everything
  up to the last complete citation), unless one of them is there twice (a
  loop) or none is complete; then the edit is dropped.

No limit is over 2,000: llama.cpp refuses a grammar that repeats anything
more often ("number of repetitions exceeds sane defaults"). Cloud models get
the plain schema. ``HINBOX_EDIT_REPLY_LIMITS=off`` gives a local model the
plain schema too (for a server that refuses or ignores the limits).
"""

import os
import re
from typing import Any, Dict, List, Tuple

from src.engine.profile_edits import ProfileEdit, ProfileEditList

EDIT_LIMITS_ENV = "HINBOX_EDIT_REPLY_LIMITS"

# Characters. In the full run's 24,687 edit replies that ended normally, 999
# of 1,000 values were shorter than: profile_sentence 578, article_says 4,541
# (99 of 100: 1,088), a new_facts item 493, anchor 589 (longest 1,429), text
# 3,063 (99 of 100: 1,224). No anchor reached 1,500. 90 of 43,134 edit texts
# passed 2,000, one of them a loop; cut and trimmed to their last citation
# they would keep 1,104 characters or more
PROFILE_SENTENCE_CHARS = 1000
ARTICLE_SAYS_CHARS = 1200
NEW_FACT_CHARS = 600
ANCHOR_CHARS = 1500
TEXT_CHARS = 2000
# Items per list (999 of 1,000 replies had at most 15, 21 and 21). The first
# two lists are only notes; the longest list of edits that ended normally had 51
MAX_DISAGREEMENTS = 10
MAX_NEW_FACTS = 30
MAX_EDITS = 50

_STRING_LIMITS = {
    ("Disagreement", "profile_sentence"): PROFILE_SENTENCE_CHARS,
    ("Disagreement", "article_says"): ARTICLE_SAYS_CHARS,
    ("ProfileEdit", "anchor"): ANCHOR_CHARS,
    ("ProfileEdit", "text"): TEXT_CHARS,
}
_LIST_LIMITS = {
    "disagreements": MAX_DISAGREEMENTS,
    "new_facts": MAX_NEW_FACTS,
    "edits": MAX_EDITS,
}

_CITATION_RUN = re.compile(r"(?:\^\[[^\]]*\][.,;:!?\"”’)]*)+")
# A piece this short ("He denied it.") can be in a text twice without a loop
_REPEAT_MIN_CHARS = 30


def limits_on() -> bool:
    return os.getenv(EDIT_LIMITS_ENV, "on").strip().lower() != "off"


def with_limits(schema: Dict[str, Any]) -> Dict[str, Any]:
    """``schema`` (of ``ProfileEditList``) with the limits written in."""
    defs = schema.get("$defs") or {}
    for (model, field), chars in _STRING_LIMITS.items():
        defs[model]["properties"][field]["maxLength"] = chars
    props = schema["properties"]
    props["new_facts"]["items"]["maxLength"] = NEW_FACT_CHARS
    for field, items in _LIST_LIMITS.items():
        props[field]["maxItems"] = items
    return schema


class LimitedEditList(ProfileEditList):
    """The same reply; only the JSON schema sent to the server differs.

    The limits are not Pydantic constraints: a server that ignores them must
    not have a long, complete reply refused when it is read."""

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        schema = with_limits(super().model_json_schema(*args, **kwargs))
        schema.pop("description", None)  # this class's docstring
        return schema


def _whole_sentences(text: str) -> str:
    """The part of a cut ``text`` worth keeping, or "" to drop the edit."""
    runs = list(_CITATION_RUN.finditer(text))
    if not runs:
        return ""
    kept = text[: runs[-1].end()].rstrip()
    pieces = [" ".join(p.split()) for p in _CITATION_RUN.split(kept)]
    long = [p for p in pieces if len(p) >= _REPEAT_MIN_CHARS]
    return "" if len(set(long)) < len(long) else kept


def without_cut_off(
    reply: ProfileEditList,
) -> Tuple[List[ProfileEdit], List[Dict[str, Any]]]:
    """(edits to apply, what was cut) for a reply written under the limits.

    Each cut entry names the field (``edits[2].text``) and, for an edit, what
    happened to it: ``dropped``, or ``kept_chars`` of its text."""
    cut: List[Dict[str, Any]] = []
    for i, d in enumerate(reply.disagreements):
        if len(d.profile_sentence) == PROFILE_SENTENCE_CHARS:
            cut.append({"field": f"disagreements[{i}].profile_sentence"})
        if len(d.article_says) == ARTICLE_SAYS_CHARS:
            cut.append({"field": f"disagreements[{i}].article_says"})
    for i, fact in enumerate(reply.new_facts):
        if len(fact) == NEW_FACT_CHARS:
            cut.append({"field": f"new_facts[{i}]"})
    edits: List[ProfileEdit] = []
    for i, edit in enumerate(reply.edits):
        if len(edit.anchor) == ANCHOR_CHARS:
            cut.append({"field": f"edits[{i}].anchor", "dropped": True})
            continue
        if len(edit.text) != TEXT_CHARS:
            edits.append(edit)
            continue
        kept = _whole_sentences(edit.text)
        if not kept:
            cut.append({"field": f"edits[{i}].text", "dropped": True})
            continue
        cut.append({"field": f"edits[{i}].text", "kept_chars": len(kept)})
        edits.append(edit.model_copy(update={"text": kept}))
    return edits, cut
