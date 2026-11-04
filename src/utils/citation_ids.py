"""Snap garbled citation IDs in a profile back to the article IDs it really has.

Local models sometimes copy an article ID with a chunk missing or repeated:
the profile cites ``^[62cdb2ad-18d6-4de6-a5858c1fb8ac]`` when the article is
``62cdb2ad-18d6-4de6-a599-a5858c1fb8ac``. That citation then points at no
article, so the frontend shows a dead link and grounding reports the claim as
having no source. A cited ID that isn't one of the profile's own sources but is
a few characters away from exactly one of them is rewritten to that source.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Tuple

from rapidfuzz.distance import Levenshtein

_CITATION = re.compile(r"\^\[([^\]]+)\]")


def _cited_ids(text: str) -> List[str]:
    return [i.strip() for g in _CITATION.findall(text) for i in g.split(",")]


def _nearest(bad: str, known: List[str]) -> str:
    """The one known ID within a quarter of the ID's length in edits, or ""."""
    limit = max(4, len(bad) // 4)
    scored = sorted((Levenshtein.distance(bad, k), k) for k in known)
    close = [(d, k) for d, k in scored if d <= limit]
    if not close or (len(close) > 1 and close[1][0] == close[0][0]):
        return ""
    return close[0][1]


def citation_repairs(text: str, known_ids: Iterable[str]) -> Dict[str, str]:
    """Map each cited ID that is not in ``known_ids`` but is near one of them
    to that known ID. IDs too far from every known ID are left out."""
    known = sorted({str(k) for k in known_ids if k})
    repairs: Dict[str, str] = {}
    for cited in _cited_ids(text):
        if cited in known or cited in repairs or not known:
            continue
        nearest = _nearest(cited, known)
        if nearest:
            repairs[cited] = nearest
    return repairs


def apply_citation_repairs(text: str, repairs: Dict[str, str]) -> str:
    """Rewrite the IDs inside ``^[...]`` markers; nothing else is touched."""
    if not repairs:
        return text

    def fix(match: "re.Match[str]") -> str:
        ids = [i.strip() for i in match.group(1).split(",")]
        return "^[" + ", ".join(repairs.get(i, i) for i in ids) + "]"

    return _CITATION.sub(fix, text)


def _repair_value(value: Any, repairs: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return repairs.get(value, apply_citation_repairs(value, repairs))
    if isinstance(value, list):
        return [_repair_value(v, repairs) for v in value]
    if isinstance(value, dict):
        return {k: _repair_value(v, repairs) for k, v in value.items()}
    return value


def repair_profile_citations(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """Fix near-miss citation IDs in ``profile["text"]`` against its ``sources``.

    The fact-list updater's saved rows (``fact_rows_json``) hold the same
    sentences, so they get the same fixes; otherwise the next update would drop
    those rows for no longer matching the text. Returns the profile and the
    number of distinct IDs fixed.
    """
    text = str(profile.get("text") or "")
    repairs = citation_repairs(text, profile.get("sources") or [])
    if not repairs:
        return profile, 0
    profile["text"] = apply_citation_repairs(text, repairs)
    rows_json = profile.get("fact_rows_json")
    if isinstance(rows_json, str) and rows_json:
        try:
            rows = json.loads(rows_json)
        except ValueError:
            rows = None
        if rows is not None:
            profile["fact_rows_json"] = json.dumps(
                _repair_value(rows, repairs), ensure_ascii=False
            )
    if isinstance(profile.get("fact_rows"), list):
        profile["fact_rows"] = _repair_value(profile["fact_rows"], repairs)
    return profile, len(repairs)
