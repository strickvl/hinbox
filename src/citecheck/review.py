"""Second look at the items where two labellers disagree.

``review_items()`` picks the test items whose labels differ between the
person's file and another labeller's (an agent's), and attaches both earlier
answers and the passages the other labeller quoted, so the labelling page can
show them beside the article. The person's new answers go to their own file;
``combine_labels()`` then writes one file from several, a later file's label
winning over an earlier one's for the same item.
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from src.citecheck.items import latest_labels, read_jsonl

_QUOTES = "\"'“”‘’„«»"
_SPLIT = re.compile(r"\s*(?:…|\.\.\.|\n|\s\|\s|\";\s*\"|”;\s*“)\s*")


def evidence_quotes(evidence: Any) -> List[str]:
    """The separate quoted passages in a labeller's ``evidence`` field (a
    string with passages split by "…", "...", newlines or " | ", or a list)."""
    parts = (
        evidence if isinstance(evidence, list) else _SPLIT.split(str(evidence or ""))
    )
    out = []
    for part in parts:
        text = str(part).strip().strip(_QUOTES).strip()
        if len(text) >= 15:
            out.append(text)
    return out


def _pattern(quote: str) -> "re.Pattern[str]":
    words = []
    for word in quote.split():
        bits = [
            f"[{re.escape(_QUOTES)}]" if c in _QUOTES else re.escape(c) for c in word
        ]
        words.append("".join(bits))
    return re.compile(r"\s+".join(words), re.IGNORECASE)


def quote_pieces(article: str, quotes: List[str]) -> List[Tuple[str, bool]]:
    """The article as (text, marked) pieces with each quoted passage found in
    it marked; every character kept, in order. A quote not found word for word
    (spacing, case and quote marks aside) is not marked."""
    spans = []
    for quote in quotes:
        found = _pattern(quote).search(article)
        if found:
            spans.append(found.span())
            continue
        # A quote that joins sentences the article keeps apart: try each one
        for part in re.split(r"(?<=[.!?])\s+", quote):
            hit = _pattern(part).search(article) if len(part) >= 15 else None
            if hit:
                spans.append(hit.span())
    pieces: List[Tuple[str, bool]] = []
    at = 0
    for start, end in sorted(spans):
        if end <= at:
            continue
        start = max(start, at)
        pieces += [(article[at:start], False), (article[start:end], True)]
        at = end
    pieces.append((article[at:], False))
    return [(text, mark) for text, mark in pieces if text]


def review_items(
    items: List[Dict[str, Any]],
    mine_path: str,
    other_path: str,
    other_name: str = "",
) -> List[Dict[str, Any]]:
    """Items labelled in both files with different labels, in item order, each
    with ``earlier`` (both answers and notes) and ``evidence`` (the other
    labeller's quotes)."""
    mine, other = latest_labels(mine_path), latest_labels(other_path)
    out = []
    for item in items:
        a, b = mine.get(item["item_id"]), other.get(item["item_id"])
        if not a or not b or a["label"] == b["label"]:
            continue
        name = other_name or b.get("labeller") or "The other labeller"
        out.append(
            {
                **item,
                "earlier": [
                    {"who": "You", "label": a["label"], "note": a.get("note", "")},
                    {"who": name, "label": b["label"], "note": b.get("note", "")},
                ],
                "evidence": evidence_quotes(b.get("evidence")),
            }
        )
    return out


def combine_labels(paths: List[str], out: str) -> Dict[str, int]:
    """One label file from several: every row of each file in turn, so the
    last file holding an item decides its label. Counts per label."""
    rows = [row for path in paths if os.path.exists(path) for row in read_jsonl(path)]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: Dict[str, int] = {}
    for row in latest_labels(out).values():
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts
