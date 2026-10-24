"""Load hand-labelled gold sets into a common ``EvalItem`` shape.

Gold files live under ``{output_dir}/eval/`` (gitignored with the rest of
``data/``). Each task pairs a ``*_candidates.jsonl`` file (the inputs shown to
the annotator) with a ``*_gold.jsonl`` file (the labels).
"""

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from src.utils.cache_utils import sha256_text

SPLITS = ("train", "dev", "test", "all")

# Stable hash buckets: 60% train / 20% dev / 20% test
_SPLIT_BOUNDS: Tuple[Tuple[str, int], ...] = (("train", 60), ("dev", 80), ("test", 100))


@dataclass(frozen=True)
class EvalItem:
    """One labelled example. ``label`` is the correct boolean answer."""

    item_id: str
    label: bool
    inputs: Dict[str, Any] = field(default_factory=dict)


def assign_split(item_id: str) -> str:
    """Deterministically bucket an item into train/dev/test by hashing its id.

    The same item always lands in the same split, so an optimizer that only
    sees ``train`` can never leak ``test`` items into its prompts.
    """
    bucket = int(sha256_text(item_id)[:8], 16) % 100
    return next(name for name, bound in _SPLIT_BOUNDS if bucket < bound)


def filter_split(items: List[EvalItem], split: str) -> List[EvalItem]:
    if split == "all":
        return items
    return [item for item in items if assign_split(item.item_id) == split]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _latest_labels(rows: List[Dict[str, Any]], key_fn) -> Dict[Any, Dict[str, Any]]:
    """Keep the most recent label per key (annotators sometimes re-label)."""
    latest: Dict[Any, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda r: r.get("timestamp", "")):
        latest[key_fn(row)] = row
    return latest


def _clean_name(value: Any) -> str:
    """Candidate names were sometimes saved as a dict or its repr string."""
    if isinstance(value, dict):
        return str(value.get("name", value))
    if isinstance(value, str) and value.startswith("{") and "'name'" in value:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, dict) and "name" in parsed:
            return str(parsed["name"])
    return str(value)


# ── Relevance ──


def _load_article_texts(articles_path: Optional[str]) -> Dict[str, str]:
    if not articles_path or not os.path.exists(articles_path):
        return {}
    table = pq.read_table(articles_path, columns=["id", "content"])
    return dict(
        zip(table.column("id").to_pylist(), table.column("content").to_pylist())
    )


def load_relevance_items(
    eval_dir: str, articles_path: Optional[str] = None
) -> List[EvalItem]:
    """Relevance gold: ``relevant`` → True, ``irrelevant`` → False.

    Uses the full article text from the articles Parquet when available,
    falling back to the snippet the annotator saw.
    """
    candidates = {
        row["article_id"]: row
        for row in _read_jsonl(os.path.join(eval_dir, "relevance_candidates.jsonl"))
    }
    gold = _latest_labels(
        _read_jsonl(os.path.join(eval_dir, "relevance_gold.jsonl")),
        lambda r: r["article_id"],
    )
    texts = _load_article_texts(articles_path)

    items = []
    for article_id, row in gold.items():
        if row["label"] not in ("relevant", "irrelevant"):
            continue
        cand = candidates.get(article_id, {})
        text = texts.get(article_id) or cand.get("content_snippet", "")
        if not text:
            continue
        items.append(
            EvalItem(
                item_id=f"relevance:{article_id}",
                label=row["label"] == "relevant",
                inputs={"text": text, "title": cand.get("title", "")},
            )
        )
    return items


# ── Merge ──

# Gold files use singular entity types; the match checker expects plural.
_PLURAL_TYPES = {
    "person": "people",
    "organization": "organizations",
    "location": "locations",
    "event": "events",
}


MERGE_SETS = ("v1", "v2")


def load_merge_items(eval_dir: str, merge_set: str = "v2") -> List[EvalItem]:
    """Merge gold: ``yes`` → True, ``no`` → False, ``unsure`` dropped.

    ``v1`` is the first annotation round (mostly easy pairs, 200-character
    snippets). ``v2`` is the mined hard-pair set from ``just eval-mine``.
    """
    suffix = "" if merge_set == "v1" else f"_{merge_set}"

    def key(row: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            _clean_name(row["entity_name"]),
            _clean_name(row["candidate_name"]),
            row["entity_type"],
        )

    candidates = {
        key(row): row
        for row in _read_jsonl(
            os.path.join(eval_dir, f"merge_candidates{suffix}.jsonl")
        )
    }
    gold = _latest_labels(
        _read_jsonl(os.path.join(eval_dir, f"merge_gold{suffix}.jsonl")), key
    )

    items = []
    for (entity_name, candidate_name, entity_type), row in gold.items():
        if row["label"] not in ("yes", "no"):
            continue
        cand = candidates.get((entity_name, candidate_name, entity_type), {})
        items.append(
            EvalItem(
                item_id=f"merge:{entity_type}:{entity_name}|{candidate_name}",
                label=row["label"] == "yes",
                inputs={
                    "new_name": entity_name,
                    "existing_name": candidate_name,
                    "new_profile_text": cand.get("entity_context", ""),
                    "existing_profile_text": cand.get("candidate_context", ""),
                    "entity_type": _PLURAL_TYPES.get(entity_type, entity_type),
                },
            )
        )
    return items
