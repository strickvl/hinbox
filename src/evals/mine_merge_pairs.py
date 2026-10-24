"""Mine hard merge pairs from the existing entity tables for labelling.

The first merge gold set was mostly easy "no" pairs. This picks pairs of
*existing* entities that look alike (similar names, or similar profiles by
embedding) but were kept apart by the pipeline. Those are exactly the cases
where the match check has to think, so they make a useful test.

Usage::

    just eval-mine                 # writes {output_dir}/eval/merge_candidates_v2.jsonl
    just eval-mine --total 200

Each output row follows the ``merge_candidates.jsonl`` shape, with the full
profile text (capped) instead of a 200-character snippet, plus a ``split``
field. Rows in the ``test`` split are meant to be labelled blind.
"""

import argparse
import ast
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from src.config_loader import DomainConfig
from src.evals.datasets import assign_split
from src.logging_config import console

# Share of the total per entity type (people and organisations dominate the data)
TYPE_SHARES: Dict[str, float] = {
    "people": 0.32,
    "organizations": 0.28,
    "locations": 0.26,
    "events": 0.14,
}
_SINGULAR = {
    "people": "person",
    "organizations": "organization",
    "locations": "location",
    "events": "event",
}
MAX_PAIRS_PER_ENTITY = 3
PROFILE_CHAR_CAP = 1500


def _parse(value: Any) -> Any:
    """Entity columns are stored as Python reprs (e.g. "{'text': ...}")."""
    if not isinstance(value, str) or value[:1] not in "[{":
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_entities(output_dir: str, entity_type: str) -> List[Dict[str, Any]]:
    rows = pq.read_table(os.path.join(output_dir, f"{entity_type}.parquet")).to_pylist()
    entities = []
    for row in rows:
        profile = _parse(row.get("profile")) or {}
        name = row.get("title") if entity_type == "events" else row.get("name")
        aliases = _parse(row.get("aliases")) or []
        alt = (
            _parse(
                row.get(
                    "alternative_titles"
                    if entity_type == "events"
                    else "alternative_names"
                )
            )
            or []
        )
        embedding = row.get("profile_embedding")
        if not name or not embedding:
            continue
        entities.append(
            {
                "name": str(name),
                "names": [str(name)] + [str(a) for a in [*aliases, *alt] if a],
                "text": str(profile.get("text", ""))
                if isinstance(profile, dict)
                else "",
                "embedding": np.asarray(embedding, dtype=np.float32),
            }
        )
    return entities


def _name_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return max(fuzz.WRatio(x, y) for x in a["names"] for y in b["names"])


def score_pairs(entities: List[Dict[str, Any]]) -> List[Tuple[int, int, float, float]]:
    """All within-type pairs as ``(i, j, name_similarity 0-100, cosine)``."""
    matrix = np.stack([e["embedding"] for e in entities])
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    cosine = matrix @ matrix.T
    return [
        (i, j, _name_similarity(entities[i], entities[j]), float(cosine[i, j]))
        for i in range(len(entities))
        for j in range(i + 1, len(entities))
    ]


def pick_hard_pairs(
    pairs: List[Tuple[int, int, float, float]], quota: int, rng: random.Random
) -> List[Tuple[int, int, float, float]]:
    """Mix three kinds of hard pair, capping how often one entity appears.

    The pipeline only asks the LLM about pairs whose names pass lexical
    blocking (WRatio >= 60) and whose profiles are close by embedding, so:

    - up to 15% near-identical names (score >= 95) kept as two entities,
      which are usually real duplicates the pipeline failed to merge
    - 50% would reach the match check today: names >= 60 and cosine >= 0.7
    - 20% near misses: names >= 60 with cosine 0.5-0.7, or very close
      profiles (cosine >= 0.85) whose names differ (acronyms, nicknames)
    - the rest: random draws from everything that passes lexical blocking
    """
    chosen: List[Tuple[int, int, float, float]] = []
    seen_pairs = set()
    uses: Dict[int, int] = {}

    def take(candidates, limit):
        for pair in candidates:
            if len(chosen) >= limit:
                return
            i, j = pair[0], pair[1]
            if (
                (i, j) in seen_pairs
                or uses.get(i, 0) >= MAX_PAIRS_PER_ENTITY
                or uses.get(j, 0) >= MAX_PAIRS_PER_ENTITY
            ):
                continue
            chosen.append(pair)
            seen_pairs.add((i, j))
            uses[i] = uses.get(i, 0) + 1
            uses[j] = uses.get(j, 0) + 1

    by_cosine = sorted(pairs, key=lambda p: -p[3])
    take([p for p in by_cosine if p[2] >= 95], round(quota * 0.15))
    take(
        [p for p in by_cosine if p[2] >= 60 and p[3] >= 0.7],
        len(chosen) + round(quota * 0.5),
    )
    take(
        [
            p
            for p in by_cosine
            if (p[2] >= 60 and 0.5 <= p[3] < 0.7) or (p[2] < 60 and p[3] >= 0.85)
        ],
        len(chosen) + round(quota * 0.2),
    )
    blocked = [p for p in pairs if p[2] >= 60]
    rng.shuffle(blocked)
    take(blocked, quota)
    return chosen


def mine(output_dir: str, total: int, seed: int = 13) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for entity_type, share in TYPE_SHARES.items():
        entities = load_entities(output_dir, entity_type)
        if len(entities) < 2:
            continue
        picked = pick_hard_pairs(score_pairs(entities), round(total * share), rng)
        for i, j, name_sim, cosine in picked:
            a, b = entities[i], entities[j]
            item_id = f"merge:{_SINGULAR[entity_type]}:{a['name']}|{b['name']}"
            rows.append(
                {
                    "entity_name": a["name"],
                    "candidate_name": b["name"],
                    "entity_type": _SINGULAR[entity_type],
                    "source": "mined",
                    "pipeline_decision": None,
                    "pipeline_confidence": None,
                    "similarity_score": round(name_sim, 1),
                    "embedding_cosine": round(cosine, 4),
                    "entity_context": a["text"][:PROFILE_CHAR_CAP],
                    "candidate_context": b["text"][:PROFILE_CHAR_CAP],
                    "split": assign_split(item_id),
                    "run_id": None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard merge pairs for labelling")
    parser.add_argument("--domain", default="guantanamo")
    parser.add_argument("--total", type=int, default=160)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--out",
        default=None,
        help="default: {output_dir}/eval/merge_candidates_v2.jsonl",
    )
    args = parser.parse_args()

    output_dir = DomainConfig(args.domain).get_output_dir()
    out = args.out or os.path.join(output_dir, "eval", "merge_candidates_v2.jsonl")
    rows = mine(output_dir, args.total, args.seed)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    blind = sum(1 for r in rows if r["split"] == "test")
    console.print(f"Wrote {len(rows)} pairs ({blind} in the blind test split) to {out}")


if __name__ == "__main__":
    main()
