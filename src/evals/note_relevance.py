"""Score the "is this note about the entity?" step against hand labels.

``just eval note-relevance --model <m> --labels <file>`` reads one row per
note (a cited sentence of a profile)::

    {"entity": "Port Elm", "entity_type": "locations", "kind": "city",
     "heading": "Arrests", "sentence": "Tom Vale was 19 when ...^[id]",
     "gold": "other", "confidence": "sure"}

``gold`` is ``about``, ``context`` or ``other`` (``src/engine/note_relevance.py``
says what each means). Each entity's sure rows are asked about in label
order, one heading's rows at a time, exactly as the pipeline asks
(``tag_notes()``), so a sentence is judged beside its neighbours.

What matters is two numbers, since only ``other`` notes are hidden:

- ``caught``: of the notes labelled ``other``, the share answered ``other``
  (the reader hides them). Bar: about 80%.
- ``hidden_about``: of the notes labelled ``about``, the share answered
  ``other``: real facts about the entity the reader would hide. Bar: under 5%.

``hidden_context`` (``context`` notes hidden) is reported beside them; it is
a softer miss, since the sentence still shows when the reader asks. Rows are
split by a hash of entity and sentence (``assign_split()``). Records hold
sentences from articles: they stay under ``{eval_dir}/runs``, never GitHub.
"""

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from src.engine.note_relevance import VERDICTS, Ask, make_asker, tag_notes
from src.evals.datasets import assign_split
from src.logging_config import console
from src.utils.cache_utils import sha256_text
from src.utils.corrections import sentence_key

LABELS_FILE = "note_labels.jsonl"


def row_id(row: Dict[str, Any]) -> str:
    return sha256_text(f"{row['entity']}|{row['sentence']}")[:16]


def load_labels(
    path: str, split: str = "all"
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Sure rows with a known gold, and counts of the rows left out."""
    rows, skipped = [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gold = str(row.get("gold") or "").strip().lower()
            if row.get("confidence", "sure") != "sure":
                skipped["unsure"] += 1
            elif gold not in VERDICTS:
                skipped["bad_gold"] += 1
            elif split != "all" and assign_split(row_id(row)) != split:
                skipped["other_split"] += 1
            else:
                rows.append({**row, "id": row_id(row), "gold": gold})
    return rows, dict(skipped)


def by_entity(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Rows grouped by entity, label order kept."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["entity_type"], row["entity"]), []).append(row)
    return list(groups.values())


def score_entity(rows: List[Dict[str, Any]], ask: Ask) -> List[Dict[str, Any]]:
    """Tag one entity's rows as the pipeline would; one record per row."""
    items = [
        {"key": sentence_key(r["sentence"]), "text": sentence_key(r["sentence"]),
         "heading": str(r.get("heading") or "")}
        for r in rows
    ]  # fmt: skip
    first = rows[0]
    start = time.monotonic()
    tags, failed = tag_notes(
        items, first["entity_type"], first["entity"], ask, first.get("kind")
    )
    seconds = round(time.monotonic() - start, 2)
    out = []
    for row, item in zip(rows, items):
        tag = tags.get(item["key"]) or {}
        answer = tag.get("verdict")
        out.append({
            "id": row["id"], "entity": row["entity"], "sentence": row["sentence"],
            "gold": row["gold"], "answer": answer, "about": tag.get("about", ""),
            "correct": answer == row["gold"], "error": answer is None,
            "failed_calls": failed, "seconds": seconds,
        })  # fmt: skip
    return out


def _share(records: List[Dict[str, Any]], gold: str, answer: str) -> Dict[str, Any]:
    rows = [r for r in records if r["gold"] == gold and not r["error"]]
    hit = sum(r["answer"] == answer for r in rows)
    return {
        "n": len(rows),
        "hit": hit,
        "share": round(hit / len(rows), 4) if rows else 0.0,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if not r["error"]]
    confusion = Counter(f"{r['gold']} -> {r['answer']}" for r in scored)
    return {
        "n": len(records),
        "errors": len(records) - len(scored),
        "accuracy": round(sum(r["correct"] for r in scored) / len(scored), 4)
        if scored
        else 0.0,
        "caught": _share(records, "other", "other"),
        "hidden_about": _share(records, "about", "other"),
        "hidden_context": _share(records, "context", "other"),
        "gold_counts": dict(Counter(r["gold"] for r in records)),
        "confusion": dict(sorted(confusion.items())),
    }


def run_note_relevance_eval(args: Any, parser: Any, eval_dir: str) -> None:
    from src.evals.runner import save_run
    from src.utils.llm import configure_llm_concurrency

    path = args.labels or os.path.join(eval_dir, LABELS_FILE)
    if not os.path.exists(path):
        parser.error(f"Note labels missing: {path}")
    rows, skipped = load_labels(path, args.split)
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        parser.error(f"No sure note labels in {path} for split {args.split}")
    console.print(f"{len(rows)} sure rows; left out: {skipped or 'none'}")
    configure_llm_concurrency(
        cloud_in_flight=args.workers, local_in_flight=args.workers
    )
    ask = make_asker(args.model, args.model.startswith("ollama/"))
    console.print(f"Tagging {len(rows)} notes on {args.model} …")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        groups = pool.map(lambda g: score_entity(g, ask), by_entity(rows))
        records = [rec for recs in groups for rec in recs]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "note-relevance", "dataset": "note_relevance", "model": args.model,
        "approach": "pipeline", "split": args.split, **summarize(records),
    }  # fmt: skip
    out = save_run(eval_dir, summary, records)
    console.print(
        json.dumps({k: v for k, v in summary.items() if k != "timestamp"}, indent=1)
    )
    console.print(f"Answers: {out}")
