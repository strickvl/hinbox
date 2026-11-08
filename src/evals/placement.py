"""Score which starter heading a model files profile sentences under.

``just eval placement --model <m>`` reads hand labels from
``{eval_dir}/placement_labels.jsonl``, one row per sentence::

    {"entity": "Vance Spath", "entity_type": "people", "kind": "default",
     "sentence": "Spath suspended the USS Cole trial ...^[id]",
     "source": "r22b-qwen", "placed": "Background > Post-military career",
     "gold": "Cases and decisions", "confidence": "sure"}

Why: whether a fix files sentences better was judged by 40-article replays
and a hand check of 10 sentences per person per run. On the same code Qwen's
right count swung from 7 to 18 between runs, so a real gain could hide in
the noise. Here every approach sees the same sentences, in the same batches,
and only ``sure`` rows are scored.

Approaches (``--placement-approaches``):

- ``reorganise``: the reorganise prompt (``section_reorganise.messages()``)
  shown each batch as numbered paragraphs, as it is shown a flat profile.
- ``classify``: ``src/engine/section_place.py``, placement alone, a few
  words on what each sentence tells about the entity before its heading.

The report has accuracy over sure rows, per gold heading, the most common
mistakes, and the baseline: how often the run a row came from had put it
under its gold heading (``placed``), per model. Rows are split into
train/dev/test by a hash of the sentence (``assign_split()``), so a prompt
tuned on dev is checked on test once at the end. Records hold sentences from
articles: they stay under ``{eval_dir}/runs`` and must not go to GitHub.
"""

import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.engine.profile_create import type_tag
from src.engine.section_place import make_asker as place_asker
from src.engine.section_place import messages as place_messages
from src.engine.section_place import read_reply
from src.engine.section_reorganise import (
    _starter_for,
    starter_headings,
    starter_notes,
)
from src.engine.section_reorganise import (
    messages as reorganise_messages,
)
from src.engine.section_split import _clean_heading
from src.engine.section_split import make_asker as split_asker
from src.evals.datasets import assign_split
from src.logging_config import console
from src.utils.cache_utils import sha256_text

APPROACHES = ("reorganise", "classify")
LABELS_FILE = "placement_labels.jsonl"


def row_id(row: Dict[str, Any]) -> str:
    return sha256_text(f"{row['entity']}|{row['sentence']}")[:16]


def load_labels(
    path: str, config: Dict[str, Any], split: str = "all"
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Sure rows whose gold is one of their kind's starter headings, and
    counts of the rows left out (``unsure``, ``bad_gold``, ``other_split``)."""
    rows, skipped = [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            starter = starter_headings(config, row["entity_type"], row.get("kind"))
            gold = _starter_for(_clean_heading(str(row.get("gold") or "")), starter)
            if row.get("confidence", "sure") != "sure":
                skipped["unsure"] += 1
            elif gold is None:
                skipped["bad_gold"] += 1
            elif split != "all" and assign_split(row_id(row)) != split:
                skipped["other_split"] += 1
            else:
                rows.append({**row, "id": row_id(row), "gold": gold})
    return rows, dict(skipped)


def batches(
    rows: List[Dict[str, Any]], size: int, seed: int = 0
) -> List[List[Dict[str, Any]]]:
    """Each entity's rows shuffled (fixed seed) and cut into batches of
    ``size``, so every approach is asked about the same groups."""
    by_entity: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        by_entity.setdefault((row["entity"], row.get("kind") or ""), []).append(row)
    out = []
    for key in sorted(by_entity):
        group = sorted(by_entity[key], key=lambda r: r["id"])
        random.Random(f"{seed}|{key}").shuffle(group)
        out += [group[i : i + size] for i in range(0, len(group), size)]
    return out


def _ask_batch(
    approach: str, batch: List[Dict[str, Any]], config: Dict[str, Any], ask
) -> List[Optional[str]]:
    first = batch[0]
    starter = starter_headings(config, first["entity_type"], first.get("kind"))
    notes = starter_notes(config, first["entity_type"], first.get("kind"))
    tag, name = type_tag(first["entity_type"]), first["entity"]
    sentences = [r["sentence"] for r in batch]
    if approach == "classify":
        reply = ask(place_messages(sentences, starter, notes, tag, name))
        return read_reply(reply, len(batch), starter)
    reply = ask(reorganise_messages(sentences, starter, tag, name, notes))
    out: List[Optional[str]] = [None] * len(batch)
    for sub in reply.subsections:
        heading = _starter_for(_clean_heading(sub.heading), starter)
        for n in sub.paragraphs:
            if 1 <= n <= len(batch) and out[n - 1] is None:
                out[n - 1] = heading or f"(new) {_clean_heading(sub.heading)}"
    return out


def score_batches(
    approach: str,
    groups: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
    ask,
    workers: int = 4,
) -> List[Dict[str, Any]]:
    """One record per row: its gold heading, the answer, and whether the
    batch's call failed."""

    def run(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = time.time()
        try:
            answers, error = _ask_batch(approach, batch, config, ask), ""
        except Exception as e:  # scored as an error, not a wrong answer
            answers, error = [None] * len(batch), f"{type(e).__name__}: {e}"
        seconds = round(time.time() - start, 2)
        return [
            {
                "id": r["id"], "entity": r["entity"], "kind": r.get("kind"),
                "sentence": r["sentence"], "gold": r["gold"], "answer": a,
                "correct": a == r["gold"], "error": error, "seconds": seconds,
            }
            for r, a in zip(batch, answers)
        ]  # fmt: skip

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [rec for recs in pool.map(run, groups) for rec in recs]


def placed_heading(row: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """The starter heading at the top of the row's ``placed`` path."""
    top = str(row.get("placed") or "").split(">")[0]
    starter = starter_headings(config, row["entity_type"], row.get("kind"))
    return _starter_for(_clean_heading(top.strip()), starter)


def baseline(rows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, str]:
    """How often the runs had put rows under their gold heading, per model
    named in ``source`` (``luna``/``qwen``)."""
    counts: Dict[str, List[int]] = {}
    for row in rows:
        source = str(row.get("source") or "").lower()
        model = next((m for m in ("luna", "qwen") if m in source), "other")
        right = placed_heading(row, config) == row["gold"]
        hit = counts.setdefault(model, [0, 0])
        hit[0], hit[1] = hit[0] + right, hit[1] + 1
    return {m: f"{r}/{n}" for m, (r, n) in sorted(counts.items())}


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if not r["error"]]
    right = sum(r["correct"] for r in scored)
    per_heading: Dict[str, List[int]] = {}
    for r in scored:
        hit = per_heading.setdefault(r["gold"], [0, 0])
        hit[0], hit[1] = hit[0] + r["correct"], hit[1] + 1
    mistakes = Counter(
        f"{r['gold']} -> {r['answer'] or 'none'}" for r in scored if not r["correct"]
    )
    seconds = sorted(r["seconds"] for r in records)
    return {
        "n": len(records),
        "errors": len(records) - len(scored),
        "right": right,
        "accuracy": round(right / len(scored), 4) if scored else 0.0,
        "per_heading": {h: f"{a}/{b}" for h, (a, b) in sorted(per_heading.items())},
        "top_mistakes": dict(mistakes.most_common(6)),
        "median_seconds": seconds[len(seconds) // 2] if seconds else 0.0,
    }


def run_placement(args: Any, parser: Any, domain_cfg: Any, eval_dir: str) -> None:
    from src.evals.runner import save_run
    from src.utils.llm import configure_llm_concurrency

    path = args.labels or os.path.join(eval_dir, LABELS_FILE)
    if not os.path.exists(path):
        parser.error(f"Placement labels missing: {path}")
    config = domain_cfg.load_config()
    rows, skipped = load_labels(path, config, args.split)
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        parser.error(f"No sure placement labels in {path} for split {args.split}")
    console.print(f"{len(rows)} sure rows; left out: {skipped or 'none'}")
    console.print(f"Where the runs had put them (right/rows): {baseline(rows, config)}")
    configure_llm_concurrency(
        cloud_in_flight=args.workers, local_in_flight=args.workers
    )
    local = args.model.startswith("ollama/")
    groups = batches(rows, args.batch, args.seed)
    for approach in args.placement_approaches.split(","):
        approach = approach.strip()
        if approach not in APPROACHES:
            parser.error(f"Unknown placement approach {approach!r}: {APPROACHES}")
        maker = place_asker if approach == "classify" else split_asker
        ask = maker(args.model, local, step=f"eval.placement.{approach}")
        console.print(
            f"Placing {len(rows)} sentences with {approach} on {args.model} …"
        )
        records = score_batches(approach, groups, config, ask, args.workers)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": "placement", "dataset": "placement", "model": args.model,
            "approach": approach, "split": args.split, "batch": args.batch,
            "seed": args.seed, **summarize(records),
        }  # fmt: skip
        out = save_run(eval_dir, summary, records)
        console.print(
            json.dumps({k: v for k, v in summary.items() if k != "timestamp"}, indent=1)
        )
        console.print(f"Answers: {out}")
