"""Score which sub-heading a model files a section's new sentences under.

``just eval subplacement --model <m> --labels <file> [--entities-dir <dir>]``
reads hand labels, one row per sentence::

    {"entity": "Barack Obama", "entity_type": "people", "kind": "government",
     "sentence": "Trump's executive order revoked ...^[id]",
     "top_heading": "Cases and decisions",
     "subheadings": ["Guantanamo detention policy", "Military commissions"],
     "placed": "Cases and decisions", "gold": "Guantanamo detention policy",
     "confidence": "sure", "source": "r31-qwen"}

``gold`` is one of ``subheadings``, or ``(above)`` (it belongs at the top of
the section, above its sub-headings) or ``(none fits)`` (no sub-heading
fits). Both count as right when the model answers "None of these", which
leaves the sentence at the top of the section: that is the best the step in
``src/engine/section_subplace.py`` can do with it.

Why: in the 9 Nov 150-article local run 46 of 55 misplaced sentences were
under the right top-level heading but the wrong sub-heading (or above the
sub-headings). A rerun takes 6 hours and its hand check swings from run to
run; here every approach sees the same sentences in the same batches.

Approaches (``--placement-approaches``):

- ``names``: the sub-headings by name alone.
- ``examples``: each sub-heading with the first words of the first cited
  sentence under it in the entity's profile (``section_subplace.describe()``,
  the row's own sentence skipped), read from ``--entities-dir``. This is
  what the pipeline shows for profiles split before the split's notes were
  stored.

The baseline is how often the run had already put the row under its gold
place (``placed``). Records hold sentences from articles: they stay under
``{eval_dir}/runs`` and must not go to GitHub.
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
from src.engine.section_place import STAY
from src.engine.section_place import make_asker as place_asker
from src.engine.section_split import _clean_heading
from src.engine.section_subplace import ask_subheadings, describe
from src.logging_config import console
from src.utils.cache_utils import sha256_text

APPROACHES = ("names", "examples")
STAY_GOLDS = ("(above)", "(none fits)")


def row_id(row: Dict[str, Any]) -> str:
    return sha256_text(f"{row['entity']}|{row['sentence']}")[:16]


def gold_answer(row: Dict[str, Any]) -> Optional[str]:
    """The answer that counts as right: a sub-heading, or ``STAY``."""
    gold = str(row.get("gold") or "").strip()
    if gold in STAY_GOLDS:
        return STAY
    subs = [_clean_heading(s) for s in row.get("subheadings") or []]
    gold = _clean_heading(gold.split(">")[-1])
    return gold if gold in subs else None


def placed_answer(row: Dict[str, Any]) -> Optional[str]:
    """Where the run had put it, in the same terms as ``gold_answer()``."""
    parts = [_clean_heading(p) for p in str(row.get("placed") or "").split(">")]
    top = _clean_heading(str(row.get("top_heading") or ""))
    if parts and parts[0] != top:
        return "(other section)"
    return parts[1] if len(parts) > 1 else STAY


def load_labels(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Sure rows with a usable gold, and counts of rows left out."""
    rows, skipped = [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gold = gold_answer(row)
            if row.get("confidence", "sure") != "sure":
                skipped["unsure"] += 1
            elif gold is None or len(row.get("subheadings") or []) < 2:
                skipped["bad_gold"] += 1
            else:
                rows.append({**row, "id": row_id(row), "gold": gold})
    return rows, dict(skipped)


def load_profiles(entities_dir: Optional[str]) -> Dict[str, str]:
    """Profile text by entity name, from ``{entities_dir}/people.parquet``
    and the other tables."""
    import pyarrow.parquet as pq

    from src.evals.profile_create import ENTITY_TABLES, _entity_name

    texts: Dict[str, str] = {}
    for table in ENTITY_TABLES if entities_dir else ():
        path = os.path.join(entities_dir, f"{table}.parquet")
        if not os.path.exists(path):
            continue
        for row in pq.read_table(path).to_pylist():
            profile = row.get("profile")
            if isinstance(profile, str):
                profile = json.loads(profile or "{}")
            text = (profile or {}).get("text")
            if _entity_name(row) and text:
                texts.setdefault(_entity_name(row), text)
    return texts


def batches(
    rows: List[Dict[str, Any]], size: int, seed: int = 0
) -> List[List[Dict[str, Any]]]:
    """Each (entity, section)'s rows shuffled (fixed seed) and cut into
    batches of ``size``: one call only offers one section's sub-headings."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["entity"], row["top_heading"]), []).append(row)
    out = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["id"])
        random.Random(f"{seed}|{key}").shuffle(group)
        out += [group[i : i + size] for i in range(0, len(group), size)]
    return out


def _described(
    approach: str, batch: List[Dict[str, Any]], profiles: Dict[str, str]
) -> List[Tuple[str, str]]:
    first = batch[0]
    subs = [_clean_heading(s) for s in first["subheadings"]]
    if approach == "names":
        return [(s, "") for s in subs]
    text = profiles.get(first["entity"], "")
    return describe(text, subs, {}, [r["sentence"] for r in batch])


def score_batches(
    approach: str,
    groups: List[List[Dict[str, Any]]],
    profiles: Dict[str, str],
    ask,
    workers: int = 4,
) -> List[Dict[str, Any]]:
    def run(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = time.time()
        first = batch[0]
        try:
            answers, error = ask_subheadings(
                [r["sentence"] for r in batch], _described(approach, batch, profiles),
                _clean_heading(first["top_heading"]), type_tag(first["entity_type"]),
                first["entity"], ask,
            ), ""  # fmt: skip
        except Exception as e:  # scored as an error, not a wrong answer
            answers, error = [None] * len(batch), f"{type(e).__name__}: {e}"
        seconds = round(time.time() - start, 2)
        return [
            {
                "id": r["id"], "entity": r["entity"], "kind": r.get("kind"),
                "sentence": r["sentence"], "gold": r["gold"], "answer": a,
                "placed": placed_answer(r), "correct": a == r["gold"],
                "error": error, "seconds": seconds,
            }
            for r, a in zip(batch, answers)
        ]  # fmt: skip

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [rec for recs in pool.map(run, groups) for rec in recs]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in records if not r["error"]]

    def share(rows: List[Dict[str, Any]], key: str) -> str:
        return f"{sum(r[key] == r['gold'] for r in rows)}/{len(rows)}"

    named = [r for r in scored if r["gold"] != STAY]
    stay = [r for r in scored if r["gold"] == STAY]
    right = sum(r["correct"] for r in scored)
    mistakes = Counter(
        f"{'sub' if r['gold'] != STAY else 'stay'} -> "
        f"{'stay' if r['answer'] == STAY else 'none' if r['answer'] is None else 'other sub'}"
        for r in scored
        if not r["correct"]
    )
    seconds = sorted(r["seconds"] for r in records)
    return {
        "n": len(records),
        "errors": len(records) - len(scored),
        "right": right,
        "accuracy": round(right / len(scored), 4) if scored else 0.0,
        "right_named_gold": share(named, "answer"),
        "right_stay_gold": share(stay, "answer"),
        # Where the run had put the same rows: what the step must beat
        "baseline": share(scored, "placed"),
        # What the pipeline ends with: an unusable answer leaves it in place
        "after_step": f"{sum((r['answer'] or r['placed']) == r['gold'] for r in scored)}/{len(scored)}",
        "mistakes": dict(mistakes.most_common()),
        "median_seconds": seconds[len(seconds) // 2] if seconds else 0.0,
    }


def run_subplacement(args: Any, parser: Any, eval_dir: str) -> None:
    from src.evals.runner import save_run
    from src.utils.llm import configure_llm_concurrency

    if not args.labels or not os.path.exists(args.labels):
        parser.error(f"Sub-heading labels missing: {args.labels}")
    rows, skipped = load_labels(args.labels)
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        parser.error(f"No sure sub-heading labels in {args.labels}")
    console.print(f"{len(rows)} sure rows; left out: {skipped or 'none'}")
    approaches = [a.strip() for a in args.placement_approaches.split(",")]
    if approaches == ["reorganise", "classify"]:  # the placement eval's default
        approaches = list(APPROACHES)
    for approach in approaches:
        if approach not in APPROACHES:
            parser.error(f"Unknown sub-heading approach {approach!r}: {APPROACHES}")
    profiles = load_profiles(args.entities_dir) if "examples" in approaches else {}
    missing = {r["entity"] for r in rows} - set(profiles)
    if "examples" in approaches and missing:
        parser.error(f"No profile in --entities-dir for: {sorted(missing)}")
    configure_llm_concurrency(
        cloud_in_flight=args.workers, local_in_flight=args.workers
    )
    local = args.model.startswith("ollama/")
    groups = batches(rows, args.batch, args.seed)
    for approach in approaches:
        ask = place_asker(args.model, local, step=f"eval.subplacement.{approach}")
        console.print(
            f"Placing {len(rows)} sentences with {approach} on {args.model} …"
        )
        records = score_batches(approach, groups, profiles, ask, args.workers)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": "subplacement", "dataset": "subplacement", "model": args.model,
            "approach": approach, "batch": args.batch, "seed": args.seed,
            **summarize(records),
        }  # fmt: skip
        out = save_run(eval_dir, summary, records)
        console.print(
            json.dumps({k: v for k, v in summary.items() if k != "timestamp"}, indent=1)
        )
        console.print(f"Answers: {out}")
