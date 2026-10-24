"""CLI: ``uv run python -m src.evals <task> --model <model>`` (or ``just eval``).

Examples::

    just eval relevance --model gemini/gemini-2.0-flash
    just eval merge --model ollama/qwen3.6:35b-a3b --split test
    just eval profile-update --model openai/gpt-6-luna --split dev
    just eval show                       # print the scoreboard
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from rich.table import Table

from src.config_loader import DomainConfig
from src.constants import CLOUD_MODEL
from src.evals.datasets import (
    MERGE_SETS,
    SPLITS,
    filter_split,
    load_merge_items,
    load_relevance_items,
)
from src.evals.profile_update import (
    UPDATERS,
    load_cases,
    run_updates,
    summarize_updates,
)
from src.evals.runner import (
    TASKS,
    load_scoreboard,
    make_pipeline_predictor,
    run_eval,
    save_run,
    summarize,
)
from src.logging_config import console
from src.utils.llm import configure_llm_concurrency

_COLUMNS = (
    ("dataset", "Dataset"),
    ("approach", "Approach"),
    ("model", "Model"),
    ("split", "Split"),
    ("n", "N"),
    ("errors", "Err"),
    ("balanced_accuracy", "Bal. acc"),
    ("accuracy", "Acc"),
    ("majority_baseline", "Always-majority"),
    ("f1", "F1"),
    ("median_seconds", "Median s"),
)
_UPDATE_COLUMNS = (
    ("approach", "Approach"),
    ("model", "Model"),
    ("split", "Split"),
    ("n", "N"),
    ("errors", "Err"),
    ("pass_rate", "Pass"),
    ("pass_new_fact", "New fact"),
    ("pass_irrelevant", "Irrelevant"),
    ("pass_disputed", "Disputed"),
    ("collapses", "Collapses"),
    ("median_retention", "Kept facts"),
    ("median_unchanged", "Unchanged"),
    ("invented_citations", "Invented cites"),
    ("dispute_claims", '"Sources differ"'),
    ("median_seconds", "Median s"),
)


def _print_rows(rows: List[Dict[str, Any]], title: str, columns=_COLUMNS) -> None:
    table = Table(title=title)
    for _, header in columns:
        table.add_column(header)
    for row in rows:
        table.add_row(*(str(row.get(key, "")) for key, _ in columns))
    console.print(table)


def _run_profile_update(args, parser, eval_dir: str) -> None:
    try:
        cases = load_cases(eval_dir, args.split, args.include_unreviewed)
    except FileNotFoundError as e:
        parser.error(f"Cases missing (run `just eval-cases`): {e.filename}")
    cases = cases[: args.limit]
    if not cases:
        parser.error(f"No profile-update cases for split {args.split!r}")
    configure_llm_concurrency(
        cloud_in_flight=args.workers, local_in_flight=args.workers
    )
    updater = UPDATERS[args.approach](args.model, args.domain)
    console.print(
        f"Updating {len(cases)} profiles with {args.model} ({args.approach}) …"
    )
    records = run_updates(cases, updater, workers=args.workers)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "profile_update",
        "dataset": "profile_update",
        "model": args.model,
        "approach": args.approach,
        "split": args.split,
        **summarize_updates(records),
    }
    path = save_run(eval_dir, summary, records)
    _print_rows([summary], "This run", _UPDATE_COLUMNS)
    console.print(f"Updated profiles and scores: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score models on hand-labelled gold sets"
    )
    parser.add_argument("task", choices=(*TASKS, "profile-update", "show"))
    parser.add_argument(
        "--model", default=CLOUD_MODEL, help="e.g. ollama/qwen3.6:35b-a3b"
    )
    parser.add_argument("--domain", default="guantanamo")
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument(
        "--merge-set", choices=MERGE_SETS, default="v2", help="merge gold set"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="score only the first N items"
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel model calls")
    parser.add_argument("--eval-dir", default=None, help="default: {output_dir}/eval")
    parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="profile-update: also score cases whose labels need review",
    )
    parser.add_argument(
        "--approach",
        choices=("pipeline", "edit"),
        default="pipeline",
        help="profile-update: the pipeline's whole-profile rewrite, or "
        "apply a list of edits",
    )
    args = parser.parse_args()

    domain_cfg = DomainConfig(args.domain)
    eval_dir = args.eval_dir or os.path.join(domain_cfg.get_output_dir(), "eval")

    if args.task == "show":
        rows = load_scoreboard(eval_dir)
        updates = [r for r in rows if r.get("task") == "profile_update"]
        others = [r for r in rows if r.get("task") != "profile_update"]
        _print_rows(others, f"Scoreboard ({eval_dir})")
        if updates:
            _print_rows(updates, "Profile updates", _UPDATE_COLUMNS)
        return

    if args.task == "profile-update":
        _run_profile_update(args, parser, eval_dir)
        return

    if args.task == "relevance":
        dataset = "relevance"
        load = lambda: load_relevance_items(eval_dir, domain_cfg.get_data_path())  # noqa: E731
    else:
        dataset = f"merge_{args.merge_set}"
        load = lambda: load_merge_items(eval_dir, args.merge_set)  # noqa: E731
    try:
        items = filter_split(load(), args.split)[: args.limit]
    except FileNotFoundError as e:
        parser.error(f"Gold file missing: {e.filename}")
    if not items:
        parser.error(f"No labelled {dataset} items found in {eval_dir}")

    configure_llm_concurrency(
        cloud_in_flight=args.workers, local_in_flight=args.workers
    )
    predictor = make_pipeline_predictor(args.task, args.model, args.domain)
    console.print(f"Scoring {len(items)} {args.task} items with {args.model} …")
    records = run_eval(items, predictor, workers=args.workers)
    summary = summarize(
        records,
        task=args.task,
        model=args.model,
        split=args.split,
        approach="pipeline",
        dataset=dataset,
    )
    path = save_run(eval_dir, summary, records)
    _print_rows([summary], "This run")
    console.print(f"Predictions: {path}")


if __name__ == "__main__":
    main()
