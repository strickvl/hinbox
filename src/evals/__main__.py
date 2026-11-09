"""CLI: ``uv run python -m src.evals <task> --model <model>`` (or ``just eval``).

Examples::

    just eval relevance --model openai/gpt-6-luna
    just eval merge --model ollama/qwen3.6:35b-a3b --split test
    just eval profile-update --model openai/gpt-6-luna --split dev
    just eval profile-update --model openai/gpt-6-luna --prompt-variant local
    just eval profile-create --model openai/gpt-6-luna --limit 40 --grounding
    just eval profile-update --rescore <run file>   # re-score saved profiles
    just eval placement --model ollama/local-qwen36-32k --split dev
    just eval subplacement --model ollama/local-qwen36-32k --labels <file>
    just eval show                       # print the scoreboard
    just eval show --grid                # latest score per model and prompt
"""

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from rich.table import Table

from src.config_loader import DomainConfig
from src.constants import CLOUD_MODEL
from src.evals.datasets import (
    MERGE_SETS,
    SPLITS,
    filter_split,
    load_merge_items,
    load_relevance_items,
    stub_existing_profiles,
)
from src.evals.placement import run_placement
from src.evals.profile_create import run_profile_create
from src.evals.profile_update import (
    UPDATERS,
    load_cases,
    rescore_run,
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
from src.evals.subplacement import run_subplacement
from src.logging_config import console
from src.utils.llm import configure_llm_concurrency
from src.utils.tracing import configure_tracing

_COLUMNS = (
    ("dataset", "Dataset"),
    ("approach", "Approach"),
    ("model", "Model"),
    ("prompt", "Prompt"),
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
    ("prompt", "Prompt"),
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
    ("unsupported_numbers", "Unsupported nums"),
    ("labelled", "Labelled"),
    ("core_missing", "Facts missed"),
    ("wrong_readings", "Wrong readings"),
    ("median_seconds", "Median s"),
)


def _print_rows(rows: List[Dict[str, Any]], title: str, columns=_COLUMNS) -> None:
    table = Table(title=title)
    for _, header in columns:
        table.add_column(header)
    for row in rows:
        table.add_row(*(str(row.get(key, "")) for key, _ in columns))
    console.print(table)


# Which prompt file each task reads; the merge check builds its prompt in code
_PROMPTS = {
    "relevance": "relevance",
    "edit": "profile_edit",
    "rewrite": "profile_update",
    "facts": "fact_extract",
}


def _prompt_file(domain_cfg: DomainConfig, key: str, model: str) -> str:
    if key not in _PROMPTS:
        return "built-in"
    return os.path.basename(domain_cfg.prompt_path(_PROMPTS[key], model))


def _print_grid(rows: List[Dict[str, Any]]) -> None:
    """Latest headline score for each model and prompt, one table per set."""
    latest: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:  # the scoreboard is in run order, so later rows win
        key = (row.get("dataset", ""), row.get("approach", ""), row.get("split", ""))
        latest[(*key, f"{row.get('model')}|{row.get('prompt', '?')}")] = row
    for table_key in sorted({k[:3] for k in latest}):
        cells = {k[3]: r for k, r in latest.items() if k[:3] == table_key}
        prompts = sorted({c.split("|", 1)[1] for c in cells})
        table = Table(title=" / ".join(table_key))
        table.add_column("Model")
        for prompt in prompts:
            table.add_column(prompt)
        for model in sorted({c.split("|", 1)[0] for c in cells}):
            row = []
            for prompt in prompts:
                r = cells.get(f"{model}|{prompt}")
                score = r and r.get("pass_rate", r.get("balanced_accuracy"))
                row.append("" if not r else f"{score} (n={r.get('n')})")
            table.add_row(model, *row)
        console.print(table)


def _rescore(path: str, cases: List[Dict[str, Any]]) -> None:
    """Print a saved run's summary as saved and as scored today."""
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ids = {c["case_id"] for c in cases}
    before = [r for r in records if r.get("item_id") in ids]
    after = rescore_run(before, cases)
    rows = [
        {"approach": "as saved", **summarize_updates(before)},
        {"approach": "today", **summarize_updates(after)},
    ]
    _print_rows(rows, os.path.basename(path), _UPDATE_COLUMNS)
    for old, new in zip(before, after):
        was = (old.get("scores") or {}).get("passed")
        now = (new.get("scores") or {}).get("passed")
        if was != now:
            s = new.get("scores") or {}
            why = (
                [f"missed {', '.join(s['core_missing'])}"]
                if s.get("core_missing")
                else []
            )
            why += [
                f"{h['kind']}: {h['sentence'][:120]}"
                for h in s.get("must_not_hits", [])
            ]
            if (
                s.get("claims_ok") is not None
                and s.get("dispute_claims")
                and not s.get("disputes_ok")
            ):
                why.append(f"{s['dispute_claims']} 'sources disagree' sentence(s)")
            console.print(f"{new['item_id']}: {was} -> {now}  {'; '.join(why)}")


def _run_profile_update(args, parser, eval_dir: str) -> None:
    try:
        cases = load_cases(
            eval_dir, args.split, args.include_unreviewed, args.claim_labels
        )
    except FileNotFoundError as e:
        parser.error(f"Cases missing (run `just eval-cases`): {e.filename}")
    if args.rescore:
        _rescore(args.rescore, cases)
        return
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
        "prompt": _prompt_file(DomainConfig(args.domain), args.approach, args.model),
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
    parser.add_argument(
        "task",
        choices=(
            *TASKS,
            "profile-update",
            "profile-create",
            "placement",
            "subplacement",
            "show",
        ),  # fmt: skip
    )
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
    parser.add_argument(
        "--existing-profile",
        choices=("saved", "stub"),
        default="saved",
        help="merge: compare against the saved profile, or the stub profile a "
        "once-mentioned entity now has (src/engine/profile_stubs.py)",
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
        choices=("edit", "rewrite", "facts"),
        default="edit",
        help="profile-update: apply a list of edits (the pipeline's default), "
        "the old whole-profile rewrite with reflection, or the fact-list "
        "prototype (src/engine/fact_list.py)",
    )
    parser.add_argument(
        "--prompt-variant",
        default=None,
        help="score <prompt>.<variant>.md instead of the one the pipeline picks "
        "for --model ('default' = the plain <prompt>.md)",
    )
    parser.add_argument(
        "--grid", action="store_true", help="show: latest score per model and prompt"
    )
    create = parser.add_argument_group("profile-create")
    create.add_argument(
        "--cases",
        choices=("entities", "replay"),
        default="entities",
        help="pairs from an entity database, or each replay entity's first article",
    )
    create.add_argument("--entities-dir", default=None, help="default: output dir")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument(
        "--create-approaches",
        default="reflect,check",
        help="comma-separated create_profile() approaches to compare",
    )
    create.add_argument(
        "--grounding",
        action="store_true",
        help="also check each cited claim against the article (cloud model)",
    )
    place = parser.add_argument_group("placement")
    place.add_argument(
        "--labels", default=None, help="default: {eval_dir}/placement_labels.jsonl"
    )
    place.add_argument("--batch", type=int, default=8, help="sentences per call")
    place.add_argument(
        "--placement-approaches",
        default="reorganise,classify",
        help="comma-separated: reorganise (the reorganise prompt), classify "
        "(src/engine/section_place.py)",
    )
    parser.add_argument(
        "--claim-labels",
        choices=("reviewed", "draft", "off"),
        default="reviewed",
        help="profile-update: which claim labels (claim_labels.jsonl) to check "
        "new_fact cases against; 'draft' also uses unreviewed ones",
    )
    parser.add_argument(
        "--rescore",
        default=None,
        metavar="RUN_FILE",
        help="profile-update: re-score a saved run file with today's checks "
        "(no model calls) and print before/after",
    )
    args = parser.parse_args()
    if args.prompt_variant:
        os.environ["HINBOX_PROMPT_VARIANT"] = args.prompt_variant

    domain_cfg = DomainConfig(args.domain)
    eval_dir = args.eval_dir or os.path.join(domain_cfg.get_output_dir(), "eval")

    if args.task == "show":
        rows = load_scoreboard(eval_dir)
        if args.grid:
            _print_grid(rows)
            return
        updates = [r for r in rows if r.get("task") == "profile_update"]
        others = [r for r in rows if r.get("task") != "profile_update"]
        _print_rows(others, f"Scoreboard ({eval_dir})")
        if updates:
            _print_rows(updates, "Profile updates", _UPDATE_COLUMNS)
        return

    if args.task == "profile-update" and args.rescore:  # no model calls to trace
        _run_profile_update(args, parser, eval_dir)
        return

    # Eval runs are traced too: their model calls carry gold labels and scores,
    # which makes them the best training examples we have
    try:
        configure_tracing(
            domain_cfg.get_output_dir(),
            local=args.model.startswith("ollama/"),
            run_name=f"eval-{args.task}",
            domain=args.domain,
        )
    except ValueError as e:
        parser.error(str(e))
    if args.task == "profile-update":
        _run_profile_update(args, parser, eval_dir)
        return
    if args.task == "profile-create":
        run_profile_create(args, parser, domain_cfg, eval_dir)
        return
    if args.task == "placement":
        run_placement(args, parser, domain_cfg, eval_dir)
        return
    if args.task == "subplacement":
        run_subplacement(args, parser, eval_dir)
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
    approach = "pipeline"
    if args.task == "merge" and args.existing_profile == "stub":
        items, swapped = stub_existing_profiles(items, domain_cfg.get_data_path())
        console.print(f"Existing profiles swapped for stubs: {swapped}/{len(items)}")
        approach = "pipeline-stub"

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
        approach=approach,
        dataset=dataset,
    )
    summary["prompt"] = _prompt_file(domain_cfg, args.task, args.model)
    path = save_run(eval_dir, summary, records)
    _print_rows([summary], "This run")
    console.print(f"Predictions: {path}")


if __name__ == "__main__":
    main()
