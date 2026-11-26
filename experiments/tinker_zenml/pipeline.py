"""Fine-tune the citation checker on Tinker as a ZenML pipeline, then compare
it with the baseline judge in Kitaru.

Run from the Hinbox checkout (``just tinker-pipeline --data-dir <dir>
--articles <parquet>``), after the data steps of ``just citecheck`` (sample,
label, plant, teach, audit) have filled ``<dir>``. The pipeline's steps:

1. ``prepare_data``: ``just citecheck export`` (training and tune chats from
   the agreed teacher labels and the planted errors) and ``prompts`` (the
   test set's prompts), unless their files exist; records the counts.
2. ``train_on_tinker``: LoRA training on Tinker with the code of
   ``scripts/tinker_citecheck.py`` (same renderer, effort 0, loss on the
   tune file), saving a sampler checkpoint; the loss curve goes into the
   step's ZenML metadata.
3. ``sample_test_set``: one reply per test prompt from the checkpoint, at
   temperature 0, with tokens and seconds.
4. ``score_test_set``: ``just citecheck score --replies`` against the hand
   labels: wrong claims passed, good claims doubted, with 95% ranges.
   ``report_card`` then draws them as one HTML page (the ``training_report``
   artifact, ``HTMLString``): both rates with ranges, any ``--baselines``
   beside them, the loss curve, the rates per kind of claim, the settings.
5. ``compare_in_kitaru`` (``--kitaru``): an experiment on the
   ``hinbox-citation-check`` agent's labelled sessions with the checkpoint
   as the model override and the ``hinbox-citation-check-checks`` evaluator,
   so quality, time, tokens and cost sit beside the baseline judge's.

What goes into ZenML is paths, counts, checkpoint names and scores: the
article-derived files stay in ``<dir>`` (under ``data/``) and are never
written as artifacts, since a ZenML server may be shared or in the cloud.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional, Tuple

from zenml import log_metadata, pipeline, step
from zenml.types import HTMLString

FREE_MODEL = "thinkingmachines/Inkling-2-Mini:peft:65536:free"


def _path(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, name)


def _rows(path: str) -> List[Dict[str, Any]]:
    """The JSONL rows in ``path``; none when it doesn't exist yet."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def replies_path(data_dir: str, checkpoint: str) -> str:
    """Where one checkpoint's test replies go: its name plus a hash of its
    whole path, so two training runs saved under the same name never share
    (and silently resume) one file."""
    import hashlib

    tag = checkpoint.rstrip("/").rsplit("/", 1)[-1] or "base"
    digest = hashlib.sha256(checkpoint.encode()).hexdigest()[:10]
    return _path(data_dir, f"test_replies.{tag}.{digest}.jsonl")


def _count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ---- the parts that call Tinker or Kitaru (replaced in tests) -------------


def tinker_train(data_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train on ``train.jsonl`` (loss on ``tune.jsonl``); the checkpoint row
    and the training log."""
    from scripts import tinker_citecheck as tc

    out = _path(data_dir, "tinker")
    os.makedirs(out, exist_ok=True)
    args = SimpleNamespace(out=out, **cfg)
    log_path = os.path.join(out, "train_log.jsonl")
    earlier = _count(log_path)  # rows from earlier runs in this folder
    client = tc.train_loop(
        args,
        _rows(_path(data_dir, "train.jsonl")),
        _rows(_path(data_dir, "tune.jsonl")),
        steps=cfg.get("steps", 0),
    )
    sampler = client.save_weights_for_sampler(cfg["name"]).result()
    state = client.save_state(cfg["name"]).result()
    row = {
        "name": cfg["name"],
        "sampler": sampler.path,
        "state": state.path,
        "model": cfg["model"],
    }
    tc._append(os.path.join(out, "checkpoints.jsonl"), row)
    return {"checkpoint": row, "log": _rows(log_path)[earlier:]}


def tinker_sample(
    checkpoint: str, prompts_path: str, out_path: str, cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Replies to every prompt not already in ``out_path`` (resumable)."""
    import tinker

    from scripts import tinker_citecheck as tc

    renderer, _ = tc._renderer(cfg["model"])
    service = tinker.ServiceClient()
    if checkpoint == "base":
        sampler = tc._resolve(service.create_sampling_client(base_model=cfg["model"]))
    else:
        sampler = tc._resolve(service.create_sampling_client(model_path=checkpoint))
    done = {tuple(r["item_ids"]) for r in _rows(out_path)}
    todo = [p for p in _rows(prompts_path) if tuple(p["item_ids"]) not in done]
    args = SimpleNamespace(**cfg)
    tc._sample_rows(sampler, renderer, todo, args, out_path=out_path)
    return _rows(out_path)


def kitaru_compare(
    name: str, model: str, prices: Dict[str, List[float]], item_ids: List[str]
) -> Dict[str, Any]:
    """Kitaru experiment over the labelled sessions of ``item_ids`` only (the
    test set this run scored), with ``model`` as the override."""
    import asyncio

    from src.citecheck.kitaru_loop import experiment

    return asyncio.run(experiment(name, model, prices, item_ids=set(item_ids)))


# ---- steps ----------------------------------------------------------------


@step
def prepare_data(
    data_dir: str, articles: str, rebuild: bool = False
) -> Annotated[Dict[str, int], "data_counts"]:
    from src.citecheck.__main__ import main as citecheck

    if rebuild or not os.path.exists(_path(data_dir, "train.jsonl")):
        citecheck(["export", "--articles", articles, "--out", data_dir])
    prompts = _path(data_dir, "test_prompts.jsonl")
    if rebuild or not os.path.exists(prompts):
        citecheck(
            [
                "prompts",
                "--items",
                _path(data_dir, "test_items.jsonl"),
                "--articles",
                articles,
                "--file",
                prompts,
            ]
        )
    counts = {
        name: _count(_path(data_dir, f"{name}.jsonl"))
        for name in ("train", "tune", "test_items", "test_prompts", "test_labels")
    }
    if not counts["train"] or not counts["test_prompts"]:
        raise RuntimeError(f"Nothing to train or test on in {data_dir}: {counts}")
    log_metadata(metadata={"counts": counts})
    return counts


@step(enable_cache=False)
def train_on_tinker(
    data_dir: str, counts: Dict[str, int], cfg: Dict[str, Any]
) -> Tuple[Annotated[str, "checkpoint"], Annotated[List[Dict[str, float]], "loss_log"]]:
    started = time.monotonic()
    result = tinker_train(data_dir, cfg)
    log = result["log"]
    last = log[-1] if log else {}
    tune = [r["tune_nll"] for r in log if "tune_nll" in r]
    log_metadata(
        metadata={
            "checkpoint": result["checkpoint"],
            "steps": len(log),
            "train_examples": counts["train"],
            "final_train_nll": last.get("train_nll"),
            "final_tune_nll": tune[-1] if tune else None,
            "best_tune_nll": min(tune) if tune else None,
            "minutes": round((time.monotonic() - started) / 60, 1),
            "config": cfg,
        }
    )
    curve = [
        {k: r[k] for k in ("step", "train_nll", "tune_nll") if k in r} for r in log
    ]
    return result["checkpoint"]["sampler"], curve


@step(enable_cache=False)
def sample_test_set(
    data_dir: str, checkpoint: str, cfg: Dict[str, Any]
) -> Annotated[str, "replies_path"]:
    out = replies_path(data_dir, checkpoint)
    replies = tinker_sample(checkpoint, _path(data_dir, "test_prompts.jsonl"), out, cfg)
    n = max(1, len(replies))
    log_metadata(
        metadata={
            "replies": len(replies),
            "mean_output_tokens": sum(r.get("output_tokens", 0) for r in replies) / n,
            "mean_seconds": sum(r.get("seconds", 0) for r in replies) / n,
            "path": out,
        }
    )
    return out


@step
def score_test_set(
    data_dir: str, replies_path: str
) -> Annotated[Dict[str, Any], "test_report"]:
    from src.citecheck.items import latest_labels, read_jsonl
    from src.citecheck.score import describe, reply_predictions, score

    items = read_jsonl(_path(data_dir, "test_items.jsonl"))
    labels = latest_labels(_path(data_dir, "test_labels.jsonl"))
    report = score(items, labels, reply_predictions(read_jsonl(replies_path)))
    print(describe(report, os.path.basename(replies_path)))
    log_metadata(
        metadata={
            "labelled": report["labelled"],
            "wrong_passed": report["wrong_passed"]["share"],
            "wrong_passed_range": [
                report["wrong_passed"]["low"],
                report["wrong_passed"]["high"],
            ],
            "good_doubted": report["good_doubted"]["share"],
            "failed_replies": report["errors"],
        }
    )
    return report


@step
def report_card(
    report: Dict[str, Any],
    loss_log: List[Dict[str, float]],
    cfg: Dict[str, Any],
    replies_path: str,
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Annotated[HTMLString, "training_report"]:
    """The run on one page in ZenML's dashboard: both error rates with their
    ranges, the baselines beside them, the loss curve, the rates per kind of
    claim and the settings. Numbers only (see ``report_card.py``)."""
    from experiments.tinker_zenml.report_card import render

    replies = _rows(replies_path)
    n = max(1, len(replies))
    sampling = {
        "mean_output_tokens": sum(r.get("output_tokens", 0) for r in replies) / n,
        "mean_seconds": sum(r.get("seconds", 0) for r in replies) / n,
    }
    return HTMLString(render(report, loss_log, cfg, baselines, sampling))


@step(enable_cache=False)
def compare_in_kitaru(
    data_dir: str,
    checkpoint: str,
    name: str,
    prices: Dict[str, List[float]],
    report: Dict[str, Any],
) -> Annotated[Dict[str, Any], "kitaru_run"]:
    items = [r["item_id"] for r in _rows(_path(data_dir, "test_items.jsonl"))]
    out = kitaru_compare(name, checkpoint, prices, items)
    table = {r["name"]: r for r in out.get("table") or []}
    log_metadata(
        metadata={
            "experiment_id": out.get("experiment_id"),
            "run_id": out.get("run_id"),
            "status": out.get("status"),
            **{
                f"{k}_{side}": table[k][f"{side}_mean"]
                for k in table
                for side in ("baseline", "result")
            },
        }
    )
    return out


@pipeline(enable_cache=False)
def citecheck_finetune(
    data_dir: str,
    articles: str,
    cfg: Dict[str, Any],
    kitaru: bool = False,
    experiment_name: str = "citation checker: tuned Inkling vs baseline",
    prices: Optional[Dict[str, List[float]]] = None,
    rebuild: bool = False,
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    counts = prepare_data(data_dir, articles, rebuild)
    checkpoint, loss_log = train_on_tinker(data_dir, counts, cfg)
    replies = sample_test_set(data_dir, checkpoint, cfg)
    report = score_test_set(data_dir, replies)
    report_card(report, loss_log, cfg, replies, baselines)
    if kitaru:
        compare_in_kitaru(data_dir, checkpoint, experiment_name, prices or {}, report)


def training_config(a: argparse.Namespace) -> Dict[str, Any]:
    """What ``scripts/tinker_citecheck.py``'s training and sampling read."""
    return {
        "model": a.model,
        "name": a.name,
        "effort": a.effort,
        "rank": a.rank,
        "lr": a.lr,
        "epochs": a.epochs,
        "steps": a.steps,
        "batch_size": a.batch_size,
        "eval_every": a.eval_every,
        "max_length": a.max_length,
        "max_tokens": a.max_tokens,
    }


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--articles", required=True)
    p.add_argument("--model", default=FREE_MODEL)
    p.add_argument("--name", default="citecheck")
    p.add_argument("--effort", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0, help="0 = the cookbook's value")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=0, help="0 = whole epochs")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument(
        "--kitaru", action="store_true", help="also run the Kitaru experiment"
    )
    p.add_argument(
        "--experiment-name", default="citation checker: tuned Inkling vs baseline"
    )
    p.add_argument("--prices", help="JSON file: model -> [$/M in, $/M out]")
    p.add_argument("--rebuild", action="store_true", help="export and prompts again")
    p.add_argument(
        "--baselines",
        help="JSON file: checker name -> {wrong_passed, good_doubted}, each a "
        "count on the same test set or [count, of]; shown in the report card",
    )
    a = p.parse_args(argv)
    prices = json.load(open(a.prices, encoding="utf-8")) if a.prices else None
    baselines = json.load(open(a.baselines, encoding="utf-8")) if a.baselines else None
    citecheck_finetune(
        data_dir=a.data_dir,
        articles=a.articles,
        cfg=training_config(a),
        kitaru=a.kitaru,
        experiment_name=a.experiment_name,
        prices=prices,
        rebuild=a.rebuild,
        baselines=baselines,
    )


if __name__ == "__main__":
    main()
