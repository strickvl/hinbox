"""Run one task with one model over a gold set and record the result.

A "predictor" is any function ``inputs -> bool``. The default predictors call
the pipeline's own relevance and match-check functions, so the scoreboard
measures exactly the prompts the pipeline uses. Later experiments (DSPy, a
fine-tuned model) only need to supply a different predictor.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.evals.datasets import EvalItem
from src.evals.metrics import binary_metrics

Predictor = Callable[[Dict[str, Any]], bool]

TASKS = ("relevance", "merge")


def _is_local(model: str) -> bool:
    return model.startswith("ollama/")


def make_pipeline_predictor(task: str, model: str, domain: str) -> Predictor:
    """Build a predictor that calls the pipeline's own prompt for ``task``."""
    if task == "relevance":
        from src.engine.relevance import gemini_check_relevance, ollama_check_relevance

        check = ollama_check_relevance if _is_local(model) else gemini_check_relevance

        def predict_relevance(inputs: Dict[str, Any]) -> bool:
            return bool(
                check(text=inputs["text"], model=model, domain=domain).is_relevant
            )

        return predict_relevance

    if task == "merge":
        from src.engine.match_checker import (
            cloud_model_check_match,
            local_model_check_match,
        )

        check = local_model_check_match if _is_local(model) else cloud_model_check_match

        def predict_merge(inputs: Dict[str, Any]) -> bool:
            result = check(model=model, **inputs)
            # The match checker swallows API errors and returns a "no" verdict;
            # surface them as errors so they aren't scored as correct rejections.
            if result.reason.startswith("API error"):
                raise RuntimeError(result.reason)
            return bool(result.is_match)

        return predict_merge

    raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")


def _predict_one(predictor: Predictor, item: EvalItem) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        prediction: Optional[bool] = predictor(item.inputs)
        error = None
    except Exception as e:  # noqa: BLE001 — one failed call must not stop the run
        prediction, error = None, f"{type(e).__name__}: {e}"
    return {
        "item_id": item.item_id,
        "label": item.label,
        "prediction": prediction,
        "correct": prediction == item.label,
        "error": error,
        "seconds": round(time.perf_counter() - start, 3),
    }


def run_eval(
    items: List[EvalItem], predictor: Predictor, workers: int = 4
) -> List[Dict[str, Any]]:
    """Score every item, in parallel, keeping the input order."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda item: _predict_one(predictor, item), items))


def summarize(
    records: List[Dict[str, Any]],
    *,
    task: str,
    model: str,
    split: str,
    approach: str,
    dataset: str = "",
) -> Dict[str, Any]:
    seconds = sorted(r["seconds"] for r in records)
    metrics = binary_metrics([(r["label"], r["prediction"]) for r in records])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "dataset": dataset or task,
        "model": model,
        "approach": approach,
        "split": split,
        "median_seconds": seconds[len(seconds) // 2] if seconds else 0.0,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def save_run(
    eval_dir: str, summary: Dict[str, Any], records: List[Dict[str, Any]]
) -> str:
    """Write per-item predictions and append the summary to the scoreboard.

    Returns the predictions file path.
    """
    runs_dir = os.path.join(eval_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    stamp = summary["timestamp"].replace(":", "-")
    name = f"{summary['task']}-{_slug(summary['approach'])}-{_slug(summary['model'])}-{stamp}.jsonl"
    path = os.path.join(runs_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    with open(os.path.join(eval_dir, "scoreboard.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({**summary, "predictions_file": name}) + "\n")
    return path


def load_scoreboard(eval_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(eval_dir, "scoreboard.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
