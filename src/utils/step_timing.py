"""Per-step wall-clock timings for a pipeline run.

``timed_step("extract.people")`` times a block and remembers the step name
for the current thread (a ``ContextVar``), so steps nest: a profile update
inside the people merge is recorded as ``merge.people/profile.update``.
Every LLM call goes through ``cloud_llm_slot()`` / ``local_llm_slot()`` in
``src/utils/llm.py``, which call ``record_llm_call()``; each call is counted
under whatever step is active, with the time spent waiting for a
concurrency slot kept apart from the time the model took.

``write_report()`` saves the totals as JSON at the end of a run. Timings are
always collected; it is a dictionary update per step, so cheap.
"""

import functools
import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

_current_step: ContextVar[Optional[str]] = ContextVar("hinbox_step", default=None)
_lock = threading.Lock()
_steps: Dict[str, Dict[str, float]] = {}
_llm: Dict[str, Dict[str, float]] = {}
F = TypeVar("F", bound=Callable[..., Any])


def _add(table: Dict[str, Dict[str, float]], key: str, **values: float) -> None:
    with _lock:
        row = table.setdefault(key, {})
        for name, value in values.items():
            row[name] = row.get(name, 0.0) + value


def current_step() -> str:
    return _current_step.get() or "(no step)"


@contextmanager
def timed_step(name: str) -> Iterator[None]:
    """Time a block under ``name``, nested under the thread's active step."""
    parent = _current_step.get()
    path = f"{parent}/{name}" if parent else name
    token = _current_step.set(path)
    start = time.perf_counter()
    try:
        yield
    finally:
        _add(_steps, path, count=1, seconds=time.perf_counter() - start)
        _current_step.reset(token)


def record_llm_call(*, wait_seconds: float, call_seconds: float, ok: bool) -> None:
    """Count one LLM request under the active step."""
    _add(
        _llm,
        current_step(),
        calls=1,
        failed=0 if ok else 1,
        wait_seconds=wait_seconds,
        call_seconds=call_seconds,
    )


def reset_timings() -> None:
    with _lock:
        _steps.clear()
        _llm.clear()


def timing_report(articles: int, wall_seconds: float) -> Dict[str, Any]:
    """Totals per step, plus per-article averages when ``articles`` > 0."""
    per = max(articles, 1)
    with _lock:
        steps = {
            path: {
                "count": int(row["count"]),
                "seconds": round(row["seconds"], 2),
                "mean_seconds": round(row["seconds"] / row["count"], 2),
                "seconds_per_article": round(row["seconds"] / per, 2),
            }
            for path, row in sorted(_steps.items())
        }
        llm = {
            path: {
                "calls": int(row["calls"]),
                "failed": int(row["failed"]),
                "call_seconds": round(row["call_seconds"], 2),
                "mean_call_seconds": round(row["call_seconds"] / row["calls"], 2),
                "wait_seconds": round(row["wait_seconds"], 2),
            }
            for path, row in sorted(_llm.items())
        }
    return {
        "articles": articles,
        "wall_seconds": round(wall_seconds, 2),
        "wall_seconds_per_article": round(wall_seconds / per, 2),
        "steps": steps,
        "llm_calls": llm,
    }


def write_report(path: str, articles: int, wall_seconds: float) -> Dict[str, Any]:
    report = timing_report(articles, wall_seconds)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def timed(name: str) -> Callable[[F], F]:
    """Decorator form of ``timed_step``."""

    def wrap(fn: F) -> F:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            with timed_step(name):
                return fn(*args, **kwargs)

        return inner  # type: ignore[return-value]

    return wrap


def save_timing_report(base_dir: str, articles: int, wall_seconds: float) -> str:
    """Write ``{base_dir}/timings/run-<timestamp>.json`` and print a summary table."""
    from rich.table import Table

    from src.logging_config import console, log

    out_dir = os.path.join(base_dir, "timings")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run-{datetime.now():%Y%m%d-%H%M%S}.json")
    report = write_report(path, articles, wall_seconds)

    table = Table(title=f"Step timings ({articles} articles, {wall_seconds:.0f}s)")
    for col in ("step", "count", "total s", "mean s", "s/article", "LLM calls"):
        table.add_column(col, justify="left" if col == "step" else "right")
    llm = report["llm_calls"]
    for step, row in report["steps"].items():
        calls = sum(v["calls"] for k, v in llm.items() if k == step)
        table.add_row(
            step,
            str(row["count"]),
            f"{row['seconds']:.1f}",
            f"{row['mean_seconds']:.2f}",
            f"{row['seconds_per_article']:.1f}",
            str(calls or ""),
        )
    console.print(table)
    log(f"Step timings written to {path}", level="info")
    return path
