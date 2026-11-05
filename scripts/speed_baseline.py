"""Reproducible speed + quality baseline for a pipeline run on a fixed article set.

Usage:
  uv run python -m scripts.speed_baseline run --out bench --label main-w32 --limit 20
  uv run python -m scripts.speed_baseline run --out bench --label w8 \
      --set max_buffered_articles=8
  uv run python -m scripts.speed_baseline compare --out bench

``run`` picks ``--limit`` Guantánamo articles once (the same deterministic pick as
``scripts/compare_extraction.py``: sorted by a hash of the article id) and saves
them to ``<out>/articles.parquet``; every later run in that folder reuses them.
It then runs the real pipeline (``src.process_and_extract``, ``--local
--relevance-check`` by default) twice, each in its own output folder
(``HINBOX_OUTPUT_DIR``) with empty entity tables:

- ``fresh``: nothing cached, so every relevance check, extraction, merge,
  profile and grounding call goes to the model.
- ``cached``: the extraction cache and the "not relevant" verdicts from the
  fresh pass are copied in first, so extraction is free and the time left is
  mostly merging, profiles and grounding.

``--set key=value`` overrides a concurrency setting for the run through
``HINBOX_<KEY>`` (e.g. ``extract_workers``, ``ollama_in_flight``,
``max_buffered_articles``). Each pass writes ``run.log``; the run writes
``<out>/<label>/summary.json`` and ``summary.md`` with the commit, models,
context size, concurrency, what the model server reports about itself, the
timings, and quality counts that need no model (``scripts/audit_entities.py``:
repeated sentences, "sources disagree" sentences, cited sentences lost, QC
failures; plus failed calls, incomplete articles and grounding scores).

Outputs hold article-derived text: keep them off GitHub (the repo is public).
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

TYPES = ["people", "organizations", "locations", "events"]
SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def git_info() -> Dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=False
        ).stdout.strip()

    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": dirty[:50],
    }


def settings_env() -> Dict[str, str]:
    """HINBOX_*/OLLAMA_*/EMBEDDING_* settings, never anything that looks like a key."""
    return {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("HINBOX_", "OLLAMA_", "EMBEDDING_", "OTEL_"))
        and not any(w in k for w in SECRET_WORDS)
    }


def _get_json(url: str) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _trim(value: Any) -> Any:
    """Drop long strings (chat templates) so the summary stays readable."""
    if isinstance(value, dict):
        return {
            k: _trim(v)
            for k, v in value.items()
            if not (isinstance(v, str) and len(v) > 300)
        }
    if isinstance(value, list):
        return [_trim(v) for v in value[:8]]
    return value


def server_info(api_url: str) -> Dict[str, Any]:
    """What the local model server says about itself (llama.cpp or Ollama)."""
    base = api_url.rstrip("/").removesuffix("/v1")
    found = {
        "api_url": api_url,
        "props": _get_json(f"{base}/props"),  # llama.cpp: n_ctx, total_slots, build
        "models": _get_json(f"{base}/v1/models"),
        "ollama_version": _get_json(f"{base}/api/version"),
        "ollama_ps": _get_json(f"{base}/api/ps"),
    }
    return {k: _trim(v) for k, v in found.items() if v is not None}


def pick_rows(domain: str, limit: int, min_chars: int, max_chars: int) -> List[Dict]:
    from scripts.compare_extraction import pick_articles
    from src.config_loader import DomainConfig

    ids = [a["id"] for a in pick_articles(domain, limit, min_chars, max_chars)]
    rows = {
        str(r.get("id")): r
        for r in pq.read_table(DomainConfig(domain).get_data_path()).to_pylist()
    }
    return [rows[i] for i in ids if i in rows]


def article_set(args: argparse.Namespace) -> Tuple[str, List[str]]:
    path = os.path.join(args.out, "articles.parquet")
    if not os.path.exists(path):
        rows = pick_rows(args.domain, args.limit, args.min_chars, args.max_chars)
        os.makedirs(args.out, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    ids = [str(i) for i in pq.read_table(path, columns=["id"]).column("id").to_pylist()]
    if len(ids) != args.limit:
        raise SystemExit(
            f"{path} holds {len(ids)} articles, not --limit {args.limit}; "
            "use a new --out for a different set"
        )
    return path, ids


def seed_cached_pass(fresh_dir: str, cached_dir: str) -> None:
    """Copy the fresh pass's extraction cache and 'not relevant' verdicts only."""
    if os.path.isdir(os.path.join(fresh_dir, "cache")):
        shutil.copytree(
            os.path.join(fresh_dir, "cache"), os.path.join(cached_dir, "cache")
        )
    with open(os.path.join(fresh_dir, "processing_status.json")) as f:
        status = json.load(f)
    keep = {k: v for k, v in status.items() if v.get("reason") == "not_relevant"}
    with open(os.path.join(cached_dir, "processing_status.json"), "w") as f:
        json.dump(keep, f, indent=2)


def run_pass(
    out_dir: str, articles: str, limit: int, env: Dict[str, str], extra: List[str]
) -> float:
    cmd = [sys.executable, "-m", "src.process_and_extract", "--limit", str(limit)]
    cmd += ["--articles-path", articles, *extra]
    print(f"$ {' '.join(cmd)}  (output -> {out_dir})", flush=True)
    t0 = time.perf_counter()
    with open(os.path.join(out_dir, "run.log"), "w") as log:
        code = subprocess.run(
            cmd,
            env={**os.environ, **env, "HINBOX_OUTPUT_DIR": out_dir},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    seconds = time.perf_counter() - t0
    if code != 0:
        raise SystemExit(f"pipeline exited {code}; see {out_dir}/run.log")
    return seconds


def latest_timing(out_dir: str) -> Dict[str, Any]:
    tdir = os.path.join(out_dir, "timings")
    runs = sorted(f for f in os.listdir(tdir) if f.startswith("run-"))
    with open(os.path.join(tdir, runs[-1])) as f:
        return json.load(f)


def llm_totals(timing: Dict[str, Any]) -> Dict[str, float]:
    rows = timing.get("llm_calls", {}).values()
    keys = ("calls", "failed", "call_seconds", "wait_seconds")
    return {k: round(sum(r.get(k, 0) for r in rows), 1) for k in keys}


def grounding_summary(tables: Dict[str, List[Dict]]) -> Dict[str, Any]:
    reports = [
        r.get("profile_grounding") or {} for rows in tables.values() for r in rows
    ]
    scores = [
        g["grounding_score"] for g in reports if g.get("grounding_score") is not None
    ]
    return {
        "profiles_scored": len(scores),
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "verified": sum(g.get("verified", 0) for g in reports),
        "citations": sum(g.get("total_citations", 0) for g in reports),
    }


def pass_quality(out_dir: str, corpus: List[str]) -> Dict[str, Any]:
    from scripts.audit_entities import build_report, load_tables

    tables = load_tables(out_dir)
    _, audit = build_report(tables, set(corpus), 0)
    with open(os.path.join(out_dir, "processing_status.json")) as f:
        status = json.load(f)
    outcomes = Counter(
        "processed" if v.get("processed") else str(v.get("reason", "?")).split(":")[0]
        for v in status.values()
    )
    totals: Counter = Counter()
    for etype in TYPES:  # other keys of `audit` are run-wide counts
        counts = audit.get(etype) or {}
        totals.update({k: v for k, v in counts.items() if isinstance(v, int)})
    totals["name_variant_duplicates"] = sum((audit.get("duplicates") or {}).values())
    totals["cross_type"] = audit.get("cross_type", 0)
    return {
        "articles": dict(outcomes),
        "entities": {t: len(tables.get(t, [])) for t in TYPES},
        "audit_totals": dict(totals),
        "audit": audit,
        "grounding": grounding_summary(tables),
    }


def summarise_pass(name: str, out_dir: str, seconds: float, ids: List[str]) -> Dict:
    timing = latest_timing(out_dir)
    top = {k: v for k, v in timing["steps"].items() if "/" not in k}
    return {
        "pass": name,
        "wall_seconds": round(seconds, 1),
        "pipeline_wall_seconds": timing["wall_seconds"],
        "articles_merged": timing["articles"],
        "seconds_per_merged_article": timing["wall_seconds_per_article"],
        "llm": llm_totals(timing),
        "top_steps": {k: v["seconds"] for k, v in top.items()},
        "timing": timing,
        "quality": pass_quality(out_dir, ids),
    }


def model_settings(local: bool) -> Dict[str, Any]:
    from src.config_loader import DomainConfig
    from src.utils.task_models import TASKS, task_model

    model_type = "ollama" if local else "gemini"
    return {
        "tasks": {t: task_model(t, model_type) for t in TASKS},
        "concurrency": DomainConfig("guantanamo").get_concurrency_config(),
    }


def run(args: argparse.Namespace) -> None:
    sets = dict(s.split("=", 1) for s in args.set)
    env = {f"HINBOX_{k.upper()}": v for k, v in sets.items()}
    os.environ.update(env)  # so the recorded concurrency includes the overrides
    articles, ids = article_set(args)
    label_dir = os.path.join(args.out, args.label)
    if os.path.exists(label_dir):
        raise SystemExit(f"{label_dir} exists; pick another --label")
    extra = (["--local"] if args.local else []) + ["--relevance-check"]
    summary: Dict[str, Any] = {
        "label": args.label,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_info(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "article_ids": ids,
        "overrides": sets,
        "pipeline_args": extra,
        "env": settings_env(),
        "models": model_settings(args.local),
        "server": server_info(os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1"))
        if args.local
        else {},
        "passes": [],
    }
    fresh, cached = os.path.join(label_dir, "fresh"), os.path.join(label_dir, "cached")
    for name, out_dir in (("fresh", fresh), ("cached", cached))[
        : 2 if args.cached else 1
    ]:
        os.makedirs(out_dir)
        if name == "cached":
            seed_cached_pass(fresh, cached)
        seconds = run_pass(out_dir, articles, len(ids), env, extra)
        summary["passes"].append(summarise_pass(name, out_dir, seconds, ids))
        print(f"{name}: {seconds:.0f}s", flush=True)
    with open(os.path.join(label_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    table = render([summary])
    with open(os.path.join(label_dir, "summary.md"), "w") as f:
        f.write(table)
    print(table)


COLUMNS = [
    ("wall s", lambda p: p["wall_seconds"]),
    ("s/article", lambda p: p["seconds_per_merged_article"]),
    ("merged", lambda p: p["articles_merged"]),
    ("LLM calls", lambda p: p["llm"]["calls"]),
    ("failed", lambda p: p["llm"]["failed"]),
    ("slot wait s", lambda p: p["llm"]["wait_seconds"]),
    ("entities", lambda p: sum(p["quality"]["entities"].values())),
    ("incomplete", lambda p: p["quality"]["articles"].get("incomplete", 0)),
    ("QC failed", lambda p: p["quality"]["audit_totals"].get("qc_failed", 0)),
    ("repeats", lambda p: p["quality"]["audit_totals"].get("repeated_sentences", 0)),
    ("disputes", lambda p: p["quality"]["audit_totals"].get("dispute_sentences", 0)),
    ("lost", lambda p: p["quality"]["audit_totals"].get("lost_sentences", 0)),
    ("dups", lambda p: p["quality"]["audit_totals"].get("name_variant_duplicates", 0)),
    ("grounding", lambda p: p["quality"]["grounding"]["mean_score"]),
]


def render(summaries: List[Dict[str, Any]]) -> str:
    head = ["run", "pass", "commit", "overrides"] + [c for c, _ in COLUMNS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        over = ", ".join(f"{k}={v}" for k, v in s["overrides"].items()) or "-"
        for p in s["passes"]:
            cells = [s["label"], p["pass"], s["git"]["commit"][:7], over]
            cells += [str(fn(p)) for _, fn in COLUMNS]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> None:
    found = []
    for label in sorted(os.listdir(args.out)):
        path = os.path.join(args.out, label, "summary.json")
        if os.path.exists(path):
            with open(path) as f:
                found.append(json.load(f))
    found.sort(key=lambda s: s["started"])
    print(render(found))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--domain", default="guantanamo")
    r.add_argument("--min-chars", type=int, default=1500)
    r.add_argument("--max-chars", type=int, default=12000)
    r.add_argument("--set", action="append", default=[], help="key=value")
    r.add_argument("--cloud", dest="local", action="store_false")
    r.add_argument("--no-cached", dest="cached", action="store_false")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
