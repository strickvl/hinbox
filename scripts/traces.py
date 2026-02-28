#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
"""Inspect pipeline trace runs and events."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config_loader import DomainConfig


def _resolve_runs_dir(domain: str) -> Path:
    """Resolve the trace runs directory for a domain."""
    config = DomainConfig(domain)
    base_dir = config.get_output_dir()
    trace_cfg = config.get_tracing_config()
    traces_subdir = str(trace_cfg.get("subdir") or "traces")
    runs_subdir = str(trace_cfg.get("runs_subdir") or "runs")
    return Path(base_dir) / traces_subdir / runs_subdir


def _list_run_ids(runs_dir: Path) -> List[str]:
    """List available run ids (newest first)."""
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    run_ids = [
        path.stem
        for path in runs_dir.iterdir()
        if path.is_file() and path.suffix == ".jsonl"
    ]
    run_ids.sort(reverse=True)
    return run_ids


def _resolve_run_file(runs_dir: Path, run_id: Optional[str]) -> Optional[Path]:
    """Resolve the selected run file, defaulting to latest."""
    run_ids = _list_run_ids(runs_dir)
    if not run_ids:
        return None

    if run_id:
        normalized = run_id[:-6] if run_id.endswith(".jsonl") else run_id
        path = runs_dir / f"{normalized}.jsonl"
        return path if path.exists() else None

    return runs_dir / f"{run_ids[0]}.jsonl"


def _load_events(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL events from a run file, skipping malformed lines."""
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skipping malformed JSON at line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    return events


def _filter_events(
    events: Iterable[Dict[str, Any]],
    stage: Optional[str],
) -> List[Dict[str, Any]]:
    """Filter events by stage if requested."""
    if not stage:
        return list(events)
    return [event for event in events if event.get("stage") == stage]


def _apply_tail(events: List[Dict[str, Any]], tail: int) -> List[Dict[str, Any]]:
    """Return only the last N events; non-positive values keep all."""
    if tail <= 0:
        return events
    return events[-tail:]


def _shorten(text: Optional[str], limit: int = 120) -> str:
    """Shorten text for one-line display."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_event_line(event: Dict[str, Any]) -> str:
    """Format one event as a compact human-readable line."""
    timestamp = str(event.get("timestamp") or "?")
    stage = str(event.get("stage") or "?")
    decision = str(event.get("decision") or "-")

    confidence = event.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{float(confidence):.2f}"
    else:
        confidence_text = "-"

    entity_type = str(event.get("entity_type") or "-")
    entity_name = str(event.get("entity_name") or "-")
    candidate_name = str(event.get("candidate_name") or "-")
    article_id = str(event.get("article_id") or "-")
    reason = _shorten(event.get("reason"))

    parts = [
        timestamp,
        stage,
        f"decision={decision}",
        f"conf={confidence_text}",
        f"entity_type={entity_type}",
        f"entity={entity_name}",
        f"candidate={candidate_name}",
        f"article={article_id}",
    ]
    if reason:
        parts.append(f"reason={reason}")
    return " | ".join(parts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect hinbox trace runs and events")
    parser.add_argument(
        "--domain",
        default="guantanamo",
        help="Domain name (default: guantanamo)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available run ids and exit",
    )
    parser.add_argument(
        "--run-id",
        help="Run id to inspect (defaults to latest run)",
    )
    parser.add_argument(
        "--stage",
        help="Filter events by stage (e.g. merge.decision)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=25,
        help="Show only the last N events after filtering (<=0 shows all)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON lines instead of human-readable summary",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    runs_dir = _resolve_runs_dir(args.domain)

    if args.list_runs:
        run_ids = _list_run_ids(runs_dir)
        if not run_ids:
            print(f"No trace runs found in: {runs_dir}")
            return 0

        print(f"Trace runs in {runs_dir} (newest first):")
        for run_id in run_ids:
            print(run_id)
        return 0

    run_file = _resolve_run_file(runs_dir, args.run_id)
    if run_file is None:
        print(f"No trace run found in: {runs_dir}", file=sys.stderr)
        return 1

    events = _load_events(run_file)
    filtered = _filter_events(events, args.stage)
    shown = _apply_tail(filtered, args.tail)

    run_id = run_file.stem
    stage_text = args.stage or "(all)"
    print(
        f"run_id={run_id} | events={len(shown)}/{len(filtered)} "
        f"after stage={stage_text} filter"
    )

    if not shown:
        return 0

    if args.raw:
        for event in shown:
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    else:
        for event in shown:
            print(_format_event_line(event))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
