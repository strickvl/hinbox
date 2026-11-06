"""Time the grounding pass on a finished run's entities and compare verdicts.

``run`` loads a run's entity tables (the folder with ``people.parquet`` etc.)
and the articles they came from, keeps each profile's stored grounding report
as the "before" verdicts, drops the reports, and runs the grounding pass
again from scratch with the current code. It writes one JSON file with the
time taken, the number of model calls, and every claim's verdict before and
after.

``compare`` reads that file and counts how verdicts moved, claim by claim.
The number to watch is "newly accepted": claims the stored report called
not_supported or unclear that are now supported or partial.

    uv run python -m scripts.grounding_bench run --entities <dir> \\
        --articles <articles.parquet> --out <file.json> --local
    uv run python -m scripts.grounding_bench compare <file.json>

Outputs hold claim text from profiles: keep them off GitHub.
"""

import argparse
import json
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import pyarrow.parquet as pq

ACCEPTED = {"supported", "partial"}


def _verdict_rows(entities: Dict[str, Dict], field: str = "profile_grounding"):
    rows = []
    for etype, table in entities.items():
        for key, data in table.items():
            report = data.get(field) or {}
            for v in report.get("verifications") or []:
                rows.append(
                    {
                        "entity": f"{etype}/{key}",
                        "article_id": v.get("article_id"),
                        "claim": v.get("claim"),
                        "level": v.get("support_level"),
                        "reasoning": v.get("reasoning"),
                    }
                )
    return rows


def run(args: argparse.Namespace) -> None:
    from src.engine.article_processor import ArticleProcessor
    from src.engine.grounding_pass import run_profile_grounding_postprocess
    from src.process_and_extract import load_existing_entities
    from src.utils.llm import configure_llm_concurrency

    entities = load_existing_entities(args.entities)
    before = _verdict_rows(entities)
    for table in entities.values():
        for data in table.values():
            data.pop("profile_grounding", None)
    rows = pq.read_table(args.articles).to_pylist()
    configure_llm_concurrency(
        cloud_in_flight=args.in_flight, local_in_flight=args.in_flight
    )
    start = time.perf_counter()
    counts = run_profile_grounding_postprocess(
        entities=entities,
        rows=rows,
        processor=ArticleProcessor.__new__(ArticleProcessor),  # stateless here
        model_type="ollama" if args.local else "gemini",
        workers=args.workers,
    )
    seconds = time.perf_counter() - start
    flags = Counter(
        f
        for table in entities.values()
        for data in table.values()
        for f in (data.get("profile_grounding") or {}).get("flags") or []
    )
    out = {
        "seconds": round(seconds, 1),
        "counts": counts,
        "flags": dict(flags),
        "before": before,
        "after": _verdict_rows(entities),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("seconds", "counts", "flags")}, indent=1))


def _key(r: Dict[str, Any]) -> Tuple[str, str, str]:
    return (r["entity"], r["article_id"], r["claim"])


def compare(args: argparse.Namespace) -> None:
    with open(args.file) as f:
        data = json.load(f)
    before: Dict[Tuple[str, str, str], List[str]] = {}
    for r in data["before"]:
        before.setdefault(_key(r), []).append(r["level"])
    moves: Counter = Counter()
    newly_accepted, newly_rejected = [], []
    unmatched = 0
    for r in data["after"]:
        old = before.get(_key(r))
        if not old:
            unmatched += 1
            continue
        was = old.pop(0)
        moves[(was, r["level"])] += 1
        if was not in ACCEPTED and r["level"] in ACCEPTED and was != "missing_source":
            newly_accepted.append(r)
        if was in ACCEPTED and r["level"] not in ACCEPTED:
            newly_rejected.append(r)
    total = sum(moves.values())
    same = sum(n for (a, b), n in moves.items() if a == b)
    print(f"seconds: {data['seconds']}  counts: {data['counts']}")
    print(f"flags: {data['flags']}")
    print(f"claims compared: {total}, same verdict: {same}, unmatched: {unmatched}")
    for (a, b), n in sorted(moves.items(), key=lambda x: -x[1]):
        print(f"  {a:>15} -> {b:<15} {n}")
    for name, items in (
        ("newly accepted", newly_accepted),
        ("newly rejected", newly_rejected),
    ):
        print(f"\n{name}: {len(items)}")
        for r in items[: args.show]:
            print(f"  [{r['entity']} | {r['article_id'][:8]}] {r['claim'][:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="re-run grounding on a finished run's entities")
    r.add_argument("--entities", required=True, help="folder with people.parquet")
    r.add_argument("--articles", required=True, help="articles Parquet file")
    r.add_argument("--out", required=True, help="JSON file to write")
    r.add_argument("--local", action="store_true", help="use the local model")
    r.add_argument("--workers", type=int, default=8, help="calls submitted at once")
    r.add_argument("--in-flight", type=int, default=4, help="model calls in flight")
    r.set_defaults(fn=run)
    c = sub.add_parser("compare", help="count how verdicts moved")
    c.add_argument("file")
    c.add_argument("--show", type=int, default=20, help="examples to print")
    c.set_defaults(fn=compare)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
