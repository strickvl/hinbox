"""Run entity extraction with several models on the same articles and diff the results.

Usage:
  uv run python -m scripts.compare_extraction run --models openai/gpt-6-luna \
      ollama/local-qwen36-32k --limit 20 --repeats 2 --out <dir>
  uv run python -m scripts.compare_extraction compare --out <dir>

``run`` picks ``--limit`` Guantánamo articles (deterministic: sorted by a hash of
the article id, skipping very short/long ones), writes them to
``<dir>/articles.jsonl``, then runs the pipeline's own extraction step
(``ArticleProcessor.extract_all_entities``: the four extractions, QC + retry,
mention filter, ALL CAPS recasing, misfiled location/organization filters) once
per model per repeat. Each run goes to ``<dir>/runs/<model>__r<n>.jsonl``, one
line per article. Nothing is merged and no entity tables are touched.

``compare`` reads those files and writes ``<dir>/diff.json`` and ``<dir>/diff.md``:
per article and type, names found by every run of a model, names only one model
found, and names two models filed under different types. Names are matched with
``names_are_variants`` (so "Guantánamo Bay" = "Guantanamo Bay", "Trump" =
"Donald Trump").
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict, List, Set

import pyarrow.parquet as pq

TYPES = ["people", "organizations", "locations", "events"]


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _name(entity_type: str, e: Dict[str, Any]) -> str:
    return str(e.get("title") if entity_type == "events" else e.get("name") or "")


def pick_articles(
    domain: str, limit: int, min_chars: int, max_chars: int
) -> List[Dict]:
    from src.config_loader import DomainConfig
    from src.engine.article_processor import ArticleProcessor

    rows = pq.read_table(DomainConfig(domain).get_data_path()).to_pylist()
    prep = ArticleProcessor(domain=domain, model_type="gemini").prepare_article_info
    arts = [prep(r, i) for i, r in enumerate(rows)]
    arts = [
        a
        for a in arts
        if min_chars <= len(a["content"]) <= max_chars
        and a["content"].lower().count("guant") >= 2
    ]
    arts.sort(key=lambda a: hashlib.sha256(a["id"].encode()).hexdigest())
    return arts[:limit]


def _extract_one(proc: Any, model: str, rep: int, a: Dict) -> Dict:
    meta: Dict[str, Any] = {}
    t0 = time.perf_counter()
    ents = proc.extract_all_entities(a["content"], a["id"], meta, max_workers=4)
    return {
        "article_id": a["id"],
        "model": model,
        "repeat": rep,
        "seconds": round(time.perf_counter() - t0, 1),
        "entities": ents,
        "phase_outcomes": meta.get("phase_outcomes", {}),
    }


def _wrapped_cloud(
    text: str,
    system_prompt: str,
    response_model: Any,
    model: str,
    temperature: float = 0,
    *,
    entity_type: str = "",
) -> List[Any]:
    """Cloud extraction as one tool call returning ``{"items": [...]}``.

    The pipeline's cloud path uses parallel tool calls (one call per entity).
    This asks for the whole list in a single call instead, the way local
    models are asked, to test whether the call style changes what is found.
    """
    from src.utils.extraction import create_messages
    from src.utils.llm import _list_item_model, _list_wrapper_model, cloud_generation

    wrapper = _list_wrapper_model(_list_item_model(response_model))
    reply = cloud_generation(
        messages=create_messages(system_content=system_prompt, user_content=text),
        response_model=wrapper,
        model=model,
        temperature=temperature,
    )
    return list(reply.items)


def run(args: argparse.Namespace) -> None:
    from src.config_loader import DomainConfig
    from src.engine.article_processor import ArticleProcessor
    from src.utils.llm import configure_llm_concurrency

    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    arts = pick_articles(args.domain, args.limit, args.min_chars, args.max_chars)
    with open(os.path.join(args.out, "articles.jsonl"), "w") as f:
        for a in arts:
            f.write(json.dumps(a, default=str) + "\n")
    print(f"{len(arts)} articles -> {args.out}/articles.jsonl")

    if args.cloud_wrapped:
        import src.engine.extractors as extractors

        extractors.extract_entities_cloud = _wrapped_cloud
    cc = DomainConfig(args.domain).get_concurrency_config()
    configure_llm_concurrency(
        cloud_in_flight=cc["llm_in_flight"], local_in_flight=cc["ollama_in_flight"]
    )
    for model in args.models:
        os.environ["HINBOX_EXTRACTION_MODEL"] = model
        model_type = "ollama" if model.startswith("ollama/") else "gemini"
        proc = ArticleProcessor(domain=args.domain, model_type=model_type)
        label = model + (
            "+wrapped" if args.cloud_wrapped and model_type != "ollama" else ""
        )
        for rep in range(1, args.repeats + 1):
            path = os.path.join(args.out, "runs", f"{_slug(label)}__r{rep}.jsonl")
            if os.path.exists(path) and not args.overwrite:
                print(f"skip {path} (exists)")
                continue

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(partial(_extract_one, proc, label, rep), arts))
            with open(path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, default=str) + "\n")
            fails = sum(
                1
                for r in results
                for o in r["phase_outcomes"].get("extraction", {}).values()
                if not o.get("success", True)
            )
            print(
                f"{label} r{rep}: {time.perf_counter() - t0:.0f}s, "
                f"{fails} failed extractions -> {path}"
            )


def _load_runs(out: str) -> Dict[str, List[Dict[str, Dict]]]:
    """model -> list of runs; each run maps article_id -> result line."""
    runs: Dict[str, List[Dict[str, Dict]]] = defaultdict(list)
    d = os.path.join(out, "runs")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(d, fn)) as f:
            lines = [json.loads(x) for x in f if x.strip()]
        if lines:
            runs[lines[0]["model"]].append({x["article_id"]: x for x in lines})
    return runs


def _same(a: str, b: str, et: str) -> bool:
    from src.utils.name_variants import names_are_variants, normalize_for_match

    if et == "events":  # titles are free text: only exact-ish matches count
        return normalize_for_match(a) == normalize_for_match(b)
    return names_are_variants(a, b, entity_type=et)


def _contains(names: Set[str], name: str, et: str) -> bool:
    return any(_same(n, name, et) for n in names)


def compare(args: argparse.Namespace) -> None:
    runs = _load_runs(args.out)
    models = sorted(runs)
    arts = [json.loads(x) for x in open(os.path.join(args.out, "articles.jsonl"))]
    report: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, Dict[str, int]]] = {
        m: {et: defaultdict(int) for et in TYPES} for m in models
    }
    for a in arts:
        aid = a["id"]
        # names[model][type] = union over repeats; stable = in every repeat
        names: Dict[str, Dict[str, Set[str]]] = {}
        stable: Dict[str, Dict[str, Set[str]]] = {}
        for m in models:
            names[m], stable[m] = {}, {}
            for et in TYPES:
                per_run = [
                    {
                        _name(et, e)
                        for e in r.get(aid, {}).get("entities", {}).get(et, [])
                    }
                    for r in runs[m]
                ]
                union = set().union(*per_run) if per_run else set()
                names[m][et] = union
                stable[m][et] = {
                    n for n in union if all(_contains(p, n, et) for p in per_run)
                }
                totals[m][et]["mean_count"] += sum(map(len, per_run)) / max(
                    len(per_run), 1
                )
                totals[m][et]["unstable"] += len(union - stable[m][et])
        entry: Dict[str, Any] = {"article_id": aid, "title": a["title"], "types": {}}
        for et in TYPES:
            row: Dict[str, Any] = {}
            for m in models:
                others = [o for o in models if o != m]
                only = sorted(
                    n
                    for n in names[m][et]
                    if not any(_contains(names[o][et], n, et) for o in others)
                )
                row[m] = {
                    "all": sorted(names[m][et]),
                    "only_here": only,
                    "unstable": sorted(names[m][et] - stable[m][et]),
                }
                totals[m][et]["only_here"] += len(only)
            entry["types"][et] = row
        # same name, different type between models
        cross = []
        for m in models:
            for o in models:
                if m >= o:
                    continue
                for et in TYPES[:3]:
                    for ot in TYPES[:3]:
                        if et == ot:
                            continue
                        for n in names[m][et]:
                            if any(
                                _same(n, x, ot) for x in names[o][ot]
                            ) and not _contains(names[o][et], n, et):
                                cross.append({"name": n, m: et, o: ot})
        entry["type_disagreements"] = cross
        report.append(entry)

    with open(os.path.join(args.out, "diff.json"), "w") as f:
        json.dump({"models": models, "totals": totals, "articles": report}, f, indent=1)
    md = [f"# Extraction diff: {', '.join(models)}", ""]
    md.append("| model | type | mean found/article-set | only this model | unstable |")
    md.append("|---|---|---|---|---|")
    for m in models:
        for et in TYPES:
            t = totals[m][et]
            md.append(
                f"| {m} | {et} | {t['mean_count']:.0f} | {t['only_here']} | {t['unstable']} |"
            )
    for e in report:
        md += ["", f"## {e['title']} ({e['article_id']})"]
        for et in TYPES:
            for m in models:
                r = e["types"][et][m]
                if r["only_here"]:
                    md.append(f"- {et}, only {m}: {', '.join(r['only_here'])}")
        for c in e["type_disagreements"]:
            md.append(f"- type differs: {c}")
    with open(os.path.join(args.out, "diff.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {args.out}/diff.md and diff.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--domain", default="guantanamo")
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--workers", type=int, default=2, help="articles at once")
    r.add_argument("--min-chars", type=int, default=1500)
    r.add_argument("--max-chars", type=int, default=12000)
    r.add_argument("--overwrite", action="store_true")
    r.add_argument(
        "--cloud-wrapped",
        action="store_true",
        help="cloud models return the list in one tool call (as local ones do)",
    )
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    args = p.parse_args()
    run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    main()
