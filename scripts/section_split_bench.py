"""Try the section split (``src/engine/section_split.py``) on real profiles.

``sections`` runs the split on every long section of a finished run's
profiles (hubs left out, since they are never updated) and writes:

- ``review.md``: for each section, the sub-headings and the start of each
  paragraph placed under them, for a person to check where things landed;
- ``summary.json``: sections found, split, refused (and why) or failed;
  whether every sentence survived; and how many cited claims the next
  grounding pass would send to the judge again (from a no-model grounding
  of the profile before the split, so 0 is the goal).

``cases`` splits the old profile of every profile-update eval case and writes
a copy of the eval folder with those profiles, so the profile-update eval can
score sectioned against flat profiles on the same cases.

``--reorganise`` first gives every profile over ``--reorganise-chars``
(default 8000) its kind's starter headings (``section_reorganise.py``), then
splits what is still too long, as the pipeline does with both steps on. The
review then lists each new section and the start of every paragraph under it:

    uv run python -m scripts.section_split_bench sections --entities <dir> \\
        --articles <articles.parquet> --model <m> --out <dir> [--min-chars 6000]
    uv run python -m scripts.section_split_bench cases --eval-dir <eval> \\
        --model <m> --out <dir> [--min-chars 3000] [--reorganise]
    just eval profile-update --model <m> --split dev --eval-dir <dir>/eval

Outputs hold profile text: keep them off GitHub.
"""

import argparse
import json
import os
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

_PATCH_LOCK = threading.Lock()


def _asker(model: str):
    from src.engine.section_split import make_asker

    return make_asker(model, model.startswith("ollama/"))


def _fake_judge(*, messages, response_model, temperature, **kw):
    from src.utils.grounding import ClaimVerification, SupportLevel

    sent = messages[1]["content"].split("## CLAIMS TO VERIFY")[1]
    n = len([ln for ln in sent.splitlines() if ln[:1].isdigit()])
    return [
        ClaimVerification(
            article_id="x", citation="?", claim="?", support_level=SupportLevel.PARTIAL
        )
        for _ in range(n)
    ]


def regrounded(before: str, after: str, articles: Dict[str, str], name: str) -> int:
    """Claims of ``after`` the grounding pass would judge again, given a report
    on ``before``. No model calls: the first report comes from a fake judge."""
    from src.utils.grounding import plan_grounding, verify_profile_grounding

    # The patch replaces a module-wide function, so worker threads take turns:
    # one finishing early must not put the real judge back under another
    # Both paths: HINBOX_GROUNDING_MODEL=ollama/... sends the judge to local_generation
    with (
        _PATCH_LOCK,
        patch("src.utils.llm.cloud_generation", side_effect=_fake_judge),
        patch("src.utils.llm.local_generation", side_effect=_fake_judge),
    ):
        report = verify_profile_grounding(
            profile_text=before, article_texts=articles, entity_name=name
        )
    plan = plan_grounding(
        profile_text=after,
        article_texts=articles,
        previous=report.model_dump(),
        entity_name=name,
    )
    return len(plan.pending())


def _split_one(
    text: str, etype: str, name: str, ask: Any, min_chars: int, reorg: Any = None
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Split ``text``; with ``reorg`` = (starter headings, asker, min chars),
    reorganise it first. Returns the new text, the reports (the reorganise
    one first, ``section`` "*") and the text the split started from."""
    from src.engine.section_reorganise import reorganise_text
    from src.engine.section_split import split_long_sections

    first: List[Dict[str, Any]] = []
    if reorg:
        starter, reorg_ask, reorg_chars, *notes = reorg
        text, report = reorganise_text(
            text, etype, name, starter, reorg_ask, min_chars=reorg_chars,
            notes=notes[0] if notes else None,
        )  # fmt: skip
        if report["status"] != "skipped":
            first = [{"section": "*", **report}]
    new_text, reports = split_long_sections(text, etype, name, ask, min_chars=min_chars)
    return new_text, first + reports, text


def _reorg(args: argparse.Namespace, etype: str, kind: Any) -> Any:
    if not args.reorganise:
        return None
    from src.config_loader import get_domain_config
    from src.engine.section_reorganise import starter_headings, starter_notes
    from src.engine.section_split import make_asker

    config = get_domain_config(args.domain).load_config()
    local = args.model.startswith("ollama/")
    ask = make_asker(args.model, local, step="profile.section_reorganise")
    notes = starter_notes(config, etype, kind)
    return starter_headings(config, etype, kind), ask, args.reorganise_chars, notes


def _reorg_units(text: str) -> List[str]:
    from src.engine.section_reorganise import _opening_and_body
    from src.engine.section_split import _units

    return _units(_opening_and_body(text)[1])[0]


def _review(
    name: str, etype: str, reports: List[Dict[str, Any]], units: Dict[str, List[str]]
) -> str:
    lines = [f"## {name} ({etype})", ""]
    for r in reports:
        lines.append(
            f"### {r['section']}: {r['status']} ({r['paragraphs']} paragraphs, "
            f"{r['chars']} chars)" + (f": {r['reason']}" if r.get("reason") else "")
        )
        for heading, numbers in r.get("subsections") or r.get("sections") or []:
            lines.append(f"- **{heading}**")
            for n in numbers:
                start = " ".join(units[r["section"]][n - 1].split())[:110]
                lines.append(f"  - [{n}] {start}…")
        lines.append("")
    return "\n".join(lines)


def sections(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    from src.engine.hub_profiles import is_hub
    from src.engine.section_split import _sentences, long_sections
    from src.process_and_extract import load_existing_entities

    entities = load_existing_entities(args.entities)
    articles = {
        str(r["id"]): r.get("content") or ""
        for r in pq.read_table(args.articles, columns=["id", "content"]).to_pylist()
    }
    kinds: Dict[str, str] = {}
    if args.kinds:  # older people tables don't store a type
        with open(args.kinds, encoding="utf-8") as f:
            kinds = json.load(f)
    jobs = []
    for etype, table in entities.items():
        for data in table.values():
            text = str((data.get("profile") or {}).get("text") or "")
            name = data.get("name") or data.get("title") or ""
            if is_hub(args.domain, etype, data.get("type")):
                continue
            long = args.reorganise and len(text) > args.reorganise_chars
            if long or long_sections(text, args.min_chars):
                kind = data.get("type") or data.get("event_type")
                jobs.append((etype, name, text, kinds.get(f"{etype}/{name}", kind)))
    jobs.sort(key=lambda j: -len(j[2]))
    jobs = jobs[: args.limit] if args.limit else jobs
    ask = _asker(args.model)

    def one(job):
        etype, name, text, kind = job
        reorg = _reorg(args, etype, kind)
        units = {"*": _reorg_units(text)} if reorg else {}
        new_text, reports, mid = _split_one(
            text, etype, name, ask, args.min_chars, reorg
        )
        units |= {s["heading"]: s["units"] for s in long_sections(mid, args.min_chars)}
        return {
            "entity": f"{etype}/{name}",
            "chars": len(text),
            "reports": reports,
            "sentences_kept": _sentences(new_text) == _sentences(text),
            "texts": (text, new_text, name),
            "review": _review(name, etype, reports, units),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, jobs))
    # After the pool: the fake judge patch would catch other threads' real calls
    for r in results:
        before, after, name = r.pop("texts")
        r["regrounded"] = regrounded(before, after, articles, name)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "review.md"), "w") as f:
        f.write(
            f"# Section split review ({args.model}, over {args.min_chars} chars"
            + (f", reorganised over {args.reorganise_chars}" if args.reorganise else "")
            + ")\n\n"
        )
        f.write("\n".join(r["review"] for r in results))
    status = Counter(rep["status"] for r in results for rep in r["reports"])
    reasons = Counter(
        rep.get("reason") for r in results for rep in r["reports"] if rep.get("reason")
    )
    summary = {
        "model": args.model,
        "min_chars": args.min_chars,
        "reorganise_chars": args.reorganise and args.reorganise_chars,
        "profiles": len(results),
        "sections": dict(status),
        "refusals": dict(reasons),
        "profiles_with_a_changed_sentence": sum(
            not r["sentences_kept"] for r in results
        ),
        "claims_regrounded": sum(r["regrounded"] for r in results),
        "per_profile": [
            {k: r[k] for k in ("entity", "chars", "sentences_kept", "regrounded")}
            | {"sections": [(x["section"], x["status"]) for x in r["reports"]]}
            for r in results
        ],
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "per_profile"}, indent=1)
    )


def cases(args: argparse.Namespace) -> None:
    from src.engine.section_split import _sentences

    src_path = os.path.join(args.eval_dir, "profile_update_cases.jsonl")
    with open(src_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ask = _asker(args.model)
    texts = {}
    for row in rows:  # cases share snapshots; split each text once
        profile = row["profile"]
        texts.setdefault(
            profile.get("text") or "",
            (row["entity_type"], row["entity_name"], _case_kind(row)),
        )

    def one(item):
        text, (etype, name, kind) = item
        reorg = _reorg(args, etype, kind)
        new_text, reports, _ = _split_one(text, etype, name, ask, args.min_chars, reorg)
        return text, new_text, reports

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        done = {t: (n, r) for t, n, r in pool.map(one, texts.items())}
    out_dir = os.path.join(args.out, "eval")
    os.makedirs(out_dir, exist_ok=True)
    changed = 0
    with open(os.path.join(out_dir, "profile_update_cases.jsonl"), "w") as f:
        for row in rows:
            new_text, _ = done[row["profile"].get("text") or ""]
            changed += new_text != (row["profile"].get("text") or "")
            row = {**row, "profile": {**row["profile"], "text": new_text}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("profile_update_reviews.jsonl", "claim_labels.jsonl"):
        if os.path.exists(os.path.join(args.eval_dir, name)):
            shutil.copy(os.path.join(args.eval_dir, name), os.path.join(out_dir, name))
    status = Counter(r["status"] for _, rs in done.values() for r in rs)
    summary = {
        "model": args.model,
        "min_chars": args.min_chars,
        "reorganise_chars": args.reorganise and args.reorganise_chars,
        "cases": len(rows),
        "cases_changed": changed,
        "profiles": len(texts),
        "sections": dict(status),
        "profiles_with_a_changed_sentence": sum(
            _sentences(n) != _sentences(t) for t, (n, _) in done.items()
        ),
    }
    with open(os.path.join(args.out, "cases_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def _case_kind(row: Dict[str, Any]) -> str:
    """A case's entity kind; older cases lack it, so look the name up."""
    from src.evals.profile_replay import DEFAULT_ENTITIES

    kinds = {e.name: e.kind for e in DEFAULT_ENTITIES}
    return row.get("entity_kind") or kinds.get(row["entity_name"], "")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sections", help="split a finished run's long sections")
    s.add_argument("--entities", required=True, help="folder with people.parquet")
    s.add_argument("--articles", required=True, help="articles Parquet file")
    s.add_argument(
        "--kinds", help='JSON {"people/<name>": "detainee"} for tables without a type'
    )
    s.add_argument("--limit", type=int, default=0, help="longest N profiles only")
    c = sub.add_parser("cases", help="split the profile-update eval cases")
    c.add_argument(
        "--eval-dir", required=True, help="folder with profile_update_cases.jsonl"
    )
    for p, default in ((s, 6000), (c, 3000)):
        p.add_argument("--model", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--min-chars", type=int, default=default)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--domain", default="guantanamo")
        p.add_argument("--reorganise", action="store_true")
        p.add_argument("--reorganise-chars", type=int, default=8000)
    args = parser.parse_args(argv)
    {"sections": sections, "cases": cases}[args.command](args)


if __name__ == "__main__":
    main()
