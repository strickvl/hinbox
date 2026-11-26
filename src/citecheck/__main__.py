"""Command line for the tuned-checker data: ``just citecheck <step> ...``.

Citation checker, in order:

    just citecheck sample  --entities <run> --articles <parquet> --out <dir>
    just citecheck label   --items <dir>/test_items.jsonl --labels <dir>/test_labels.jsonl --articles <parquet>
    just citecheck review  --items ... --labels <yours> --agent-labels <theirs> --articles <parquet>
    just citecheck combine --labels <theirs> <yours> <review> --out <combined>   # later files win
    just citecheck score   --items ... --labels ... --stored <run>         # the run's own judge
    just citecheck score   --items ... --labels ... --articles ... --model ollama/<m>
    just citecheck plant   --entities <run> --articles <parquet> --out <dir>
    just citecheck teach   --entities <run> --articles <parquet> --out <dir> --n 12000 [--dry-run]
    just citecheck audit   --out <dir> --n 100      # then label <dir>/audit_items.jsonl
    just citecheck export  --articles <parquet> --out <dir>
    just citecheck prompts --items ... --articles ... --file <dir>/test_prompts.jsonl
    just citecheck score   --items ... --labels ... --replies <replies.jsonl>

Merge check:

    just citecheck merge-data  --entities <run> --eval-dir <eval> --out <dir> [--teach]
    just citecheck merge-score --eval-dir <eval> --replies <replies.jsonl>

Kitaru (``src/citecheck/kitaru_loop.py``; needs ``uv sync --extra kitaru``):

    just citecheck kitaru-sessions    --items ... --articles ... --model <judge> --file <dir>/sessions.jsonl [--labels ...]
    kitaru session import <dir>/sessions.jsonl --importer kitaru/kitaru-jsonl@latest --agent hinbox-citation-check@latest
    just citecheck kitaru-investigate --name "Citation test set"
    just citecheck kitaru-pull        --investigation <id> --labels <dir>/test_labels.jsonl
    just citecheck kitaru-experiment  --name "Haiku vs tuned" --model tinker://... [--prices prices.json]

Everything written holds article text: keep ``--out`` under ``data/``.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

from src.citecheck.items import (
    article_meta,
    article_texts,
    claim_rows,
    latest_labels,
    read_jsonl,
    sample_test_set,
    write_jsonl,
)


def _entities(path: str) -> Dict[str, Dict]:
    from src.utils.file_ops import load_existing_entities

    return load_existing_entities(path)


def _teachers(raw: str) -> tuple:
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def cmd_sample(a: argparse.Namespace) -> None:
    path = os.path.join(a.out, "test_items.jsonl")
    if os.path.exists(path) and not a.force:
        sys.exit(f"{path} exists: the test set is frozen once drawn (--force redraws)")
    texts = article_texts(a.articles)
    rows = claim_rows(_entities(a.entities))
    items = sample_test_set(rows, n=a.n, seed=a.seed, have_text=texts.keys())
    write_jsonl(path, items)
    splits: Dict[str, int] = {}
    for r in rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    strata: Dict[str, int] = {}
    for r in items:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    print(f"{len(rows)} cited claims; by article split {splits}")
    print(f"wrote {len(items)} test items {strata} to {path}")


def cmd_label(a: argparse.Namespace) -> None:
    import uvicorn

    from src.citecheck.label_app import make_app

    items = read_jsonl(a.items)
    if not items:
        sys.exit(f"no items in {a.items}")
    app = make_app(
        items,
        a.labels,
        article_texts(a.articles),
        labeller=a.who,
        meta=article_meta(a.articles),
    )
    print(f"labelling {len(items)} items on http://127.0.0.1:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


def cmd_review(a: argparse.Namespace) -> None:
    """Serve the items where your labels and another labeller's differ; your
    new answers go to ``--out``, then ``combine`` puts the files together."""
    import uvicorn

    from src.citecheck.label_app import make_app
    from src.citecheck.review import review_items

    items = review_items(read_jsonl(a.items), a.labels, a.agent_labels, a.agent_name)
    if not items:
        sys.exit("no items where the two label files disagree")
    app = make_app(
        items,
        a.out,
        article_texts(a.articles),
        labeller=a.who,
        meta=article_meta(a.articles),
    )
    print(f"reviewing {len(items)} disagreements on http://127.0.0.1:{a.port}")
    print(
        f"answers go to {a.out}; then: just citecheck combine --labels "
        f"{a.agent_labels} {a.labels} {a.out} --out <combined.jsonl>"
    )
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


def cmd_combine(a: argparse.Namespace) -> None:
    from src.citecheck.review import combine_labels

    counts = combine_labels(a.labels, a.out)
    print(f"wrote {a.out}: {sum(counts.values())} items {counts} (later files win)")


def cmd_plant(a: argparse.Namespace) -> None:
    from src.citecheck.planted import planted_claims

    rows = claim_rows(_entities(a.entities))
    pairs = {(r["entity_name"], r["article_id"]) for r in rows}
    made = planted_claims(
        pairs, article_texts(a.articles), per_pair=a.per_pair, seed=a.seed
    )
    path = os.path.join(a.out, "planted.jsonl")
    write_jsonl(path, made)
    kinds: Dict[str, int] = {}
    for r in made:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(made)} planted claims {kinds} to {path}")


def cmd_teach(a: argparse.Namespace) -> None:
    from src.citecheck.teach import (
        article_chunks,
        estimate_tokens,
        run_teachers,
        teacher_sample,
    )

    texts = article_texts(a.articles)
    claims = teacher_sample(claim_rows(_entities(a.entities)), set(texts), a.n, a.seed)
    chunks = article_chunks(claims)
    teachers = _teachers(a.teachers)
    tokens = estimate_tokens(chunks, texts)
    print(
        f"{len(claims)} claims from {len({c['article_id'] for c in claims})} articles, "
        f"{len(chunks)} calls per teacher, about {tokens / 1e6:.1f}M prompt tokens "
        f"per teacher ({', '.join(teachers)})"
    )
    if a.dry_run:
        return
    path = os.path.join(a.out, "teacher_verdicts.jsonl")
    print(run_teachers(claims, texts, path, teachers=teachers, workers=a.workers))


def _agreed(out: str, teachers: tuple) -> List[Dict[str, Any]]:
    from src.citecheck.teach import agreed_labels

    labels, dropped = agreed_labels(
        read_jsonl(os.path.join(out, "teacher_verdicts.jsonl")), teachers
    )
    print(f"{len(labels)} teacher labels agreed; left out {dropped}")
    return labels


def cmd_audit(a: argparse.Namespace) -> None:
    labels = _agreed(a.out, _teachers(a.teachers))
    rng = random.Random(a.seed)
    picked = rng.sample(labels, min(a.n, len(labels)))
    keep = ("item_id", "entity_name", "article_id", "claim", "level", "reasoning")
    path = os.path.join(a.out, "audit_items.jsonl")
    write_jsonl(path, [{k: r.get(k) for k in keep} for r in picked])
    print(f"wrote {len(picked)} audit items to {path}")


def cmd_export(a: argparse.Namespace) -> None:
    from src.citecheck.export import build_examples, split_labels

    labels: List[Dict[str, Any]] = []
    if not a.no_teachers:
        labels += _agreed(a.out, _teachers(a.teachers))
    if not a.no_planted:
        labels += read_jsonl(os.path.join(a.out, "planted.jsonl"))
    texts = article_texts(a.articles)
    summary: Dict[str, Any] = {}
    for name, part in zip(("train", "tune"), split_labels(labels)):
        examples, counts = build_examples(part, texts, max_claims=a.max_claims)
        write_jsonl(os.path.join(a.out, f"{name}.jsonl"), examples)
        summary[name] = counts
    with open(os.path.join(a.out, "export_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def cmd_prompts(a: argparse.Namespace) -> None:
    from src.citecheck.score import prompt_rows

    rows = prompt_rows(read_jsonl(a.items), article_texts(a.articles))
    write_jsonl(a.file, rows)
    print(f"wrote {len(rows)} prompts to {a.file}")


def cmd_score(a: argparse.Namespace) -> None:
    from src.citecheck import score as s

    items = read_jsonl(a.items)
    labels = latest_labels(a.labels)
    extra: Dict[str, Any] = {}
    if a.stored:
        predicted = s.stored_predictions(items, claim_rows(_entities(a.stored)))
        name = f"stored verdicts in {a.stored}"
    elif a.replies:
        predicted = s.reply_predictions(read_jsonl(a.replies))
        name = f"replies in {a.replies}"
    elif a.model:
        if not a.articles:
            sys.exit("--model needs --articles")
        predicted, extra = s.predict_with_model(
            items, article_texts(a.articles), a.model, workers=a.workers
        )
        name = a.model
    else:
        sys.exit("give one of --stored, --replies or --model")
    report = {**s.score(items, labels, predicted), **extra, "judge": name}
    print(s.describe(report, name))
    if extra:
        print(f"  {extra['seconds']} s for {extra['articles']} articles")
    if a.save:
        with open(a.save, "w") as f:
            json.dump({"report": report, "predicted": predicted}, f, indent=1)


def cmd_merge_data(a: argparse.Namespace) -> None:
    from src.citecheck import merge_data as m

    gold = m.gold_pairs(a.eval_dir)
    test = [p for p in gold if p["split"] == "test"]
    avoid = {p["inputs"][k] for p in test for k in ("new_name", "existing_name")}
    split = m.split_positives(_entities(a.entities), a.split_pages, avoid_names=avoid)
    train_gold = [p for p in gold if p["split"] != "test"]
    pairs = train_gold + split
    teachers = _teachers(a.teachers)
    teacher_path = os.path.join(a.out, "merge_teacher.jsonl")
    if a.teach:
        print(m.teach_pairs(pairs, teachers, teacher_path))
    rows, dropped = m.agreed_pairs(pairs, read_jsonl(teacher_path), teachers)
    print(
        f"{len(rows)} of {len(pairs)} training pairs have agreed reasoning; {dropped}"
    )
    write_jsonl(os.path.join(a.out, "merge_train.jsonl"), m.merge_examples(rows))
    write_jsonl(os.path.join(a.out, "merge_test_prompts.jsonl"), m.merge_prompts(test))
    write_jsonl(os.path.join(a.out, "merge_test_pairs.jsonl"), test)
    print(
        f"{len(train_gold)} hand-labelled + {len(split)} split-page pairs for training; "
        f"{len(test)} hand-labelled test pairs ({sum(p['truth'] for p in test)} same)"
    )


def cmd_merge_score(a: argparse.Namespace) -> None:
    from src.citecheck import merge_data as m

    test = [p for p in m.gold_pairs(a.eval_dir) if p["split"] == "test"]
    print(json.dumps(m.score_replies(test, read_jsonl(a.replies)), indent=1))


def cmd_kitaru_sessions(a: argparse.Namespace) -> None:
    from src.citecheck.kitaru_loop import build_sessions

    items = read_jsonl(a.items)
    labels = latest_labels(a.labels) if a.labels else {}
    counts = build_sessions(items, article_texts(a.articles), labels, a.model, a.file)
    print(json.dumps(counts))


def _ids(path: str) -> Any:
    return {r["item_id"] for r in read_jsonl(path)} if path else None


def cmd_kitaru_investigate(a: argparse.Namespace) -> None:
    import asyncio

    from src.citecheck.kitaru_loop import investigate

    inv, n = asyncio.run(investigate(a.name, _ids(a.items)))
    print(f"investigation {inv.id}: {n} sessions, one question each")


def cmd_kitaru_pull(a: argparse.Namespace) -> None:
    import asyncio

    from src.citecheck.kitaru_loop import pull

    print(json.dumps(asyncio.run(pull(a.investigation, a.labels, a.who))))


def cmd_kitaru_experiment(a: argparse.Namespace) -> None:
    import asyncio

    from src.citecheck.kitaru_loop import describe_table, experiment, save_json

    prices = json.load(open(a.prices, encoding="utf-8")) if a.prices else None
    out = asyncio.run(
        experiment(a.name, a.model, prices, _ids(a.items), wait=not a.no_wait)
    )
    if out.get("table"):
        print(describe_table(out["table"], a.model))
    if a.save:
        save_json(a.save, out)
    print(json.dumps({k: v for k, v in out.items() if k != "table"}))


def main(argv: List[str] = None) -> None:
    from src.citecheck.teach import TEACHERS

    p = argparse.ArgumentParser(prog="citecheck", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    teachers = ",".join(TEACHERS)

    def add(name: str, fn, *args: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        for arg in args:
            sp.add_argument(f"--{arg}", required=True)
        return sp

    sp = add("sample", cmd_sample, "entities", "articles", "out")
    sp.add_argument("--n", type=int, default=400)
    sp.add_argument("--seed", type=int, default=20261126)
    sp.add_argument("--force", action="store_true")
    sp = add("label", cmd_label, "items", "labels", "articles")
    sp.add_argument("--port", type=int, default=5078)
    sp.add_argument("--who", default=os.getenv("HINBOX_REVIEWER", ""))
    sp = add("plant", cmd_plant, "entities", "articles", "out")
    sp.add_argument("--per-pair", type=int, default=2)
    sp.add_argument("--seed", type=int, default=7)
    sp = add("teach", cmd_teach, "entities", "articles", "out")
    sp.add_argument("--n", type=int, default=12000)
    sp.add_argument("--seed", type=int, default=11)
    sp.add_argument("--teachers", default=teachers)
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--dry-run", action="store_true")
    sp = add("audit", cmd_audit, "out")
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--teachers", default=teachers)
    sp = add("export", cmd_export, "articles", "out")
    sp.add_argument("--max-claims", type=int, default=20)
    sp.add_argument("--teachers", default=teachers)
    sp.add_argument("--no-planted", action="store_true")
    sp.add_argument("--no-teachers", action="store_true")
    add("prompts", cmd_prompts, "items", "articles", "file")
    sp = add("review", cmd_review, "items", "labels", "agent-labels", "articles")
    sp.add_argument("--out", default="data/citecheck/review_labels.jsonl")
    sp.add_argument("--agent-name", default="")
    sp.add_argument("--port", type=int, default=5078)
    sp.add_argument("--who", default=os.getenv("HINBOX_REVIEWER", ""))
    sp = add("combine", cmd_combine, "out")
    sp.add_argument("--labels", nargs="+", required=True)
    sp = add("score", cmd_score, "items", "labels")
    sp.add_argument("--articles")
    sp.add_argument("--model")
    sp.add_argument("--stored")
    sp.add_argument("--replies")
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--save")
    sp = add("merge-data", cmd_merge_data, "entities", "eval-dir", "out")
    sp.add_argument("--split-pages", type=int, default=1500)
    sp.add_argument("--teach", action="store_true")
    sp.add_argument("--teachers", default=teachers)
    add("merge-score", cmd_merge_score, "eval-dir", "replies")
    sp = add(
        "kitaru-sessions", cmd_kitaru_sessions, "items", "articles", "model", "file"
    )
    sp.add_argument("--labels")
    sp = add("kitaru-investigate", cmd_kitaru_investigate, "name")
    sp.add_argument("--items", help="only these test items (default: all)")
    sp = add("kitaru-pull", cmd_kitaru_pull, "investigation", "labels")
    sp.add_argument("--who", default=os.getenv("HINBOX_REVIEWER", "kitaru"))
    sp = add("kitaru-experiment", cmd_kitaru_experiment, "name", "model")
    sp.add_argument("--items", help="only these test items (default: all)")
    sp.add_argument("--prices", help="JSON file: model -> [$/M in, $/M out]")
    sp.add_argument("--save", help="write the run and its table here as JSON")
    sp.add_argument("--no-wait", action="store_true")

    a = p.parse_args(argv)
    # ``--out`` is a folder, except for review and combine, whose ``--out`` is
    # a label file (they make its folder themselves)
    if getattr(a, "out", None) and a.fn not in (cmd_review, cmd_combine):
        os.makedirs(a.out, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
