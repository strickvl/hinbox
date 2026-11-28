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

Several answers per claim and paired comparisons (``src/citecheck/votes.py``):

    just citecheck votes   --items ... --labels ... --replies <replies with --votes N> [--save <file>]
    just citecheck paired  --items ... --labels ... --sources untuned=<file> v4=<file> ... [--core-of a,b --audit-of a,b --audit-out <ids>]

Claims self-distillation never gets right (``src/citecheck/hard.py``):

    just citecheck hard    --export <export dir> --distilled <distill dir> --out <dir> [--planted <planted.jsonl>]
    (then ``distill`` <dir>/hard_train.jsonl and hard_tune.jsonl with more samples)
    just citecheck reteach --prompts <dir>/hard_train.jsonl --log <distilled>.log.jsonl --teacher <model> --out <file> --max-dollars 5 --price-in X --price-out Y [--dry-run]
    just citecheck mix     --base <distill dir> --extra <kept rows>... --out <dir> [--repeat 4] [--planted ... --drop-kinds planted:year,planted:number]

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
    from src.citecheck.export import build_examples, cap_planted, split_labels

    labels: List[Dict[str, Any]] = []
    if not a.no_teachers:
        from src.citecheck.teach import same_rules

        path = os.path.join(a.out, "teacher_verdicts.jsonl")
        same_rules(read_jsonl(path), path, "set HINBOX_GROUNDING_RULES to match")
        labels += _agreed(a.out, _teachers(a.teachers))
    if not a.no_planted:
        labels += read_jsonl(os.path.join(a.out, "planted.jsonl"))
    texts = article_texts(a.articles)
    summary: Dict[str, Any] = {}
    for name, part in zip(("train", "tune"), split_labels(labels)):
        if a.planted_ratio is not None:
            part = cap_planted(part, a.planted_ratio)
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
        from src.utils.grounding import GROUNDING_RULES

        name = a.model + (" (rules=strict)" if GROUNDING_RULES == "strict" else "")
    else:
        sys.exit("give one of --stored, --replies or --model")
    report = {**s.score(items, labels, predicted), **extra, "judge": name}
    print(s.describe(report, name))
    if extra:
        print(f"  {extra['seconds']} s for {extra['articles']} articles")
    if a.save:
        with open(a.save, "w") as f:
            json.dump({"report": report, "predicted": predicted}, f, indent=1)


def cmd_votes(a: argparse.Namespace) -> None:
    from src.citecheck import votes as v

    items, labels = read_jsonl(a.items), latest_labels(a.labels)
    report = v.vote_report(items, labels, v.by_sample(read_jsonl(a.replies)))
    print(v.describe_votes(report, a.replies))
    if a.save:
        with open(a.save, "w") as f:
            json.dump(report, f, indent=1)


def cmd_paired(a: argparse.Namespace) -> None:
    from src.citecheck import votes as v
    from src.citecheck.score import score

    items, labels = read_jsonl(a.items), latest_labels(a.labels)
    sources: Dict[str, Any] = {}
    for spec in a.sources:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--sources takes name=path, got {spec!r}")
        sources[name] = v.load_predictions(path)
    base_name = next(iter(sources))
    base = sources[base_name][0]
    core_names = _teachers(a.core_of) or tuple(sources)
    core = v.wrong_passed_by_all(items, labels, (sources[n][0] for n in core_names))
    print(
        f"{len(core)} wrong claims passed by every one of {', '.join(core_names)}; "
        f"compared with {base_name}: 'better/worse' counts claims only one of the two "
        "got right, p is a two-sided sign test on them"
    )
    print(
        "| checker | read as | wrong passed | good doubted | failed | wrong: better/worse (p) "
        "| good: better/worse (p) | core caught |"
    )
    print("|---|---|---|---|---|---|---|---|")
    rows = {}
    for name, (pred, kind) in sources.items():
        t = score(items, labels, pred)
        pr = v.paired(items, labels, base, pred)
        caught = sum(pred.get(i) != "supported" for i in core)
        w, g = pr["wrong"], pr["good"]
        print(
            f"| {name} | {kind} | {t['wrong_passed']['count']} | {t['good_doubted']['count']} "
            f"| {t['errors']} | {w['other_better']}/{w['other_worse']} ({w['p']}) "
            f"| {g['other_better']}/{g['other_worse']} ({g['p']}) | {caught}/{len(core)} |"
        )
        rows[name] = {"kind": kind, "paired": pr, "core_caught": caught}
    if a.audit_out:
        names = _teachers(a.audit_of) or tuple(sources)
        ids = v.all_disagree(items, labels, (sources[n][0] for n in names))
        with open(a.audit_out, "w") as f:
            f.write("\n".join(ids) + ("\n" if ids else ""))
        print(
            f"{len(ids)} claims where {', '.join(names)} all disagree with the key: {a.audit_out}"
        )
    if a.save:
        with open(a.save, "w") as f:
            json.dump({"base": base_name, "core": core, "rows": rows}, f, indent=1)


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


def _kinds(path: str) -> Dict:
    from src.citecheck.hard import label_kinds

    return label_kinds(read_jsonl(path)) if path else {}


def cmd_hard(a: argparse.Namespace) -> None:
    from src.citecheck.hard import hard_claims, hard_prompts, hard_summary

    kinds = _kinds(a.planted)
    summary: Dict[str, Any] = {}
    for split in ("train", "tune"):
        examples = read_jsonl(os.path.join(a.export, f"{split}.jsonl"))
        log = read_jsonl(os.path.join(a.distilled, f"{split}.jsonl.log.jsonl"))
        if not examples or not log:
            sys.exit(
                f"no {split}.jsonl in {a.export} or no distill log in {a.distilled}"
            )
        rows = hard_prompts(hard_claims(examples, log, a.min_answers), kinds)
        write_jsonl(os.path.join(a.out, f"hard_{split}.jsonl"), rows)
        summary[split] = {
            "examples": len(examples),
            "tried": len(log),
            "hard_claims": len(rows),
            "by_kind": hard_summary(rows),
        }
    with open(os.path.join(a.out, "hard_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def cmd_reteach(a: argparse.Namespace) -> None:
    from src.citecheck.hard import estimate_dollars, missed, teacher_rows

    prompts = missed(read_jsonl(a.prompts), read_jsonl(a.log))
    cost = estimate_dollars(prompts, a.price_in, a.price_out)
    print(
        f"{len(prompts)} hard prompts never kept; estimated ${cost:.2f} with {a.teacher}"
    )
    if cost > a.max_dollars:
        sys.exit(
            f"estimate ${cost:.2f} is over --max-dollars {a.max_dollars}: stopping"
        )
    if a.dry_run:
        return
    rows, counts = teacher_rows(prompts, a.teacher)
    write_jsonl(a.out, rows)
    print(
        json.dumps(
            {"teacher": a.teacher, "estimated_dollars": round(cost, 2), **counts}
        )
    )


def cmd_mix(a: argparse.Namespace) -> None:
    import shutil

    from src.citecheck.hard import mixed

    kinds = _kinds(a.planted)
    drop = tuple(k.strip() for k in a.drop_kinds.split(",") if k.strip())
    if drop and not kinds:
        sys.exit("--drop-kinds needs --planted")
    summary: Dict[str, Any] = {}
    for split in ("train", "tune"):
        base = read_jsonl(os.path.join(a.base, f"{split}.jsonl"))
        extra = [
            r
            for path in a.extra
            if f"_{split}." in os.path.basename(path)
            for r in read_jsonl(path)
        ]
        rows, counts = mixed(
            base,
            extra,
            a.repeat if split == "train" else 1,
            kinds=kinds,
            drop_kinds=drop,
        )
        write_jsonl(os.path.join(a.out, f"{split}.jsonl"), rows)
        summary[split] = counts
    for name in ("test_items.jsonl", "test_labels.jsonl", "test_prompts.jsonl"):
        src = os.path.join(a.base, name)
        if not os.path.exists(src):
            sys.exit(f"{src} is missing: the pipeline scores from it")
        shutil.copyfile(src, os.path.join(a.out, name))
    with open(os.path.join(a.out, "mix_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


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
    sp.add_argument(
        "--planted-ratio",
        type=float,
        default=None,
        help="at most this many planted claims per teacher claim, in each split",
    )
    add("prompts", cmd_prompts, "items", "articles", "file")
    sp = add("hard", cmd_hard, "export", "distilled", "out")
    sp.add_argument("--planted", default="", help="planted.jsonl, to count kinds")
    sp.add_argument("--min-answers", type=int, default=3)
    sp = add("reteach", cmd_reteach, "prompts", "log", "teacher", "out")
    sp.add_argument("--max-dollars", type=float, required=True)
    sp.add_argument("--price-in", type=float, required=True, help="$ per M tokens")
    sp.add_argument("--price-out", type=float, required=True, help="$ per M tokens")
    sp.add_argument("--dry-run", action="store_true")
    sp = add("mix", cmd_mix, "base", "out")
    sp.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="kept rows; a name holding _train. or _tune. picks the split",
    )
    sp.add_argument("--repeat", type=int, default=4, help="times each extra train row")
    sp.add_argument("--planted", default="")
    sp.add_argument("--drop-kinds", default="")
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
    sp = add("votes", cmd_votes, "items", "labels", "replies")
    sp.add_argument("--save")
    sp = add("paired", cmd_paired, "items", "labels")
    sp.add_argument(
        "--sources", nargs="+", required=True, help="name=path; the first is the base"
    )
    sp.add_argument(
        "--core-of", default="", help="names: core = wrong claims all of them pass"
    )
    sp.add_argument("--audit-of", default="", help="names for --audit-out")
    sp.add_argument("--audit-out", default="")
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
    if getattr(a, "out", None) and a.fn not in (cmd_review, cmd_combine, cmd_reteach):
        os.makedirs(a.out, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
