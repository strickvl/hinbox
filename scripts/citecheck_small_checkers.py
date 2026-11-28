# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main",
#   "transformers>=4.40",
#   "torch>=2.2",
#   "pyarrow>=15",
# ]
# ///
"""Score small open fact checkers on the citation test set, beside Inkling.

Each checker gives every (article, claim) pair a score from 0 to 1, the
chance the article backs the claim. A checker "passes" a claim when its
score is at or above a threshold. Two thresholds are reported:

- 0.5, the checker's own default;
- the threshold at which it warns on (does not pass) exactly ``--warnings``
  labelled claims, the number Inkling warns on for the same items. At equal
  warnings, the two can be compared on which claims they warn on: fewer
  wrong claims passed means the warnings land in better places.

Checkers (``--checkers``, comma list):

- ``hhem``: Vectara HHEM-2.1-Open (``vectara/hallucination_evaluation_model``,
  T5-base, transformers). The article is cut into overlapping chunks of
  about ``--chunk-words`` words and the claim's score is its best chunk's.
- ``flan-t5``: MiniCheck-Flan-T5-Large through the ``minicheck`` package,
  which chunks the article itself and takes the best chunk.
- ``bespoke-7b``: Bespoke-MiniCheck-7B through the same package; it needs
  vLLM, so a CUDA machine (not a Mac).

Run on the random half of the test set only (``--stratum random``): the
other half was drawn from claims Qwen doubted, which these checkers never
saw. Scores go to ``<out>/<checker>.jsonl`` (one row per claim, resumable)
and the table to ``<out>/RESULT.md``. The scores hold no article text, but
the item ids point at test claims: keep ``<out>`` under ``data/``.

    uv run --script scripts/citecheck_small_checkers.py \\
      --items data/citecheck/distill_v4/test_items.jsonl \\
      --labels data/citecheck/test_labels_combined.jsonl \\
      --articles <articles parquet> --out data/citecheck/small_checkers \\
      --stratum random --warnings <Inkling's warnings> --checkers hhem,flan-t5
"""

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import pyarrow.parquet as pq


def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_labels(path: str) -> Dict[str, str]:
    """Item id -> yes/no/unsure; a later row wins."""
    return {r["item_id"]: r["label"] for r in read_jsonl(path)}


def article_texts(path: str) -> Dict[str, str]:
    t = pq.read_table(path, columns=["id", "content"]).to_pydict()
    return {str(i): c or "" for i, c in zip(t["id"], t["content"])}


def chunks(text: str, words: int, overlap: int = 50) -> List[str]:
    toks = text.split()
    if len(toks) <= words:
        return [text]
    step = max(1, words - overlap)
    return [" ".join(toks[i : i + words]) for i in range(0, len(toks) - overlap, step)]


def hhem_scorer(chunk_words: int) -> Callable[[str, List[str]], List[float]]:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    )
    model.eval()

    def score(doc: str, claims: List[str]) -> List[float]:
        parts = chunks(doc, chunk_words)
        pairs = [(p, c) for c in claims for p in parts]
        out: List[float] = []
        for i in range(0, len(pairs), 16):
            out += [float(x) for x in model.predict(pairs[i : i + 16])]
        n = len(parts)
        return [max(out[k * n : (k + 1) * n]) for k in range(len(claims))]

    return score


def minicheck_scorer(name: str, cache: str) -> Callable[[str, List[str]], List[float]]:
    from minicheck.minicheck import MiniCheck

    model = MiniCheck(model_name=name, cache_dir=cache)

    def score(doc: str, claims: List[str]) -> List[float]:
        _, probs, _, _ = model.score(docs=[doc] * len(claims), claims=claims)
        return [float(p) for p in probs]

    return score


def threshold_for(scored: List[Tuple[float, str]], warnings: int) -> float:
    """The lowest threshold at which at most ``warnings`` claims score below
    it (ties can leave fewer)."""
    ranked = sorted(s for s, _ in scored)
    if warnings <= 0:
        return ranked[0]
    if warnings >= len(ranked):
        return ranked[-1] + 1e-9
    return ranked[warnings]


def tally(scored: List[Tuple[float, str]], cut: float) -> Dict[str, int]:
    no = [s for s, lab in scored if lab == "no"]
    yes = [s for s, lab in scored if lab == "yes"]
    return {
        "wrong_passed": sum(s >= cut for s in no),
        "wrong": len(no),
        "good_doubted": sum(s < cut for s in yes),
        "good": len(yes),
        "warnings": sum(s < cut for s, _ in scored),
    }


def run(checker: str, scorer, groups, texts, out_path: str) -> float:
    """Scores every claim not yet in ``out_path``; seconds per article."""
    done = (
        {r["item_id"] for r in read_jsonl(out_path)}
        if os.path.exists(out_path)
        else set()
    )
    spent, articles = 0.0, 0
    with open(out_path, "a") as f:
        for aid, rows in groups.items():
            todo = [r for r in rows if r["item_id"] not in done]
            if not todo:
                continue
            start = time.time()
            scores = scorer(texts.get(aid, ""), [r["claim"] for r in todo])
            took = time.time() - start
            spent, articles = spent + took, articles + 1
            for r, s in zip(todo, scores):
                f.write(json.dumps({"item_id": r["item_id"], "score": s}) + "\n")
            f.flush()
            print(f"[{checker}] {aid}: {len(todo)} claims in {took:.1f} s", flush=True)
    return spent / articles if articles else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--items", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--articles", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stratum", default="random", help="'' for every item")
    p.add_argument("--warnings", type=int, required=True, help="Inkling's warnings")
    p.add_argument("--checkers", default="hhem,flan-t5")
    p.add_argument("--chunk-words", type=int, default=350)
    p.add_argument("--cache", default="data/citecheck/small_checkers/models")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    labels = latest_labels(a.labels)
    items = [
        r
        for r in read_jsonl(a.items)
        if (not a.stratum or r.get("stratum") == a.stratum)
        and labels.get(r["item_id"]) in ("yes", "no")
    ]
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in items:
        groups[r["article_id"]].append(r)
    texts = article_texts(a.articles)
    makers = {
        "hhem": lambda: hhem_scorer(a.chunk_words),
        "flan-t5": lambda: minicheck_scorer("flan-t5-large", a.cache),
        "bespoke-7b": lambda: minicheck_scorer("Bespoke-MiniCheck-7B", a.cache),
    }
    lines = [
        f"{len(items)} labelled claims ({a.stratum or 'all'} items) over {len(groups)} "
        f"articles; equal warnings = {a.warnings}",
        "",
        "| checker | threshold | wrong passed | good doubted | warnings | s per article |",
        "|---|---|---|---|---|---|",
    ]
    for checker in [c.strip() for c in a.checkers.split(",") if c.strip()]:
        out_path = os.path.join(a.out, f"{checker}.jsonl")
        per_article = run(checker, makers[checker](), groups, texts, out_path)
        got = {r["item_id"]: r["score"] for r in read_jsonl(out_path)}
        scored = [
            (got[r["item_id"]], labels[r["item_id"]])
            for r in items
            if r["item_id"] in got
        ]
        for name, cut in (
            ("0.5", 0.5),
            ("equal warnings", threshold_for(scored, a.warnings)),
        ):
            t = tally(scored, cut)
            lines.append(
                f"| {checker} | {name} ({cut:.3f}) | {t['wrong_passed']}/{t['wrong']} "
                f"| {t['good_doubted']}/{t['good']} | {t['warnings']} | {per_article:.1f} |"
            )
    with open(os.path.join(a.out, "RESULT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
