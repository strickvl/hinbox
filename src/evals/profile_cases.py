"""Build profile-update test cases from the snapshots ``just eval-replay`` saved.

Each case is "this existing profile + this article -> what should a good update
do?". Three kinds can be built without a model or a human, because we know the
right behaviour from how the case was made:

- ``irrelevant``: the profile after step k, plus an article that names the
  entity only once in passing. A good update changes almost nothing.
- ``new_fact``: the profile after step k, plus the real next article. A good
  update adds what is new, cites that article, and keeps everything else.
- ``disputed``: the profile after step k, plus a copy of the next article with
  one year changed (2002 -> 2003) where the profile already states that year
  with a citation. A good update keeps both years, each with its own source,
  and says they disagree. It should not silently pick one.

``superseded`` and ``corrected`` cases need judgement about what the article
says, so they come later with model-drafted, human-checked labels.

Usage::

    just eval-cases --model openai/gpt-6-luna   # snapshots grown with that model

Writes ``{output_dir}/eval/profile_update_cases.jsonl``.
"""

import argparse
import json
import os
import random
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.evals.datasets import assign_split
from src.evals.profile_replay import _slug_model
from src.logging_config import console

# Profile lengths to test at: after this many articles (when the replay got there)
DEFAULT_STEPS: Tuple[int, ...] = (1, 3, 5, 10, 20, 30, 39)
IRRELEVANT_PER_STEP = 1
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
_CITATION = re.compile(r"\^\[([^\]]+)\]")
DISPUTED_SUFFIX = "-disputed"


def load_snapshots(entity_dir: str) -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Return (manifest, {step: snapshot}) for one entity's replay folder."""
    with open(os.path.join(entity_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    steps = {}
    for name in os.listdir(entity_dir):
        match = re.match(r"step_(\d+)\.json$", name)
        if match:
            with open(os.path.join(entity_dir, name), encoding="utf-8") as f:
                steps[int(match.group(1))] = json.load(f)
    return manifest, steps


def cited_years(profile_text: str) -> Dict[str, List[str]]:
    """Map each year in the profile to the article ids cited in its sentence."""
    years: Dict[str, List[str]] = {}
    for sentence in re.split(r"(?<=[.!?\]])\s+(?=[A-Z#])|\n+", profile_text):
        ids = _CITATION.findall(sentence)
        if not ids:
            continue
        for year in _YEAR.findall(_CITATION.sub("", sentence)):
            years.setdefault(year, [])
            years[year].extend(i for i in ids if i not in years[year])
    return years


def make_disputed_article(
    article_text: str, profile_text: str, rng: random.Random
) -> Optional[Dict[str, Any]]:
    """Change a year that the article shares with a cited profile sentence.

    Only years that appear exactly once in the article are used, so the change
    hits one claim (like a reporting error) instead of shifting every date in
    the story. Returns None when the article has no such year.
    """
    article_years = _YEAR.findall(article_text)
    once = {y for y in article_years if article_years.count(y) == 1}
    shared = sorted(set(cited_years(profile_text)) & once)
    if not shared:
        return None
    year = rng.choice(shared)
    new_year = str(int(year) + rng.choice((-1, 1)))
    return {
        "text": re.sub(rf"\b{year}\b", new_year, article_text),
        "original_value": year,
        "disputed_value": new_year,
        "original_sources": cited_years(profile_text)[year],
    }


def _case(
    kind: str,
    manifest: Dict[str, Any],
    snapshot: Dict[str, Any],
    article_id: str,
    article_text: str,
    expected: Dict[str, Any],
) -> Dict[str, Any]:
    entity = manifest["entity"]
    case_id = f"{kind}:{entity['name']}:{snapshot['step']}:{article_id}"
    return {
        "case_id": case_id,
        "kind": kind,
        "entity_name": entity["name"],
        "entity_type": entity["entity_type"],
        "snapshot_model": manifest["model"],
        "step": snapshot["step"],
        "profile": snapshot["profile"],
        "article_id": article_id,
        "article_text": article_text,
        "expected": expected,
        # Split by entity, so a tuned prompt never sees test entities at all
        "split": assign_split(entity["name"]),
    }


def build_cases(
    manifest: Dict[str, Any],
    steps: Dict[int, Dict[str, Any]],
    texts: Dict[str, str],
    at_steps: Tuple[int, ...] = DEFAULT_STEPS,
    seed: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Yield the cases for one entity's snapshots."""
    rng = random.Random(f"{seed}:{manifest['entity']['name']}")
    chosen = manifest["articles"]
    passing = [a for a in manifest.get("passing_mentions", []) if a["id"] in texts]
    for k in at_steps:
        snapshot = steps.get(k)
        if snapshot is None:
            continue
        profile_text = snapshot["profile"].get("text", "")

        for article in rng.sample(passing, min(IRRELEVANT_PER_STEP, len(passing))):
            yield _case(
                "irrelevant",
                manifest,
                snapshot,
                article["id"],
                texts[article["id"]],
                {"should_change": False},
            )

        if k >= len(chosen) or chosen[k]["id"] not in texts:
            continue  # no next article
        next_id = chosen[k]["id"]
        yield _case(
            "new_fact",
            manifest,
            snapshot,
            next_id,
            texts[next_id],
            {"must_cite": next_id},
        )

        disputed = make_disputed_article(texts[next_id], profile_text, rng)
        if disputed:
            yield _case(
                "disputed",
                manifest,
                snapshot,
                next_id + DISPUTED_SUFFIX,
                disputed.pop("text"),
                {"keep_both": True, **disputed},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="model the snapshots used")
    parser.add_argument("--domain", default="guantanamo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-dir", default=None, help="default: {output_dir}/eval")
    args = parser.parse_args()

    domain_cfg = DomainConfig(args.domain)
    eval_dir = args.eval_dir or os.path.join(domain_cfg.get_output_dir(), "eval")
    root = os.path.join(eval_dir, "profile_replay", _slug_model(args.model))
    articles = pq.read_table(
        domain_cfg.get_data_path(), columns=["id", "content"]
    ).to_pylist()
    texts = {a["id"]: a["content"] or "" for a in articles}

    cases: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        entity_dir = os.path.join(root, name)
        if not os.path.exists(os.path.join(entity_dir, "manifest.json")):
            continue
        manifest, steps = load_snapshots(entity_dir)
        built = list(build_cases(manifest, steps, texts, seed=args.seed))
        console.print(f"{manifest['entity']['name']}: {len(built)} cases")
        cases.extend(built)

    out = os.path.join(eval_dir, "profile_update_cases.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    by_kind: Dict[str, int] = {}
    for case in cases:
        by_kind[case["kind"]] = by_kind.get(case["kind"], 0) + 1
    console.print(f"{len(cases)} cases {by_kind} -> {out}")


if __name__ == "__main__":
    main()
