"""Build profile-update test cases from the snapshots ``just eval-replay`` saved.

Each case is "this existing profile + this article -> what should a good update
do?". Three kinds can be built without a model or a human, because we know the
right behaviour from how the case was made:

- ``irrelevant``: the profile after step k, plus an article that names the
  entity only once in passing. A good update keeps everything and changes
  little. One mention can still carry a small new fact ("X was released"), so
  adding a sentence or two is allowed; the case is scored on what was lost and
  how much was rewritten. Articles from no later than the profile's latest
  article are preferred, so the case doesn't jump years ahead.
- ``new_fact``: the profile after step k, plus the real next article. A good
  update adds what is new, cites that article, and keeps everything else.
- ``disputed``: the profile after step k, plus a copy of the next article with
  one year changed (2002 -> 2003) where the profile already states that year
  with a citation. A good update keeps both years, each with its own source,
  and says they disagree. It should not silently pick one. The year must
  sit in sentences that share words in both texts, but whether they make the
  same claim still needs review. When both write a month or day next to the
  year, those must match too ("Oct. 12, 2000" is not "January 2000").

Cases marked ``needs_review`` are not scored until their label is confirmed.

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
MIN_SHARED_WORDS = 2
# Hashing 10 entity names into splits put 5 in test and left 3 to tune on, so
# the replay entities get fixed splits: every entity type is in train, and
# dev/test each hold two different types. Other entities fall back to hashing.
ENTITY_SPLITS: Dict[str, str] = {
    "Abd al Rahim al Nashiri": "train",
    "James Pohl": "train",
    "Abu Zubaydah": "train",
    "U.S. Southern Command": "train",
    "Camp Justice": "train",
    "Guantánamo hunger strike": "train",
    "International Committee of the Red Cross": "dev",
    "USS Cole bombing": "dev",
    "Vance Spath": "test",
    "Camp 7": "test",
}


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


# A period after these isn't a sentence end ("Rear Adm. Harris", "U.S.",
# "George W. Bush", "Oct. 12"), so pieces ending in one are joined again
_ABBREVIATION = re.compile(
    r"(?:\b[A-Z]|\b(?:Adm|Gen|Lt|Col|Capt|Cmdr|Sgt|Maj|Mr|Mrs|Ms|Dr|St|Sen|Rep"
    r"|Gov|Jr|Sr|No|Inc|Corp|Co|vs|Jan|Feb|Mar|Apr|Aug|Sept|Sep|Oct|Nov|Dec))\.$"
)
_MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)


def _sentences(text: str) -> List[str]:
    """Split on newlines, then where . ! ? or a citation's "]" is followed by a
    capital or header, without splitting after common abbreviations."""
    sentences: List[str] = []
    for line in text.split("\n"):
        pieces = re.split(r"(?<=[.!?\]])\s+(?=[A-Z#])", line)
        merged: List[str] = []
        for piece in pieces:
            if merged and _ABBREVIATION.search(merged[-1]):
                merged[-1] += " " + piece
            else:
                merged.append(piece)
        sentences.extend(m for m in merged if m.strip())
    return sentences


def _date_parts(sentence: str, year: str) -> Tuple[Optional[str], Optional[str]]:
    """Month (3 letters) and day written just before ``year``, if any."""
    match = re.search(
        rf"\b([A-Z][a-z]{{2,8}})\.?\s+(?:(\d{{1,2}}),?\s+)?{year}\b", sentence
    )
    if not match or match.group(1)[:3].lower() not in _MONTHS:
        return None, None
    return match.group(1)[:3].lower(), match.group(2)


def _same_date(a: str, b: str, year: str) -> bool:
    """False when both sentences give a month (or day) for the year and they
    differ: "Oct. 12, 2000" and "January 2000" are different events."""
    (month_a, day_a), (month_b, day_b) = _date_parts(a, year), _date_parts(b, year)
    if month_a and month_b and month_a != month_b:
        return False
    return not (day_a and day_b and day_a != day_b)


def _content_words(text: str) -> set:
    """Lower-cased words of 5+ letters: enough to tell what a sentence is about."""
    return set(re.findall(r"[a-z]{5,}", _CITATION.sub("", text).lower()))


def cited_years(profile_text: str) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Map each year in the profile to the cited sentences that state it."""
    years: Dict[str, List[Tuple[str, List[str]]]] = {}
    for sentence in _sentences(profile_text):
        ids = [i.strip() for g in _CITATION.findall(sentence) for i in g.split(",")]
        if ids:
            for year in set(_YEAR.findall(_CITATION.sub("", sentence))):
                years.setdefault(year, []).append((sentence, ids))
    return years


def make_disputed_article(
    article_text: str,
    profile_text: str,
    rng: random.Random,
    min_shared_words: int = MIN_SHARED_WORDS,
) -> Optional[Dict[str, Any]]:
    """Change a year that the article and a cited profile sentence both state.

    Only years that appear exactly once in the article are used, so the change
    hits one claim (like a reporting error) instead of shifting every date in
    the story. The article sentence and the profile sentence must also share
    ``min_shared_words`` content words, so "captured in 2002" isn't paired with
    "charged in 2002". That is a heuristic, so these cases are marked for
    review. Returns None when no year qualifies.
    """
    article_years = _YEAR.findall(article_text)
    profile_years = cited_years(profile_text)
    candidates = []
    for year in sorted(y for y in set(article_years) if article_years.count(y) == 1):
        article_sentence = next(
            t for t in _sentences(article_text) if re.search(rf"\b{year}\b", t)
        )
        for sentence, ids in profile_years.get(year, []):
            shared = _content_words(article_sentence) & _content_words(sentence)
            if len(shared) >= min_shared_words and _same_date(
                article_sentence, sentence, year
            ):
                candidates.append((year, article_sentence, sentence, ids))
    if not candidates:
        return None
    year, article_sentence, profile_sentence, ids = rng.choice(candidates)
    new_year = str(int(year) + rng.choice((-1, 1)))
    return {
        "text": re.sub(rf"\b{year}\b", new_year, article_text),
        "original_value": year,
        "disputed_value": new_year,
        "original_sources": ids,
        # Shown to the reviewer, who confirms both sentences make the same claim
        "article_sentence": re.sub(rf"\b{year}\b", new_year, article_sentence),
        "profile_sentence": profile_sentence,
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
        # Disputed labels come from a word-overlap heuristic, so they stay out
        # of scoring until a person (or a checked model) confirms them
        "needs_review": kind == "disputed",
        # Split by entity, so a tuned prompt never sees test entities at all
        "split": ENTITY_SPLITS.get(entity["name"]) or assign_split(entity["name"]),
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

        # Prefer passing mentions from no later than the profile's own timeline,
        # so the case doesn't jump years ahead of what the profile knows
        latest = chosen[k - 1].get("published_date") if k <= len(chosen) else None
        in_time = [
            a
            for a in passing
            if latest is not None
            and a.get("published_date") is not None
            and a["published_date"] <= latest
        ]
        pool = in_time or passing
        for article in rng.sample(pool, min(IRRELEVANT_PER_STEP, len(pool))):
            yield _case(
                "irrelevant",
                manifest,
                snapshot,
                article["id"],
                texts[article["id"]],
                # Scored on loss and churn only: one mention can still carry a
                # small new fact, and adding it is fine
                {"keep_all": True, "max_new_sentences": 2},
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
