"""Grow long entity profiles by replaying an entity's articles in date order.

The profile-update eval needs profiles at many lengths (after 1, 5, 10, 20, 40
articles), but the entity database has nothing longer than three articles. The
raw corpus does: Nashiri is named in over 200 articles. So for a handful of
well-covered entities we create a profile from the first article that is
mostly about them, then run the pipeline's own ``update_profile`` on each later
one, saving the profile after every step.

These snapshots are realistic starting points for test cases, not gold answers.

Usage::

    just eval-replay --model openai/gpt-6-luna                 # default entities
    just eval-replay --model openai/gpt-6-luna --max-articles 10 --only Nashiri
    just eval-replay --list                                    # article counts only

Output, under ``{output_dir}/eval/profile_replay/<model>/<entity>/``:

- ``manifest.json``: the entity, its match terms, the articles chosen (in order)
  and the "passing mention" articles (named once), kept for irrelevant-article
  test cases.
- ``step_001.json`` ...: the profile after each step, with the article that
  produced it. A run that stops part-way resumes from the last saved step.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.constants import CLOUD_MODEL
from src.logging_config import console


@dataclass(frozen=True)
class ReplayEntity:
    """An entity to replay, and the strings that mark an article as about it."""

    name: str
    entity_type: str  # person / organization / location / event (singular)
    terms: Tuple[str, ...]


# Ten well-covered entities across the four types. Article counts are from the
# Miami Herald corpus (1,620 articles); "substantive" = a term appears 2+ times.
DEFAULT_ENTITIES: Tuple[ReplayEntity, ...] = (
    ReplayEntity("Abd al Rahim al Nashiri", "person", ("Nashiri",)),
    ReplayEntity("James Pohl", "person", ("Pohl",)),
    ReplayEntity("Vance Spath", "person", ("Spath",)),
    ReplayEntity("Abu Zubaydah", "person", ("Abu Zubaydah", "Zubaydah")),
    ReplayEntity(
        "International Committee of the Red Cross",
        "organization",
        ("Red Cross", "ICRC"),
    ),
    ReplayEntity("U.S. Southern Command", "organization", ("Southern Command",)),
    ReplayEntity("Camp 7", "location", ("Camp 7",)),
    ReplayEntity("Camp Justice", "location", ("Camp Justice",)),
    ReplayEntity("USS Cole bombing", "event", ("USS Cole",)),
    ReplayEntity("Guantánamo hunger strike", "event", ("hunger strike",)),
)


def parse_entity_spec(spec: str) -> ReplayEntity:
    """Parse ``"Name|type|term1,term2"`` from the command line."""
    parts = spec.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--entity must look like 'Name|type|term1,term2', got {spec!r}"
        )
    name, entity_type, terms = (p.strip() for p in parts)
    return ReplayEntity(name, entity_type, tuple(t.strip() for t in terms.split(",")))


def count_mentions(text: str, terms: Tuple[str, ...]) -> int:
    """Case-insensitive count of term occurrences in ``text``.

    Longer terms are tried first and matches don't overlap, so "Abu Zubaydah"
    counts once, not once for itself and again for "Zubaydah".
    """
    ordered = sorted(terms, key=len, reverse=True)
    pattern = "|".join(re.escape(term) for term in ordered)
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def select_articles(
    articles: List[Dict[str, Any]],
    entity: ReplayEntity,
    max_articles: int,
    min_mentions: int = 2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split the corpus into (substantive, passing) articles for ``entity``.

    Substantive articles name the entity ``min_mentions`` times or more, or in
    the title; they are returned oldest first, capped at ``max_articles``.
    Passing-mention articles name it once (kept for "irrelevant article" test
    cases). Articles that never name it are ignored.
    """
    substantive, passing = [], []
    for article in articles:
        content = article.get("content") or ""
        mentions = count_mentions(content, entity.terms)
        in_title = count_mentions(article.get("title") or "", entity.terms) > 0
        row = {
            "id": article["id"],
            "title": article.get("title") or "",
            "published_date": article.get("published_date"),
            "mentions": mentions,
        }
        if mentions >= min_mentions or (in_title and mentions):
            substantive.append(row)
        elif mentions == 1:
            passing.append(row)

    def by_date(row: Dict[str, Any]) -> Tuple[bool, Any, str]:
        # Undated articles go last; the flag keeps None from being compared
        # with a date of another type (int timestamp, string, datetime)
        date = row["published_date"]
        return date is None, date if date is not None else 0, row["id"]

    return sorted(substantive, key=by_date)[:max_articles], sorted(passing, key=by_date)


def entity_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _step_path(entity_dir: str, step: int) -> str:
    return os.path.join(entity_dir, f"step_{step:03d}.json")


def load_last_step(entity_dir: str) -> Optional[Dict[str, Any]]:
    """Return the highest-numbered saved step, or None."""
    if not os.path.isdir(entity_dir):
        return None
    steps = sorted(f for f in os.listdir(entity_dir) if re.match(r"step_\d+\.json$", f))
    if not steps:
        return None
    with open(os.path.join(entity_dir, steps[-1]), encoding="utf-8") as f:
        return json.load(f)


def check_resumable(entity_dir: str, chosen: List[Dict[str, Any]]) -> None:
    """Refuse to resume when saved steps came from a different article order.

    Changing ``--min-mentions``, the match terms, or the corpus can change which
    articles are chosen. Resuming by step number would then build on a profile
    from the old order while the manifest claims the new one. Raising a longer
    or shorter ``--max-articles`` keeps the order, so it is allowed.
    """
    if not os.path.isdir(entity_dir):
        return
    for name in sorted(os.listdir(entity_dir)):
        match = re.match(r"step_(\d+)\.json$", name)
        if not match:
            continue
        step = int(match.group(1))
        if step > len(chosen):
            continue
        with open(os.path.join(entity_dir, name), encoding="utf-8") as f:
            saved_id = json.load(f)["article"]["id"]
        if saved_id != chosen[step - 1]["id"]:
            raise ValueError(
                f"{entity_dir} holds step {step} for article {saved_id}, but this "
                f"run would use {chosen[step - 1]['id']}. Use another --eval-dir "
                "or delete the old snapshots."
            )


def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)  # a crash never leaves a half-written step


def replay_entity(
    entity: ReplayEntity,
    chosen: List[Dict[str, Any]],
    texts: Dict[str, str],
    entity_dir: str,
    model: str,
    domain: str,
) -> int:
    """Create then update the profile one article at a time. Returns steps done."""
    from src.engine.profiles import VersionedProfile, create_profile, update_profile

    check_resumable(entity_dir, chosen)
    last = load_last_step(entity_dir)
    done = last["step"] if last else 0
    profile = last["profile"] if last else None
    versioned = VersionedProfile(**last["profile_versions"]) if last else None
    kwargs = {"domain": domain, "model": model}

    for step, article in enumerate(chosen, start=1):
        if step <= done:
            continue
        start = time.perf_counter()
        text = texts[article["id"]]
        previous_text = (profile or {}).get("text")
        if profile is None:
            profile, versioned, history = create_profile(
                entity.entity_type, entity.name, text, article["id"], **kwargs
            )
        else:
            profile, versioned, history = update_profile(
                entity.entity_type,
                entity.name,
                profile,
                versioned,
                text,
                article["id"],
                **kwargs,
            )
        _write_json(
            _step_path(entity_dir, step),
            {
                "step": step,
                "entity": asdict(entity),
                "model": model,
                "article": article,
                "profile": profile,
                # update_profile keeps the old profile when the model's answer
                # can't be parsed, so identical text usually means a failed
                # update. (A failed API call raises instead and stops the entity.)
                "unchanged": profile.get("text") == previous_text,
                "profile_versions": versioned.model_dump(mode="json"),
                "reflection_history": history,
                "seconds": round(time.perf_counter() - start, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        console.print(
            f"[dim]{entity.name}[/] step {step}/{len(chosen)}: "
            f"{len(profile.get('text', ''))} chars"
        )
        done = step
    return done


def _slug_model(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=CLOUD_MODEL, help="e.g. openai/gpt-6-luna")
    parser.add_argument("--domain", default="guantanamo")
    parser.add_argument("--max-articles", type=int, default=40)
    parser.add_argument("--min-mentions", type=int, default=2)
    parser.add_argument(
        "--entity",
        type=parse_entity_spec,
        action="append",
        help="'Name|type|term1,term2' (repeatable); replaces the default list",
    )
    parser.add_argument(
        "--only", default=None, help="only entities whose name contains this"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="entities replayed in parallel"
    )
    parser.add_argument(
        "--list", action="store_true", help="print article counts, no model calls"
    )
    parser.add_argument("--eval-dir", default=None, help="default: {output_dir}/eval")
    args = parser.parse_args()

    domain_cfg = DomainConfig(args.domain)
    entities = args.entity or list(DEFAULT_ENTITIES)
    if args.only:
        entities = [e for e in entities if args.only.lower() in e.name.lower()]
    columns = ["id", "title", "content", "published_date"]
    articles = pq.read_table(domain_cfg.get_data_path(), columns=columns).to_pylist()
    texts = {a["id"]: a["content"] or "" for a in articles}

    eval_dir = args.eval_dir or os.path.join(domain_cfg.get_output_dir(), "eval")
    root = os.path.join(eval_dir, "profile_replay", _slug_model(args.model))

    plans = []
    for entity in entities:
        chosen, passing = select_articles(
            articles, entity, args.max_articles, args.min_mentions
        )
        console.print(
            f"{entity.name} ({entity.entity_type}): {len(chosen)} articles to replay, "
            f"{len(passing)} passing mentions"
        )
        plans.append((entity, chosen, passing))
    if args.list:
        return

    def run(plan: Tuple[ReplayEntity, List[Dict[str, Any]], List[Dict[str, Any]]]):
        entity, chosen, passing = plan
        entity_dir = os.path.join(root, entity_slug(entity.name))
        try:
            check_resumable(entity_dir, chosen)  # before the manifest is replaced
        except ValueError as e:
            return entity.name, 0, e
        os.makedirs(entity_dir, exist_ok=True)
        _write_json(
            os.path.join(entity_dir, "manifest.json"),
            {
                "entity": asdict(entity),
                "model": args.model,
                "domain": args.domain,
                "min_mentions": args.min_mentions,
                "articles": chosen,
                "passing_mentions": passing,
            },
        )
        try:
            return (
                entity.name,
                replay_entity(
                    entity, chosen, texts, entity_dir, args.model, args.domain
                ),
                None,
            )
        except Exception as e:  # noqa: BLE001 — one entity failing must not stop the rest
            return entity.name, (load_last_step(entity_dir) or {}).get("step", 0), e

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run, plans))
    for name, steps, error in results:
        status = f"[red]stopped: {error}[/]" if error else "[green]done[/]"
        console.print(f"{name}: {steps} steps saved, {status}")
    console.print(f"Snapshots in {root}")
    if any(error for _, _, error in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
