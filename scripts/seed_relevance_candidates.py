#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
"""Sample articles from the domain's parquet file for relevance annotation."""

import argparse
import random
from typing import List, Optional

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.eval.relevance_candidates import (
    RelevanceCandidate,
    save_relevance_candidates,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample articles for relevance annotation"
    )
    parser.add_argument(
        "--domain",
        default="guantanamo",
        help="Domain name (default: guantanamo)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=300,
        help="Maximum number of articles to sample (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def seed_relevance_candidates(
    domain: str,
    max_articles: int = 300,
    seed: int = 42,
) -> List[RelevanceCandidate]:
    """Load articles, filter empties, sample, and build candidates."""
    config = DomainConfig(domain)
    data_path = config.get_data_path()

    table = pq.read_table(data_path)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} articles from {data_path}")

    # Filter out articles with empty content
    rows = [r for r in rows if r.get("content") and r["content"].strip()]
    print(f"{len(rows)} articles after filtering empties")

    # Sample
    rng = random.Random(seed)
    if len(rows) > max_articles:
        rows = rng.sample(rows, max_articles)
    else:
        rng.shuffle(rows)
    print(f"Sampled {len(rows)} articles")

    candidates: List[RelevanceCandidate] = []
    for row in rows:
        content = row.get("content", "")
        snippet = content[:600] if content else ""
        pub_date = row.get("published_date")
        if pub_date is not None:
            pub_date = str(pub_date)

        candidates.append(
            RelevanceCandidate(
                article_id=str(row["id"]),
                title=row.get("title", "(no title)"),
                content_snippet=snippet,
                url=row.get("url"),
                published_date=pub_date,
                content_length=len(content),
            )
        )

    return candidates


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    candidates = seed_relevance_candidates(
        domain=args.domain,
        max_articles=args.max_articles,
        seed=args.seed,
    )
    path = save_relevance_candidates(args.domain, candidates)
    print(f"Wrote {len(candidates)} relevance candidates -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
