#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
"""Generate seed merge candidates for annotation from entity Parquet data."""

import argparse
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate seed merge candidates for the eval harness"
    )
    parser.add_argument(
        "--domain",
        default="guantanamo",
        help="Domain name (default: guantanamo)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Maximum number of candidate pairs to generate (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from src.eval.seed import seed_and_save

    count, path = seed_and_save(
        domain=args.domain,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    print(f"Generated {count} merge candidates -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
