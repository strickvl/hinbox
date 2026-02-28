#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
"""Score a pipeline trace run's merge decisions against gold labels."""

import argparse
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate merge decisions against gold labels"
    )
    parser.add_argument(
        "--domain",
        default="guantanamo",
        help="Domain name (default: guantanamo)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Trace run ID to evaluate (defaults to latest run)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    from src.eval.scorer import score_run

    result = score_run(domain=args.domain, run_id=args.run_id)

    if result.total_evaluated == 0:
        print("No merge decisions matched gold labels.")
        print(f"  Skipped (no gold label): {result.skipped_no_gold}")
        print(f"  Skipped (unsure): {result.skipped_unsure}")
        return 0

    print(f"Merge Evaluation Results ({args.domain})")
    print("=" * 45)
    print(f"  True Positives:   {result.true_positives}")
    print(f"  False Positives:  {result.false_positives}")
    print(f"  True Negatives:   {result.true_negatives}")
    print(f"  False Negatives:  {result.false_negatives}")
    print(f"  Skipped (unsure): {result.skipped_unsure}")
    print(f"  Skipped (no gold):{result.skipped_no_gold}")
    print("-" * 45)
    print(f"  Precision: {result.precision:.3f}")
    print(f"  Recall:    {result.recall:.3f}")
    print(f"  F1:        {result.f1:.3f}")
    print(f"  Total evaluated:  {result.total_evaluated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
