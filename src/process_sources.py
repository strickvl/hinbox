"""Command-line import for evidence corpus manifests."""

import argparse
from pathlib import Path
from typing import Sequence

from src.research.ingestion import CorpusIngestor
from src.research.service import ResearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a research corpus manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--managed-root", type=Path, required=True)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Enforce local-only processing (ingestion itself makes no model calls)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = ResearchService.open(args.database)
    result = CorpusIngestor(service, args.managed_root).ingest_manifest(args.manifest)
    print(
        f"Imported {result.imported_representations} representation(s) "
        f"from {result.imported_sources} source record(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
