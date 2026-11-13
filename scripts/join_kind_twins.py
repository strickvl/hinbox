"""Join places and organizations a finished run saved twice (``src/engine/kind_twins.py``).

    uv run python -m scripts.join_kind_twins --run <entities dir> --dry-run
    uv run python -m scripts.join_kind_twins --run <entities dir> --local [--hubs-only]

``--hubs-only`` joins only countries, regions and news outlets, which are
linked without a model call.

Works on the run folder in place, so point it at a copy. The tables are saved
after each pair, so a stopped run loses at most one pair, and running the same
command again carries on (joined pairs are gone; a pair whose update kept
failing keeps the articles already fed in and is asked for the rest).
Afterwards, run the pipeline on the folder (``HINBOX_OUTPUT_DIR=<dir> just
process --local --relevance-check --limit 0``): it reads no articles and only
checks the citations of the changed profiles.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Set, Tuple

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.engine import ArticleProcessor
from src.engine.corrections_pass import configure_corrections
from src.engine.hub_profiles import is_hub
from src.engine.kind_twins import TWIN_TABLES, donor_articles, find_twins, join_pair
from src.engine.merge_evidence import batch_embed_texts
from src.engine.profile_view import embedding_text
from src.logging_config import log
from src.process_and_extract import load_and_validate_articles
from src.utils.corrections import corrections_path, load_corrections
from src.utils.embeddings.similarity import (
    ensure_local_embeddings_available,
    get_embedding_manager,
    reset_embedding_manager_cache,
)
from src.utils.entity_keys import plain_kind
from src.utils.file_ops import write_entities_table
from src.utils.letters_page import stored_letters_text
from src.utils.llm import configure_llm_concurrency
from src.utils.processing_status import ProcessingStatus
from src.utils.task_models import check_task_models


def _article_texts(domain: str, path: str, run: str) -> Dict[str, str]:
    processor, status = ArticleProcessor(domain=domain), ProcessingStatus(run)
    texts = {}
    for i, row in enumerate(load_and_validate_articles(path) or [], 1):
        info = processor.prepare_article_info(row, i)
        texts[info["id"]] = stored_letters_text(
            info["content"], status.get_metadata(info["id"])
        )
    return texts


def _ident(row: Dict[str, Any]) -> Tuple[str, str]:
    return (row.get("name") or "", str(plain_kind(row.get("type"))))


def _embed(row: Dict[str, Any]) -> None:
    manager = get_embedding_manager()
    text = embedding_text(row["profile"]["text"])
    vecs, model, dim, fp = batch_embed_texts(manager, [text], 1, "embed.profile")
    row.update(
        profile_embedding=vecs[0],
        profile_embedding_model=model,
        profile_embedding_dim=dim or len(vecs[0] or []),
        profile_embedding_fingerprint=fp,
    )


def _pairs(
    table: str, rows: List[Dict[str, Any]], domain: str, hubs_only: bool
) -> List[Tuple[int, List[int], bool]]:
    """(keeper, donors, link only) for each pair, on the rows as they are now."""
    out = []
    for keeper, donors in find_twins(rows):
        hub = is_hub(domain, table, plain_kind(rows[keeper].get("type")))
        if hub or not hubs_only:
            out.append((keeper, donors, hub))
    return out


def _plan(table: str, rows: List[Dict[str, Any]], domain: str, hubs: bool) -> None:
    for keeper, donors, hub in _pairs(table, rows, domain, hubs):
        calls = sum(len(donor_articles(rows[keeper], rows[d])) for d in donors)
        how = "linked only, no model call" if hub else "fed in"
        log(
            f"{table}: '{rows[keeper]['name']}' keeps 1 of {len(donors) + 1}, "
            f"{calls} articles {how}",
            level="info",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="entities folder (a copy)")
    ap.add_argument("--domain", default="guantanamo")
    ap.add_argument("--articles", help="articles Parquet (default: the domain's)")
    ap.add_argument("--local", action="store_true", help="local models only")
    ap.add_argument("--dry-run", action="store_true", help="list pairs, no calls")
    ap.add_argument("--limit", type=int, default=0, help="join at most N pairs")
    ap.add_argument(
        "--hubs-only",
        action="store_true",
        help="only countries, regions and outlets (no model calls)",
    )
    args = ap.parse_args()

    model_type = "ollama" if args.local else "gemini"
    if args.local:
        os.environ["EMBEDDING_MODE"] = "local"
        reset_embedding_manager_cache()
        ensure_local_embeddings_available()
    check_task_models(model_type)
    config = DomainConfig(args.domain)
    cc = config.get_concurrency_config()
    configure_llm_concurrency(
        cloud_in_flight=cc["llm_in_flight"], local_in_flight=cc["ollama_in_flight"]
    )
    paths = {t: os.path.join(args.run, f"{t}.parquet") for t in TWIN_TABLES}
    tables = {t: pq.read_table(p).to_pylist() for t, p in paths.items()}
    if args.dry_run:
        for table, rows in tables.items():
            _plan(table, rows, args.domain, args.hubs_only)
        return 0

    texts = _article_texts(
        args.domain, args.articles or config.get_data_path(), args.run
    )
    # The edit guard keeps a historian's strikes and sentences, as in a run
    configure_corrections(
        load_corrections(corrections_path(args.domain, config.load_config()))
    )
    joined = failed = 0
    for table in TWIN_TABLES:
        tried: Set[Tuple[str, str]] = set()
        while not args.limit or joined + failed < args.limit:
            rows = tables[table]  # indexes change after each join: ask again
            todo = [
                p
                for p in _pairs(table, rows, args.domain, args.hubs_only)
                if _ident(rows[p[0]]) not in tried
            ]
            if not todo:
                break
            keeper, donors, _ = todo[0]
            tried.add(_ident(rows[keeper]))
            work, drop, fed = rows[keeper], [], 0
            for d in donors:
                work, report = join_pair(
                    table, work, rows[d], texts, model_type, args.domain
                )
                fed += report["updated"]
                if report.get("failed"):
                    log(
                        f"{table}: '{work['name']}' not joined yet: "
                        f"{report['failed']}; {report['updated']} articles "
                        "fed in are kept",
                        level="warning",
                    )
                    break
                drop.append(d)
                log(
                    f"{table}: joined '{work['name']}' "
                    f"({report['updated']} articles fed in)",
                    level="success",
                )
            if drop or fed:  # save what was fed in, joined or not
                _embed(work)
                kept = [work if i == keeper else r for i, r in enumerate(rows)]
                tables[table] = [r for i, r in enumerate(kept) if i not in drop]
                write_entities_table(table, tables[table], args.run)
            if len(drop) < len(donors):
                failed += 1
            else:
                joined += 1
    log(
        f"Joined {joined} pairs, {failed} not joined (run again to retry)",
        level="success" if not failed else "warning",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
