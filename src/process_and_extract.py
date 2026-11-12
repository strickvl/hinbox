#!/usr/bin/env python3
"""Main article processing and entity extraction pipeline.

This script implements the core processing pipeline that reads articles from the Miami Herald
dataset, extracts entities (people, events, locations, organizations) using either cloud-based
(Gemini) or local (Ollama) language models, and merges the results with existing entity data.
The pipeline includes relevance checking, entity deduplication, and comprehensive processing
status tracking.

Concurrency model (Phase 2 speed audit):
  - Multiple extraction workers process articles in parallel via ThreadPoolExecutor.
  - Within each article, the 4 entity-type extractions also run concurrently.
  - A shared LLM semaphore bounds cloud API concurrency.
  - A single merge actor (the main thread) consumes extraction results in article
    order and is the *only* writer to the shared ``entities`` dict and
    ``ProcessingStatus`` sidecar, so no locking is needed.
"""

import argparse
import contextvars
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.engine import ArticleProcessor, EntityMerger, configure_match_check_memo
from src.engine.corrections_merges import carry_out_merges
from src.engine.corrections_pass import (
    configure_corrections,
    describe_report,
    run_corrections_pass,
    write_report,
)
from src.engine.grounding_pass import run_profile_grounding_postprocess
from src.exceptions import ArticleLoadError
from src.logging_config import console, get_logger, log, set_show_profiles, set_verbose
from src.utils.article_window import ordered_window
from src.utils.cache_utils import sha256_text
from src.utils.corrections import (
    DONE,
    corrections_path,
    last_states,
    load_corrections,
)
from src.utils.embeddings.similarity import (
    ensure_local_embeddings_available,
    reset_embedding_manager_cache,
)
from src.utils.extraction import configure_extraction_sidecar_cache
from src.utils.file_ops import load_existing_entities, write_entities_table
from src.utils.grounding import describe_claim_counts
from src.utils.letters_page import split_letters, stored_letters_text
from src.utils.llm import configure_llm_concurrency
from src.utils.processing_status import ProcessingStatus
from src.utils.run_summary import log_processing_summary
from src.utils.server_health import ModelServerUnreachable, unreachable
from src.utils.step_timing import (
    save_timing_report,
    timed_step,
    write_partial_report,
)
from src.utils.task_models import check_task_models
from src.utils.tracing import (
    article_scope,
    configure_tracing,
    run_in_span,
    start_article_span,
)

# Get module-specific logger
logger = get_logger("process_and_extract")


# ---------------------------------------------------------------------------
# Data structures for producer / consumer pipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArticleExtractionResult:
    """Immutable result produced by an extraction worker.

    Contains everything the merge actor needs to apply the result without
    re-reading the article.
    """

    row_index: int
    row: Dict[str, Any]
    article_info: Dict[str, str]
    processing_metadata: Dict[str, Any]
    extraction_timestamp: str
    extracted_entities: Dict[str, List[Dict[str, Any]]]
    was_extracted: bool
    skip_reason: Optional[str] = None


def ensure_dir(directory: str) -> None:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory to ensure exists

    Note:
        Creates parent directories as needed. No-op if directory is empty string.
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def write_entities_to_files(entities: Dict[str, Dict], base_dir: str) -> None:
    """Write entities to their respective Parquet files.

    Writes each entity type as a single atomic Parquet file (one write per
    type, 4 total) rather than per-entity read-modify-write cycles.

    Args:
        entities: Dictionary of entity types containing their entity dictionaries
        base_dir: Domain-specific output directory
    """
    for entity_type, entity_dict in entities.items():
        write_entities_table(entity_type, list(entity_dict.values()), base_dir)


def setup_arguments_and_config() -> argparse.Namespace:
    """Setup command line arguments and configuration for the processing pipeline.

    Defines and parses all command line arguments used by the processing pipeline,
    including model selection, processing limits, relevance checking, and various
    processing options.

    Returns:
        Parsed command line arguments namespace with configured options

    Note:
        Also configures verbose logging if requested via --verbose flag.
        Logs the final argument configuration for debugging purposes.
    """
    parser = argparse.ArgumentParser(
        description="Process articles from a Parquet file and extract entities."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local approach (Ollama/spaCy) rather than Gemini or cloud models",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of articles to process (default: 5)",
    )
    parser.add_argument(
        "--relevance-check",
        action="store_true",
        help="Perform a relevance check on each article before extraction",
    )
    parser.add_argument(
        "--articles-path",
        type=str,
        default=None,
        help="Path to the raw articles Parquet file (defaults to domain config)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Process articles even if they've been processed before",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging for debugging the reflection mechanism",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="guantanamo",
        help="Domain configuration to use (default: guantanamo)",
    )
    parser.add_argument(
        "--show-profiles",
        action="store_true",
        help="Display full Rich profile panels during merge (off by default for compact output)",
    )
    parser.add_argument(
        "--skip-grounding",
        action="store_true",
        help="Don't run the grounding check after merging (the next run checks them)",
    )
    args = parser.parse_args()

    # Configure logger level based on verbosity flag
    if args.verbose:
        set_verbose(True)
        log("Verbose logging enabled")

    if args.show_profiles:
        set_show_profiles(True)

    log(
        f"[bold blue]Arguments:[/] limit={args.limit}, local={args.local}, relevance_check={args.relevance_check}, force_reprocess={args.force_reprocess}"
    )

    # Resolve domain config and ensure output directory exists
    config = DomainConfig(args.domain)
    base_dir = config.get_output_dir()
    if not args.articles_path:
        args.articles_path = config.get_data_path()
    ensure_dir(base_dir)
    return args


def load_and_validate_articles(articles_path: str) -> List[Dict]:
    """Load and validate articles from parquet file."""
    # Check if articles parquet exists
    if not os.path.exists(articles_path):
        log(f"ERROR: Articles file not found at {articles_path}", level="error")
        return []

    # Read entire Parquet into memory
    try:
        table = pq.read_table(articles_path)
        rows = table.to_pylist()
        log(f"Loaded {len(rows)} articles from {articles_path}", level="success")
        return rows
    except Exception as e:
        error = ArticleLoadError(
            f"Failed to read articles from {articles_path}",
            {"file_path": articles_path, "original_error": str(e)},
        )
        log("Failed to read Parquet file", level="error", exception=error)
        return []


def merge_all_entities(
    extracted_entities: Dict[str, List],
    entities: Dict[str, Dict],
    article_info: Dict[str, str],
    extraction_timestamp: str,
    model_type: str,
    processing_metadata: Dict,
    processor: ArticleProcessor,
    domain_config: DomainConfig = None,
) -> None:
    """Merge all extracted entities with existing entities."""
    log("Merging extracted entities...", level="processing")

    # Reuse caller-provided config or construct once for all entity types
    domain_cfg = domain_config or DomainConfig(processor.domain)

    # Convert Pydantic to dict if needed
    people_dicts = processor.convert_pydantic_to_dict(
        extracted_entities.get("people", [])
    )
    org_dicts = processor.convert_pydantic_to_dict(
        extracted_entities.get("organizations", [])
    )
    loc_dicts = processor.convert_pydantic_to_dict(
        extracted_entities.get("locations", [])
    )
    event_dicts = processor.convert_pydantic_to_dict(
        extracted_entities.get("events", [])
    )

    # Merge each entity type using EntityMerger directly
    merge_inputs = [
        ("people", people_dicts),
        ("organizations", org_dicts),
        ("locations", loc_dicts),
        ("events", event_dicts),
    ]

    def merge_one(entity_type: str, entity_dicts: List[Dict]) -> None:
        try:
            merge_start = datetime.now()
            merger = EntityMerger(entity_type)
            with timed_step(f"merge.{entity_type}"):
                merge_stats = merger.merge_entities(
                    entity_dicts,
                    entities,
                    article_info["id"],
                    article_info["title"],
                    article_info["url"],
                    article_info["published_date"],
                    article_info["content"],
                    extraction_timestamp,
                    model_type,
                    domain=processor.domain,
                    domain_config=domain_cfg,
                    article_texts=processor.article_texts,
                )
            merge_duration = (datetime.now() - merge_start).total_seconds()
            log(
                f"Merged {len(entity_dicts)} {entity_type} in {merge_duration:.2f}s "
                f"(new={merge_stats.new} merged={merge_stats.merged} skipped={merge_stats.skipped})",
                level="success",
            )
            if merge_stats.failed_profiles:
                processing_metadata[f"failed_profiles_{entity_type}"] = (
                    merge_stats.failed_profiles
                )
            if merge_stats.failed_checks:
                processing_metadata[f"failed_checks_{entity_type}"] = (
                    merge_stats.failed_checks
                )
        except Exception as e:
            log(f"Error merging {entity_type}", level="error", exception=e)
            processing_metadata[f"error_{entity_type}"] = str(e)

    # Each type's merge reads and writes only entities[entity_type], so the
    # four can run at once; copy_context() keeps their timings nested
    workers = domain_cfg.get_concurrency_config()["merge_parallel_types"]
    with ThreadPoolExecutor(max_workers=min(workers, len(merge_inputs))) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, merge_one, et, dicts)
            for et, dicts in merge_inputs
        ]
        for future in futures:
            future.result()


def incomplete_reasons(processing_metadata: Dict) -> List[str]:
    """What failed while extracting or merging this article, if anything.

    An article with any of these stays unprocessed so the next run tries it
    again; marking it done would skip it, and lose those entities, for good.
    """
    outcomes = processing_metadata.get("phase_outcomes", {})
    if "extraction" not in outcomes:
        return ["extraction"]
    reasons = [
        f"extract_{et}"
        for et, outcome in outcomes["extraction"].items()
        if not outcome.get("success", True)
    ]
    return reasons + [
        key
        for key in processing_metadata
        if key.startswith(("error_", "failed_profiles_", "failed_checks_"))
    ]


def write_results_and_statistics(
    processed_rows: List[Dict],
    entities: Dict[str, Dict],
    args: argparse.Namespace,
    processed_count: int,
    skipped_relevance_count: int,
    skipped_already_processed: int,
    article_count: int,
    base_dir: str,
    status_tracker: ProcessingStatus = None,
    skipped_relevance_error: int = 0,
    skipped_relevance_cached: int = 0,
    skipped_incomplete: int = 0,
) -> None:
    """Write results to files and log comprehensive statistics."""
    save_entities_then_status(entities, base_dir, status_tracker)

    # Log comprehensive summary
    log_processing_summary(
        article_count,
        processed_count,
        skipped_relevance_count,
        skipped_already_processed,
        entities,
        processed_rows,
        skipped_relevance_error=skipped_relevance_error,
        skipped_relevance_cached=skipped_relevance_cached,
        skipped_incomplete=skipped_incomplete,
    )


SAVE_EVERY_ENV = "HINBOX_SAVE_EVERY"
SAVE_EVERY_DEFAULT = 25


def save_every_articles() -> int:
    """How many merged articles between saves of the entity tables during a
    run (``HINBOX_SAVE_EVERY``, default 25; 0 saves only at the end or when
    stopped). Before, a run saved only at its end, on Ctrl+C or when the
    model server was gone, so a crash or shutdown lost every merge and
    profile call made so far."""
    try:
        return max(0, int(os.getenv(SAVE_EVERY_ENV, SAVE_EVERY_DEFAULT)))
    except ValueError:
        return SAVE_EVERY_DEFAULT


def save_entities_then_status(
    entities: Dict[str, Dict], base_dir: str, status_tracker: Optional[ProcessingStatus]
) -> None:
    """Write the entity tables, then the status file.

    In this order, a failed entity write leaves the articles unprocessed on
    disk, so the next run redoes them instead of skipping them.
    """
    log("Saving updated entity tables...", level="processing")
    with timed_step("write"):
        write_entities_to_files(entities, base_dir)
    log("Entity tables successfully saved", level="success")
    if status_tracker:
        status_tracker.flush()


def extract_single_article_only(
    row: Dict,
    row_index: int,
    processor: ArticleProcessor,
    args: argparse.Namespace,
    *,
    status_snapshot: Dict[str, Dict[str, Any]],
    skip_if_unchanged: bool = True,
    extract_per_article: int = 1,
) -> ArticleExtractionResult:
    """Run skip checks, relevance, and extraction for one article.

    This function is safe to call from a worker thread: it never mutates
    shared state (``entities``, ``ProcessingStatus``).

    The ``status_snapshot`` contains full metadata per article ID (including
    ``content_hash`` when available) for skip-if-unchanged detection.
    """
    article_info = processor.prepare_article_info(row, row_index)
    processing_metadata = processor.initialize_processing_metadata(row)
    article_id = article_info["id"]
    extraction_timestamp = datetime.now().isoformat()

    empty = ArticleExtractionResult(
        row_index=row_index,
        row=row,
        article_info=article_info,
        processing_metadata=processing_metadata,
        extraction_timestamp=extraction_timestamp,
        extracted_entities={},
        was_extracted=False,
    )

    # Compute content hash for skip-if-unchanged and later persistence
    content_hash = sha256_text(article_info.get("content") or "")
    processing_metadata["content_hash"] = content_hash

    # Skip if already processed (use snapshot — no thread-safety issue)
    prior = status_snapshot.get(article_id, {})
    already_processed = prior.get("processed", False)
    if already_processed and not args.force_reprocess:
        # If skip_if_unchanged is active and we have a stored hash, check it
        prior_hash = prior.get("content_hash")
        if skip_if_unchanged and prior_hash and prior_hash != content_hash:
            # Content changed — fall through to reprocess
            logger.info(
                f"Article {article_id}: content changed since last run, reprocessing"
            )
        else:
            return ArticleExtractionResult(
                **{**empty.__dict__, "skip_reason": "already_processed"}
            )

    if not article_info["content"]:
        return ArticleExtractionResult(
            **{**empty.__dict__, "skip_reason": "no_content"}
        )

    # Relevance check
    if args.relevance_check:
        relevance_fp = processor.relevance_fingerprint()
        processing_metadata["relevance_fingerprint"] = relevance_fp
        # Reuse an earlier "not relevant" verdict while the article text, model
        # and relevance prompt are all unchanged, instead of paying for the
        # same LLM call again on every run.
        if (
            skip_if_unchanged
            and not args.force_reprocess
            and prior.get("reason") == "not_relevant"
            and prior.get("content_hash") == content_hash
            and prior.get("relevance_fingerprint") == relevance_fp
        ):
            processing_metadata["processed"] = False
            processing_metadata["reason"] = "Not relevant"
            return ArticleExtractionResult(
                **{**empty.__dict__, "skip_reason": "not_relevant_cached"}
            )

        # A letters-to-the-editor page is checked letter by letter, and only
        # the relevant letters are extracted, so signers and names from
        # off-topic letters don't become entities
        letters = split_letters(article_info["content"])
        if letters:
            rel_outcome = processor.check_letters_relevance(letters, article_id)
        else:
            rel_outcome = processor.check_relevance(article_info["content"], article_id)
        processing_metadata.setdefault("phase_outcomes", {})
        processing_metadata["phase_outcomes"]["relevance"] = (
            rel_outcome.to_metadata_dict()
        )
        if not rel_outcome.success:
            # The check itself failed (API error, bad response), so we don't
            # know whether the article is relevant. Skip it without
            # extracting, but record it separately from a real "not relevant"
            # verdict. Skipped articles stay unprocessed, so it is retried on
            # the next run.
            error_msg = rel_outcome.error.message if rel_outcome.error else "unknown"
            log(
                f"Relevance check failed for article {article_id}: {error_msg}",
                level="warning",
            )
            processing_metadata["processed"] = False
            processing_metadata["reason"] = f"Relevance check failed: {error_msg}"
            return ArticleExtractionResult(
                **{**empty.__dict__, "skip_reason": "relevance_error"}
            )
        if not rel_outcome.value:
            processing_metadata["processed"] = False
            processing_metadata["reason"] = "Not relevant"
            return ArticleExtractionResult(
                **{**empty.__dict__, "skip_reason": "not_relevant"}
            )
        if letters:
            article_info["content"] = rel_outcome.value
            processor.article_texts[article_id] = rel_outcome.value

    # Extract all entity types (potentially parallel within this article)
    with timed_step("extraction"):
        extracted_entities = processor.extract_all_entities(
            article_info["content"],
            article_id,
            processing_metadata,
            args.verbose,
            max_workers=extract_per_article,
        )

    return ArticleExtractionResult(
        row_index=row_index,
        row=row,
        article_info=article_info,
        processing_metadata=processing_metadata,
        extraction_timestamp=extraction_timestamp,
        extracted_entities=extracted_entities,
        was_extracted=True,
    )


def merge_and_finalize(
    result: ArticleExtractionResult,
    *,
    entities: Dict[str, Dict],
    processor: ArticleProcessor,
    args: argparse.Namespace,
    domain_config: DomainConfig,
    status_tracker: Optional[ProcessingStatus],
    counters: Dict[str, int],
) -> None:
    """Apply merge + status updates for one extraction result.

    **Must only be called from the single merge actor** (main thread).
    """
    article_id = result.article_info["id"]

    if not result.was_extracted:
        reason = result.skip_reason or "unknown"
        if reason == "not_relevant":
            counters["skipped_relevance_count"] += 1
            if status_tracker:
                # Store what produced the verdict so the next run can reuse it
                status_tracker.mark_skipped(
                    article_id,
                    "not_relevant",
                    metadata={
                        "content_hash": result.processing_metadata.get("content_hash"),
                        "relevance_fingerprint": result.processing_metadata.get(
                            "relevance_fingerprint"
                        ),
                    },
                )
        elif reason == "not_relevant_cached":
            # Sidecar entry from the earlier run is still valid; leave it as is
            counters["skipped_relevance_cached"] += 1
        elif reason == "relevance_error":
            counters["skipped_relevance_error"] += 1
            if status_tracker:
                status_tracker.mark_skipped(article_id, "relevance_error")
        elif reason == "already_processed":
            counters["skipped_already_processed"] += 1
        elif reason == "no_content":
            counters["skipped_no_content"] += 1
            if status_tracker:
                status_tracker.mark_skipped(article_id, "no_content")
        return

    log(f"\n[bold]Merging article #{result.row_index}[/bold]")

    # Merge extracted entities into shared entities dict
    merge_all_entities(
        result.extracted_entities,
        entities,
        result.article_info,
        result.extraction_timestamp,
        processor.model_type,
        result.processing_metadata,
        processor,
        domain_config=domain_config,
    )

    # Finalize processing metadata
    processor.finalize_processing_metadata(
        result.processing_metadata,
        result.extracted_entities,
        result.extraction_timestamp,
        args.verbose,
        result.row_index,
    )

    problems = incomplete_reasons(result.processing_metadata)
    if problems:
        result.processing_metadata["reason"] = f"incomplete: {', '.join(problems)}"
        # The types that worked come from the extraction cache next time, and
        # entities already merged from this article are not updated twice
        log(
            f"Article {article_id} incomplete ({', '.join(problems)}), "
            "will retry next run",
            level="warning",
        )
        counters["skipped_incomplete"] += 1
        if status_tracker:
            status_tracker.mark_skipped(
                article_id, "incomplete", metadata={"problems": problems}
            )
        return

    # Record in sidecar
    if status_tracker:
        status_tracker.mark_processed(article_id, result.processing_metadata)

    counters["processed_count"] += 1


def process_articles_batch(
    rows: List[Dict],
    processor: ArticleProcessor,
    entities: Dict[str, Dict],
    args: argparse.Namespace,
    domain_config: DomainConfig = None,
    status_tracker: ProcessingStatus = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Process a batch of articles with concurrent extraction + serial merge.

    Extraction workers run in a thread pool (``extract_workers``).  The main
    thread consumes results **in article order** and is the only writer to
    ``entities`` / ``status_tracker``.  At most ``max_buffered_articles``
    articles are submitted but not yet merged (``ordered_window``).
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    processed_rows: List[Dict] = []
    counters: Dict[str, int] = {
        "processed_count": 0,
        "skipped_relevance_count": 0,
        "skipped_relevance_error": 0,
        "skipped_relevance_cached": 0,
        "skipped_already_processed": 0,
        "skipped_no_content": 0,
        "skipped_incomplete": 0,
    }

    # Read concurrency settings
    domain_cfg = domain_config or DomainConfig(processor.domain)
    cc = domain_cfg.get_concurrency_config()
    extract_workers = cc["extract_workers"]
    extract_per_article = cc["extract_per_article"]
    # Articles submitted for extraction but not yet merged; at least one per
    # worker, or workers sit idle
    window = max(cc["max_buffered_articles"], extract_workers)

    # Cache config for skip-if-unchanged
    cache_cfg = domain_cfg.get_cache_config()
    skip_if_unchanged = cache_cfg.get("enabled", True) and cache_cfg.get(
        "articles", {}
    ).get("skip_if_unchanged", True)

    # Full metadata snapshot (avoids touching ProcessingStatus from workers).
    # Includes content_hash for skip-if-unchanged detection.
    status_snapshot: Dict[str, Dict[str, Any]] = (
        status_tracker.snapshot() if status_tracker else {}
    )

    batch_start = time.perf_counter()
    active_count = min(len(rows), args.limit)
    active_rows = rows[:active_count]
    remaining_rows = rows[active_count:]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Articles", total=active_count)

        # --- Submit extraction work, at most `window` articles ahead ---
        # Not a `with` block: its exit would wait for running extractions
        # before an interrupt could save the merged entities.
        pool = ThreadPoolExecutor(max_workers=extract_workers)

        def submit(item: Tuple[int, Dict]) -> Tuple[Any, Any]:
            row_index, row = item
            span = start_article_span(row_index, row)
            future = pool.submit(
                run_in_span,
                span,
                extract_single_article_only,
                row,
                row_index,
                processor,
                args,
                status_snapshot=status_snapshot,
                skip_if_unchanged=skip_if_unchanged,
                extract_per_article=extract_per_article,
            )
            return span, future

        save_every = save_every_articles()
        merged = 0
        try:
            # --- Consume results in submission (article) order ---
            for span, future in ordered_window(
                enumerate(active_rows, 1), submit, window
            ):
                result = future.result()
                processed_rows.append(result.row)
                with article_scope(span, result):
                    merge_and_finalize(
                        result,
                        entities=entities,
                        processor=processor,
                        args=args,
                        domain_config=domain_cfg,
                        status_tracker=status_tracker,
                        counters=counters,
                    )

                progress.update(task_id, advance=1)
                # Only articles that went through merging count: a resumed
                # run skips its already-processed ones without saving
                merged += bool(result.was_extracted)
                if save_every and result.was_extracted and merged % save_every == 0:
                    # A crash, kill or power cut keeps what is saved here;
                    # without it a 30-hour run lost everything merged so far
                    save_entities_then_status(
                        entities, domain_cfg.get_output_dir(), status_tracker
                    )
                write_partial_report(
                    domain_cfg.get_output_dir(),
                    counters["processed_count"],
                    time.perf_counter() - batch_start,
                )
                why = unreachable()
                if why:
                    raise ModelServerUnreachable(why)
        except (KeyboardInterrupt, ModelServerUnreachable):
            # Drop queued extractions and don't wait for running ones
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()

    # Append rows beyond the limit unchanged
    processed_rows.extend(remaining_rows)

    return processed_rows, counters


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def main():
    """Main processing function - orchestrates the entire article processing workflow."""
    log("Starting script...")
    run_start = time.perf_counter()
    # A shutdown or `kill` stops the run like Ctrl+C: it saves what is merged
    signal.signal(signal.SIGTERM, _interrupt)

    # Setup and initialization
    args = setup_arguments_and_config()
    config = DomainConfig(args.domain)
    base_dir = config.get_output_dir()
    entities = load_existing_entities(base_dir)

    # Initialize processor
    model_type = "ollama" if args.local else "gemini"

    # Enforce privacy: when --local is active, force local embeddings
    # so no data leaves the machine.
    if args.local:
        os.environ["EMBEDDING_MODE"] = "local"
        reset_embedding_manager_cache()
        ensure_local_embeddings_available()
        log(
            "Privacy mode: embeddings forced LOCAL (--local flag)",
            level="info",
        )

    # Per-task models (HINBOX_<TASK>_MODEL) and the trace endpoint; both refuse
    # anything off this machine/LAN under --local
    try:
        task_models = check_task_models(model_type)
        configure_tracing(base_dir, local=args.local, domain=args.domain)
    except ValueError as e:
        log(str(e), level="error")
        sys.exit(1)
    if len(set(task_models.values())) > 1:
        log(
            "Models per task: "
            + ", ".join(f"{task}={model}" for task, model in task_models.items()),
            level="info",
        )

    processor = ArticleProcessor(domain=args.domain, model_type=model_type)

    # Configure LLM concurrency limiters. Both are set because a task can have
    # its own model on the other side (e.g. a local merge check in a cloud run).
    cc = config.get_concurrency_config()
    configure_llm_concurrency(
        cloud_in_flight=cc["llm_in_flight"],
        local_in_flight=cc["ollama_in_flight"],
    )
    log(
        f"Concurrency: {cc['extract_workers']} workers, "
        f"{cc['extract_per_article']} types/article, "
        f"{cc['llm_in_flight']} cloud / {cc['ollama_in_flight']} local LLM in-flight, "
        f"{cc['max_buffered_articles']} articles buffered",
        level="info",
    )

    # Configure match-check memoization (per-run LRU for temperature=0 calls)
    cache_cfg = config.get_cache_config()
    match_cfg = cache_cfg.get("match_check", {})
    configure_match_check_memo(
        enabled=cache_cfg.get("enabled", True) and match_cfg.get("enabled", True),
        max_items=match_cfg.get("max_items", 8192),
    )

    # Configure persistent extraction sidecar cache
    configure_extraction_sidecar_cache(base_dir=base_dir, cache_cfg=cache_cfg)

    # Initialize sidecar processing status tracker
    status_tracker = ProcessingStatus(base_dir)
    log(
        f"Processing status: {status_tracker.total_processed} previously processed",
        level="info",
    )

    # Load articles
    rows = load_and_validate_articles(args.articles_path)
    if not rows:
        return
    for i, row in enumerate(rows, 1):  # numbered as process_articles_batch does
        info = processor.prepare_article_info(row, i)
        processor.article_texts[info["id"]] = stored_letters_text(
            info["content"], status_tracker.get_metadata(info["id"])
        )

    # A historian's corrections: the edit guard checks them during merging,
    # and the pass after merging carries out the strikes
    fixes_path = corrections_path(args.domain, config.load_config())
    configure_corrections(load_corrections(fixes_path))
    fixes = None

    # Process articles, then check grounding. Ctrl+C in either keeps the
    # articles merged so far (grounding is left for the next run)
    article_count = len(rows)
    try:
        # Splits and joins first, so merging meets the pages as corrected
        with timed_step("corrections_merges"):
            early = carry_out_merges(
                entities,
                processor.article_texts,
                model_type,
                args.domain,
                {i for i, s in last_states(fixes_path, base_dir).items() if s in DONE},
            )
        processed_rows, counters = process_articles_batch(
            rows,
            processor,
            entities,
            args,
            domain_config=config,
            status_tracker=status_tracker,
        )
        # Before grounding, so the checker judges the corrected text
        with timed_step("corrections_pass"):
            fixes = run_corrections_pass(
                entities, fixes_path, base_dir, write=False, early=early
            )
        if fixes:
            log(
                describe_report(fixes),
                level="warning" if fixes["counts"].get("needs a look") else "info",
            )
        # Also on a run with nothing new: a report whose check failed last
        # time (the server went away) is asked again; current ones cost nothing
        if not args.skip_grounding:
            log("\nRunning profile grounding verification...", level="processing")
            with timed_step("grounding_pass"):
                grounding_counts = run_profile_grounding_postprocess(
                    entities=entities,
                    rows=processed_rows,
                    processor=processor,
                    model_type=model_type,
                    workers=cc["grounding_workers"],
                )
            log(
                f"Grounding complete: {grounding_counts['verified']} verified, "
                f"{grounding_counts['skipped_unchanged']} unchanged, "
                f"{grounding_counts['skipped_no_citations']} no citations; "
                f"{grounding_counts['llm_checks']} article checks in "
                f"{grounding_counts['llm_calls']} model calls, "
                f"{grounding_counts['reused_checks']} kept from earlier runs",
                level="success",
            )
            log(
                "Claims in all grounding reports: "
                f"{describe_claim_counts(grounding_counts)} "
                f"({grounding_counts['profiles_with_problems']} profiles affected)",
                level="warning" if grounding_counts["not_supported"] else "info",
            )
            if grounding_counts["profiles_with_failed_checks"]:
                log(
                    f"{grounding_counts['profiles_with_failed_checks']} profiles "
                    "have a failed grounding check; the next run asks again",
                    level="warning",
                )
            why = unreachable()
            if why:
                raise ModelServerUnreachable(f"during grounding, {why}")
    except KeyboardInterrupt:
        log("Interrupted: saving entities merged so far...", level="warning")
        save_entities_then_status(entities, base_dir, status_tracker)
        if fixes:
            write_report(fixes_path, fixes)
        sys.exit(130)
    except ModelServerUnreachable as e:
        # Going on would only mark every later article incomplete
        log(
            f"Stopping: {e}. Saving entities merged so far; check the model "
            "server, then run the same command again to continue.",
            level="error",
        )
        save_entities_then_status(entities, base_dir, status_tracker)
        if fixes:
            write_report(fixes_path, fixes)
        sys.exit(1)

    # Write results and statistics
    write_results_and_statistics(
        processed_rows,
        entities,
        args,
        counters["processed_count"],
        counters["skipped_relevance_count"],
        counters["skipped_already_processed"],
        article_count,
        base_dir,
        status_tracker=status_tracker,
        skipped_relevance_error=counters["skipped_relevance_error"],
        skipped_relevance_cached=counters["skipped_relevance_cached"],
        skipped_incomplete=counters["skipped_incomplete"],
    )
    if fixes:
        write_report(fixes_path, fixes)
    save_timing_report(
        base_dir, counters["processed_count"], time.perf_counter() - run_start
    )


if __name__ == "__main__":
    main()
