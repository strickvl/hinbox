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
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.config_loader import DomainConfig
from src.constants import disable_llm_callbacks
from src.engine import ArticleProcessor, EntityMerger, configure_match_check_memo
from src.exceptions import ArticleLoadError
from src.logging_config import console, get_logger, log, set_show_profiles, set_verbose
from src.utils.cache_utils import sha256_text
from src.utils.embeddings.similarity import (
    ensure_local_embeddings_available,
    reset_embedding_manager_cache,
)
from src.utils.file_ops import write_entities_table
from src.utils.llm import configure_llm_concurrency
from src.utils.processing_status import ProcessingStatus
from src.utils.quality_controls import CITATION_RE, verify_profile_grounding

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


def load_existing_entities(base_dir: str) -> Dict[str, Dict]:
    """Load existing entities from Parquet files if they exist.

    Reads all existing entity Parquet files and loads them into memory for merging
    with newly extracted entities. Handles different keying schemes for each entity type.

    Returns:
        Dictionary with keys 'people', 'events', 'locations', 'organizations', each
        containing a dictionary of entities keyed by their unique identifiers:
        - people: keyed by name (string)
        - events: keyed by (title, start_date) tuple
        - locations: keyed by (name, type) tuple
        - organizations: keyed by (name, type) tuple

    Note:
        Missing Parquet files result in empty dictionaries for those entity types.
        This allows the pipeline to work even when starting from scratch.
    """
    people = {}
    events = {}
    locations = {}
    organizations = {}

    # Build file paths under the domain's output directory
    people_path = os.path.join(base_dir, "people.parquet")
    events_path = os.path.join(base_dir, "events.parquet")
    locations_path = os.path.join(base_dir, "locations.parquet")
    orgs_path = os.path.join(base_dir, "organizations.parquet")

    # People
    if os.path.exists(people_path):
        people_table = pq.read_table(people_path)
        for p in people_table.to_pylist():
            people[p["name"]] = p

    # Events
    if os.path.exists(events_path):
        events_table = pq.read_table(events_path)
        for e in events_table.to_pylist():
            events[(e["title"], e.get("start_date", ""))] = e

    # Locations
    if os.path.exists(locations_path):
        locations_table = pq.read_table(locations_path)
        for loc in locations_table.to_pylist():
            locations[(loc["name"], loc.get("type", ""))] = loc

    # Organizations
    if os.path.exists(orgs_path):
        orgs_table = pq.read_table(orgs_path)
        for o in orgs_table.to_pylist():
            organizations[(o["name"], o.get("type", ""))] = o

    return {
        "people": people,
        "events": events,
        "locations": locations,
        "organizations": organizations,
    }


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

    for entity_type, entity_dicts in merge_inputs:
        try:
            merge_start = datetime.now()
            merger = EntityMerger(entity_type)
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
            )
            merge_duration = (datetime.now() - merge_start).total_seconds()
            log(
                f"Merged {len(entity_dicts)} {entity_type} in {merge_duration:.2f}s "
                f"(new={merge_stats.new} merged={merge_stats.merged} skipped={merge_stats.skipped})",
                level="success",
            )
        except Exception as e:
            log(f"Error merging {entity_type}", level="error", exception=e)
            processing_metadata[f"error_{entity_type}"] = str(e)


def write_updated_articles(processed_rows: List[Dict], articles_path: str) -> None:
    """Write updated articles back to the parquet file."""
    temp_file = articles_path + ".tmp.parquet"
    try:
        log("Writing updated articles to parquet file...", level="processing")
        new_table = pa.Table.from_pylist(processed_rows)
        pq.write_table(new_table, temp_file)
        os.replace(temp_file, articles_path)
        log("Successfully updated articles parquet file", level="success")
    except Exception as e:
        log("Could not write updated articles to parquet", level="error", exception=e)
        if os.path.exists(temp_file):
            os.remove(temp_file)


def calculate_reflection_statistics(
    processed_rows: List[Dict], processed_count: int
) -> Tuple[int, float]:
    """Calculate reflection statistics from processed articles."""
    total_reflection_attempts = 0
    for row in processed_rows:
        if (
            "processing_metadata" in row
            and row["processing_metadata"] is not None
            and "reflection_summary" in row["processing_metadata"]
            and row["processing_metadata"]["reflection_summary"] is not None
        ):
            total_reflection_attempts += row["processing_metadata"][
                "reflection_summary"
            ].get("total_attempts", 0)

    avg_reflections = (
        total_reflection_attempts / processed_count if processed_count > 0 else 0
    )
    return total_reflection_attempts, avg_reflections


def log_processing_summary(
    article_count: int,
    processed_count: int,
    skipped_relevance_count: int,
    skipped_already_processed: int,
    entities: Dict[str, Dict],
    processed_rows: List[Dict],
) -> None:
    """Log comprehensive processing summary and statistics."""
    log("\n[bold]Processing complete[/bold]", level="info")

    # Article processing summary
    log(f"Articles read: {article_count}", level="info")
    log(f"Articles processed: {processed_count}", level="info")
    log(f"Articles skipped (relevance): {skipped_relevance_count}", level="info")
    log(
        f"Articles skipped (already processed): {skipped_already_processed}",
        level="info",
    )

    # Entity counts
    log("\nFinal entity counts:", level="info")
    log(f"• People: {len(entities['people'])}", level="info")
    log(f"• Organizations: {len(entities['organizations'])}", level="info")
    log(f"• Locations: {len(entities['locations'])}", level="info")
    log(f"• Events: {len(entities['events'])}", level="info")

    # Reflection statistics
    if processed_count > 0:
        total_reflection_attempts, avg_reflections = calculate_reflection_statistics(
            processed_rows, processed_count
        )
        if total_reflection_attempts > 0:
            log("\nReflection statistics:", level="info")
            log(
                f"• Total reflection attempts: {total_reflection_attempts}",
                level="info",
            )
            log(
                f"• Average reflection attempts per article: {avg_reflections:.2f}",
                level="info",
            )


def run_profile_grounding_postprocess(
    *,
    entities: Dict[str, Dict],
    rows: List[Dict],
    processor: ArticleProcessor,
    model_type: str,
) -> Dict[str, int]:
    """Run grounding verification on entity profiles as a batch post-processing step.

    For each entity with citations in its profile, verifies that claims are
    supported by source article text. Stores the grounding report on each
    entity as entity["profile_grounding"]. Skips entities whose profile
    hasn't changed since the last verification (hash-based).

    Returns summary counts of verified/skipped/failed entities.
    """
    import hashlib

    # Build article_id -> content map from rows
    article_texts: Dict[str, str] = {}
    for i, row in enumerate(rows):
        info = processor.prepare_article_info(row, i)
        if info["id"] and info["content"]:
            article_texts[info["id"]] = info["content"]

    counts = {"verified": 0, "skipped_unchanged": 0, "skipped_no_citations": 0}

    for entity_type, entity_dict in entities.items():
        for entity_key, entity_data in entity_dict.items():
            profile_text = (entity_data.get("profile") or {}).get("text", "")
            if not profile_text:
                continue

            # Check if profile has citations worth verifying
            citations = CITATION_RE.findall(profile_text)
            if not citations:
                counts["skipped_no_citations"] += 1
                continue

            # Skip if profile hasn't changed since last grounding check
            current_hash = hashlib.sha256(profile_text.encode()).hexdigest()
            existing_grounding = entity_data.get("profile_grounding")
            if (
                existing_grounding
                and isinstance(existing_grounding, dict)
                and existing_grounding.get("profile_text_hash") == current_hash
            ):
                counts["skipped_unchanged"] += 1
                continue

            # Run grounding verification
            report = verify_profile_grounding(
                profile_text=profile_text,
                article_texts=article_texts,
                model_type=model_type,
            )

            entity_data["profile_grounding"] = report.model_dump(exclude_none=True)
            counts["verified"] += 1

            if report.grounding_score is not None:
                log(
                    f"Grounding for {entity_type}/{entity_key}: "
                    f"score={report.grounding_score:.2f}, "
                    f"verified={report.verified}/{report.total_citations}",
                    level="info",
                )

    return counts


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
) -> None:
    """Write results to files and log comprehensive statistics."""
    # Flush processing status sidecar (replaces full articles rewrite)
    if status_tracker:
        status_tracker.flush()

    # Write entity files
    log("Saving updated entity tables...", level="processing")
    write_entities_to_files(entities, base_dir)
    log("Entity tables successfully saved", level="success")

    # Log comprehensive summary
    log_processing_summary(
        article_count,
        processed_count,
        skipped_relevance_count,
        skipped_already_processed,
        entities,
        processed_rows,
    )


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
    already_processed = prior.get("processed", False) or processing_metadata.get(
        "processed"
    )
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
        rel_outcome = processor.check_relevance(article_info["content"], article_id)
        processing_metadata.setdefault("phase_outcomes", {})
        processing_metadata["phase_outcomes"]["relevance"] = (
            rel_outcome.to_metadata_dict()
        )
        if not rel_outcome.value:
            processing_metadata["processed"] = False
            processing_metadata["reason"] = "Not relevant"
            return ArticleExtractionResult(
                **{**empty.__dict__, "skip_reason": "not_relevant"}
            )

    # Extract all entity types (potentially parallel within this article)
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
                status_tracker.mark_skipped(article_id, "not_relevant")
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
    ``entities`` / ``status_tracker``.
    """
    from concurrent.futures import ThreadPoolExecutor

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
        "skipped_already_processed": 0,
        "skipped_no_content": 0,
    }

    # Read concurrency settings
    domain_cfg = domain_config or DomainConfig(processor.domain)
    cc = domain_cfg.get_concurrency_config()
    extract_workers = cc["extract_workers"]
    extract_per_article = cc["extract_per_article"]

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

        # --- Submit extraction work to thread pool ---
        with ThreadPoolExecutor(max_workers=extract_workers) as pool:
            futures = [
                pool.submit(
                    extract_single_article_only,
                    row,
                    row_index,
                    processor,
                    args,
                    status_snapshot=status_snapshot,
                    skip_if_unchanged=skip_if_unchanged,
                    extract_per_article=extract_per_article,
                )
                for row_index, row in enumerate(active_rows, 1)
            ]

            # --- Consume results in submission (article) order ---
            for future in futures:
                result = future.result()
                processed_rows.append(result.row)

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

    # Append rows beyond the limit unchanged
    processed_rows.extend(remaining_rows)

    return processed_rows, counters


def main():
    """Main processing function - orchestrates the entire article processing workflow."""
    log("Starting script...")

    # Setup and initialization
    args = setup_arguments_and_config()
    config = DomainConfig(args.domain)
    base_dir = config.get_output_dir()
    entities = load_existing_entities(base_dir)

    # Initialize processor
    model_type = "ollama" if args.local else "gemini"

    # Enforce privacy: when --local is active, force local embeddings and
    # disable telemetry so no data leaves the machine.
    if args.local:
        disable_llm_callbacks()
        os.environ["EMBEDDING_MODE"] = "local"
        reset_embedding_manager_cache()
        ensure_local_embeddings_available()
        log(
            "Privacy mode: embeddings + callbacks forced LOCAL (--local flag)",
            level="info",
        )

    processor = ArticleProcessor(domain=args.domain, model_type=model_type)

    # Configure LLM concurrency limiter
    cc = config.get_concurrency_config()
    configure_llm_concurrency(
        cloud_in_flight=cc["llm_in_flight"] if model_type == "gemini" else None,
        local_in_flight=cc["ollama_in_flight"] if model_type == "ollama" else None,
    )
    log(
        f"Concurrency: {cc['extract_workers']} workers, "
        f"{cc['extract_per_article']} types/article, "
        f"{cc['llm_in_flight']} LLM in-flight",
        level="info",
    )

    # Configure match-check memoization (per-run LRU for temperature=0 calls)
    cache_cfg = config.get_cache_config()
    match_cfg = cache_cfg.get("match_check", {})
    configure_match_check_memo(
        enabled=cache_cfg.get("enabled", True) and match_cfg.get("enabled", True),
        max_items=match_cfg.get("max_items", 8192),
    )

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

    # Process articles
    article_count = len(rows)
    processed_rows, counters = process_articles_batch(
        rows,
        processor,
        entities,
        args,
        domain_config=config,
        status_tracker=status_tracker,
    )

    # Post-processing: verify profile grounding
    if counters["processed_count"] > 0:
        log("\nRunning profile grounding verification...", level="processing")
        grounding_counts = run_profile_grounding_postprocess(
            entities=entities,
            rows=processed_rows,
            processor=processor,
            model_type=model_type,
        )
        log(
            f"Grounding complete: {grounding_counts['verified']} verified, "
            f"{grounding_counts['skipped_unchanged']} unchanged, "
            f"{grounding_counts['skipped_no_citations']} no citations",
            level="success",
        )

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
    )


if __name__ == "__main__":
    main()
