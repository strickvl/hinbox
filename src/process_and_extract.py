#!/usr/bin/env python3
"""
Script merging logic from run.py and extract_entities.py, iterating over articles in
data/raw_sources/miami_herald_articles.parquet, extracting entities (people, events,
locations, organizations) via Gemini or local approach, then merging results
into the data/entities/*.parquet files.
"""

import argparse
import os
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.constants import (
    ARTICLES_PATH,
    EVENTS_OUTPUT_PATH,
    LOCATIONS_OUTPUT_PATH,
    ORGANIZATIONS_OUTPUT_PATH,
    OUTPUT_DIR,
    PEOPLE_OUTPUT_PATH,
)
from src.exceptions import ArticleLoadError
from src.logging_config import get_logger, log, set_verbose
from src.merge import merge_events, merge_locations, merge_organizations, merge_people
from src.processing.article_processor import ArticleProcessor
from src.utils.file_ops import write_entity_to_file

# Get module-specific logger
logger = get_logger("process_and_extract")


def ensure_dir(directory: str):
    """
    Ensure that a directory exists, creating it if necessary.
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def load_existing_entities() -> Dict[str, Dict]:
    """
    Load existing entities from Parquet files if they exist.
    Returns a dict with keys: people, events, locations, organizations
    Each is a dictionary keyed by the relevant unique tuple or name.
    """
    import pyarrow.parquet as pq

    people = {}
    events = {}
    locations = {}
    organizations = {}

    # People
    if os.path.exists(PEOPLE_OUTPUT_PATH):
        people_table = pq.read_table(PEOPLE_OUTPUT_PATH)
        for p in people_table.to_pylist():
            people[p["name"]] = p

    # Events
    if os.path.exists(EVENTS_OUTPUT_PATH):
        events_table = pq.read_table(EVENTS_OUTPUT_PATH)
        for e in events_table.to_pylist():
            events[(e["title"], e.get("start_date", ""))] = e

    # Locations
    if os.path.exists(LOCATIONS_OUTPUT_PATH):
        locations_table = pq.read_table(LOCATIONS_OUTPUT_PATH)
        for l in locations_table.to_pylist():
            locations[(l["name"], l.get("type", ""))] = l

    # Organizations
    if os.path.exists(ORGANIZATIONS_OUTPUT_PATH):
        orgs_table = pq.read_table(ORGANIZATIONS_OUTPUT_PATH)
        for o in orgs_table.to_pylist():
            organizations[(o["name"], o.get("type", ""))] = o

    return {
        "people": people,
        "events": events,
        "locations": locations,
        "organizations": organizations,
    }


def write_entities_to_files(entities: Dict[str, Dict]):
    """
    Write entities to their respective Parquet files.
    """
    for entity_type, entity_dict in entities.items():
        for entity_key, entity_data in entity_dict.items():
            write_entity_to_file(entity_type, entity_key, entity_data)


def setup_arguments_and_config() -> argparse.Namespace:
    """Setup command line arguments and configuration."""
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
        default=ARTICLES_PATH,
        help="Path to the raw articles Parquet file",
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
    args = parser.parse_args()

    # Configure logger level based on verbosity flag
    if args.verbose:
        set_verbose(True)
        log("Verbose logging enabled")

    log(
        f"[bold blue]Arguments:[/] limit={args.limit}, local={args.local}, relevance_check={args.relevance_check}, force_reprocess={args.force_reprocess}"
    )

    ensure_dir(OUTPUT_DIR)
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
) -> None:
    """Merge all extracted entities with existing entities."""
    log("Merging extracted entities...", level="processing")

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

    # Merge each entity type
    merge_operations = [
        ("people", people_dicts, merge_people),
        ("organizations", org_dicts, merge_organizations),
        ("locations", loc_dicts, merge_locations),
        ("events", event_dicts, merge_events),
    ]

    for entity_type, entity_dicts, merge_func in merge_operations:
        try:
            merge_start = datetime.now()
            merge_func(
                entity_dicts,
                entities,
                article_info["id"],
                article_info["title"],
                article_info["url"],
                article_info["published_date"],
                article_info["content"],
                extraction_timestamp,
                model_type,
            )
            merge_duration = (datetime.now() - merge_start).total_seconds()
            log(
                f"Merged {len(entity_dicts)} {entity_type} in {merge_duration:.2f}s",
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
            log(f"\nReflection statistics:", level="info")
            log(
                f"• Total reflection attempts: {total_reflection_attempts}",
                level="info",
            )
            log(
                f"• Average reflection attempts per article: {avg_reflections:.2f}",
                level="info",
            )


def write_results_and_statistics(
    processed_rows: List[Dict],
    entities: Dict[str, Dict],
    args: argparse.Namespace,
    processed_count: int,
    skipped_relevance_count: int,
    skipped_already_processed: int,
    article_count: int,
) -> None:
    """Write results to files and log comprehensive statistics."""
    # Write updated articles
    write_updated_articles(processed_rows, args.articles_path)

    # Write entity files
    log("Saving updated entity tables...", level="processing")
    write_entities_to_files(entities)
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


def process_single_article(
    row: Dict,
    row_index: int,
    processor: ArticleProcessor,
    entities: Dict[str, Dict],
    args: argparse.Namespace,
    langfuse_session_id: str,
    langfuse_trace_id: str,
) -> Tuple[Dict, bool, str]:
    """Process a single article through the extraction pipeline.

    Returns:
        Tuple of (updated_row, was_processed, skip_reason)
    """
    log(f"\n[bold]Processing article #{row_index}[/bold]")

    # Extract article information
    article_info = processor.prepare_article_info(row, row_index)

    # Initialize processing metadata
    processing_metadata = processor.initialize_processing_metadata(row)

    # Skip if already processed and not forced
    if processing_metadata.get("processed") and not args.force_reprocess:
        log("Article already processed, skipping...", level="warning")
        return row, False, "already_processed"

    if not article_info["content"]:
        log("Article has no content, skipping extraction", level="warning")
        return row, False, "no_content"

    # Relevance check
    if args.relevance_check:
        if not processor.check_relevance(article_info["content"], article_info["id"]):
            processing_metadata["processed"] = False
            processing_metadata["reason"] = "Not relevant"
            return row, False, "not_relevant"

    extraction_timestamp = datetime.now().isoformat()

    # Extract all entity types
    extracted_entities = processor.extract_all_entities(
        article_info["content"],
        article_info["id"],
        processing_metadata,
        args.verbose,
    )

    # Merge all entities
    merge_all_entities(
        extracted_entities,
        entities,
        article_info,
        extraction_timestamp,
        processor.model_type,
        processing_metadata,
        processor,
    )

    # Finalize processing metadata
    processor.finalize_processing_metadata(
        processing_metadata,
        extracted_entities,
        extraction_timestamp,
        args.verbose,
        row_index,
    )

    return row, True, "processed"


def process_articles_batch(
    rows: List[Dict],
    processor: ArticleProcessor,
    entities: Dict[str, Dict],
    args: argparse.Namespace,
    langfuse_session_id: str,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Process a batch of articles and return results with statistics."""
    processed_rows = []
    counters = {
        "processed_count": 0,
        "skipped_relevance_count": 0,
        "skipped_already_processed": 0,
        "skipped_no_content": 0,
    }

    for row_index, row in enumerate(rows, 1):
        langfuse_trace_id = f"{row['id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if row_index > args.limit:
            # We've hit the limit; keep the rest unmodified
            processed_rows.append(row)
            continue

        # Process the article
        updated_row, was_processed, skip_reason = process_single_article(
            row,
            row_index,
            processor,
            entities,
            args,
            langfuse_session_id,
            langfuse_trace_id,
        )

        processed_rows.append(updated_row)

        if was_processed:
            counters["processed_count"] += 1
        elif skip_reason == "not_relevant":
            counters["skipped_relevance_count"] += 1
        elif skip_reason == "already_processed":
            counters["skipped_already_processed"] += 1
        elif skip_reason == "no_content":
            counters["skipped_no_content"] += 1

    return processed_rows, counters


def main():
    """Main processing function - orchestrates the entire article processing workflow."""
    log("Starting script...")

    # create a unique session id
    langfuse_session_id = f"{uuid.uuid4()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Setup and initialization
    args = setup_arguments_and_config()
    entities = load_existing_entities()

    # Initialize processor
    model_type = "ollama" if args.local else "gemini"
    processor = ArticleProcessor(domain=args.domain, model_type=model_type)

    # Load articles
    rows = load_and_validate_articles(args.articles_path)
    if not rows:
        return

    # Process articles
    article_count = len(rows)
    processed_rows, counters = process_articles_batch(
        rows, processor, entities, args, langfuse_session_id
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
    )


if __name__ == "__main__":
    main()
