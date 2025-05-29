#!/usr/bin/env python3
"""
Script merging logic from run.py and extract_entities.py, iterating over articles in
data/raw_sources/miami_herald_articles.parquet, extracting entities (people, events,
locations, organizations) via Gemini or local approach, then merging results
into the data/entities/*.parquet files.
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from src.constants import (
    ARTICLES_PATH,
    CLOUD_MODEL,
    EVENTS_OUTPUT_PATH,
    LOCATIONS_OUTPUT_PATH,
    OLLAMA_MODEL,
    ORGANIZATIONS_OUTPUT_PATH,
    OUTPUT_DIR,
    PEOPLE_OUTPUT_PATH,
)
from src.events import gemini_extract_events, ollama_extract_events
from src.exceptions import (
    ArticleLoadError,
    EntityExtractionError,
    RelevanceCheckError,
)
from src.locations import gemini_extract_locations, ollama_extract_locations
from src.logging_config import get_logger, log, set_verbose
from src.merge import merge_events, merge_locations, merge_organizations, merge_people
from src.organizations import gemini_extract_organizations, ollama_extract_organizations
from src.people import gemini_extract_people, ollama_extract_people
from src.relevance import gemini_check_relevance, ollama_check_relevance
from src.utils.error_handler import (
    handle_article_processing_error,
)
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


def track_reflection_attempts(
    extracted_entities, entity_type: str, processing_metadata: Dict, verbose: bool
) -> int:
    """Track reflection attempts for entity extraction."""
    reflection_history = []
    reflection_attempts = 1

    # Check if the response has reflection history attribute
    if hasattr(extracted_entities, "reflection_history"):
        reflection_history = extracted_entities.reflection_history
        reflection_attempts = len(reflection_history) if reflection_history else 1

        if verbose and reflection_history:
            log(f"Reflection history for {entity_type} extraction:", level="debug")
            for i, reflection in enumerate(reflection_history):
                passed = reflection.get("passed", False)
                feedback = reflection.get("feedback", "No feedback")
                log(
                    f"  Attempt {i + 1}: {'✓' if passed else '✗'} {feedback}",
                    level="debug",
                )

    # For entity extraction modules that return merged results
    if (
        isinstance(extracted_entities, dict)
        and "reflection_history" in extracted_entities
    ):
        reflection_history = extracted_entities["reflection_history"]
        reflection_attempts = len(reflection_history) if reflection_history else 1

    return reflection_attempts


def extract_single_entity_type(
    entity_type: str, article_content: str, model_type: str, domain: str
) -> List[Dict]:
    """Extract a single entity type from article content."""
    try:
        if model_type == "ollama":
            if entity_type == "people":
                return ollama_extract_people(
                    article_content, model="qwq", domain=domain
                )
            elif entity_type == "organizations":
                return ollama_extract_organizations(
                    article_content, model="qwq", domain=domain
                )
            elif entity_type == "locations":
                return ollama_extract_locations(
                    article_content, model="qwq", domain=domain
                )
            elif entity_type == "events":
                return ollama_extract_events(
                    article_content, model="qwq", domain=domain
                )
        else:
            if entity_type == "people":
                return gemini_extract_people(article_content, domain=domain)
            elif entity_type == "organizations":
                return gemini_extract_organizations(article_content, domain=domain)
            elif entity_type == "locations":
                return gemini_extract_locations(article_content, domain=domain)
            elif entity_type == "events":
                return gemini_extract_events(article_content, domain=domain)
    except Exception as e:
        log(f"Error extracting {entity_type}", level="error", exception=e)
        return []

    return []


def extract_entities_from_article(
    article_content: str,
    article_id: str,
    model_type: str,
    domain: str,
    processing_metadata: Dict,
    verbose: bool,
) -> Dict[str, List[Dict]]:
    """Extract all entity types from article content."""
    extracted_entities = {}
    entity_types = ["people", "organizations", "locations", "events"]

    for entity_type in entity_types:
        log(f"Extracting {entity_type}...", level="processing")
        start_time = datetime.now()

        try:
            entities = extract_single_entity_type(
                entity_type, article_content, model_type, domain
            )

            # Track reflection attempts
            reflection_attempts = track_reflection_attempts(
                entities, entity_type, processing_metadata, verbose
            )

            duration = (datetime.now() - start_time).total_seconds()
            log(
                f"Extracted {len(entities)} {entity_type} in {duration:.2f}s",
                level="success",
            )

            if reflection_attempts > 1:
                log(
                    f"Required {reflection_attempts} reflection iterations",
                    level="debug",
                )

            # Update reflection metadata
            processing_metadata["reflection_attempts"][entity_type] = {
                "attempts": reflection_attempts,
                "success": bool(entities),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
            }

            # Update summary counts
            processing_metadata["reflection_summary"]["total_attempts"] += (
                reflection_attempts
            )
            processing_metadata["reflection_summary"]["successful_attempts"] += 1

            extracted_entities[entity_type] = entities

        except Exception as e:
            error = EntityExtractionError(
                f"{entity_type.title()} extraction failed",
                entity_type,
                article_id,
                {"original_error": str(e), "model_type": model_type},
            )
            handle_article_processing_error(
                article_id, f"{entity_type}_extraction", error
            )
            extracted_entities[entity_type] = []
            processing_metadata["reflection_attempts"][entity_type] = {
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
                "attempts": 1,
                "success": False,
            }
            processing_metadata["reflection_summary"]["failed_attempts"] += 1

    return extracted_entities


def convert_pydantic_to_dict(items: List) -> List[Dict]:
    """Convert Pydantic models to dictionaries."""
    dicts = []
    for obj in items:
        # pydantic models have model_dump() or dict()
        if hasattr(obj, "model_dump"):
            dicts.append(obj.model_dump())
        elif hasattr(obj, "dict"):
            dicts.append(obj.dict())
        else:
            dicts.append(obj)
    return dicts


def merge_all_entities(
    extracted_entities: Dict[str, List],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: str,
    article_content: str,
    extraction_timestamp: str,
    model_type: str,
    processing_metadata: Dict,
) -> None:
    """Merge all extracted entities with existing entities."""
    log("Merging extracted entities...", level="processing")

    # Convert Pydantic to dict if needed
    people_dicts = convert_pydantic_to_dict(extracted_entities.get("people", []))
    org_dicts = convert_pydantic_to_dict(extracted_entities.get("organizations", []))
    loc_dicts = convert_pydantic_to_dict(extracted_entities.get("locations", []))
    event_dicts = convert_pydantic_to_dict(extracted_entities.get("events", []))

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
                article_id,
                article_title,
                article_url,
                article_published_date,
                article_content,
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


def update_processing_metadata(
    processing_metadata: Dict,
    extracted_entities: Dict[str, List],
    extraction_timestamp: str,
    model_type: str,
    verbose: bool,
    row_index: int,
) -> float:
    """Update processing metadata and calculate processing time."""
    # Mark processing complete
    processing_metadata["reflection_used"] = True
    processing_metadata["processed"] = True
    processing_metadata["processing_completed"] = datetime.now().isoformat()

    # Calculate processing time
    start_time = datetime.fromisoformat(processing_metadata["processing_started"])
    end_time = datetime.fromisoformat(processing_metadata["processing_completed"])
    processing_time = (end_time - start_time).total_seconds()

    # Record total reflection stats
    total_reflection_attempts = processing_metadata["reflection_summary"][
        "total_attempts"
    ]

    # Store extraction counts and processing time
    processing_metadata["entities_extracted"] = {
        "people": len(extracted_entities.get("people", [])),
        "organizations": len(extracted_entities.get("organizations", [])),
        "locations": len(extracted_entities.get("locations", [])),
        "events": len(extracted_entities.get("events", [])),
        "total": sum(len(entities) for entities in extracted_entities.values()),
    }
    processing_metadata["processing_time_seconds"] = processing_time

    # Log reflection summary
    log(f"Reflection summary for article #{row_index}:", level="info")
    log(f"  Total reflection attempts: {total_reflection_attempts}", level="info")

    # Only log detailed reflection stats in verbose mode or if there were multiple attempts
    if (
        verbose or total_reflection_attempts > 4
    ):  # 4 = minimum if all extractions took just 1 attempt
        for entity_type, reflection_data in processing_metadata[
            "reflection_attempts"
        ].items():
            attempts = reflection_data.get("attempts", 1)
            duration = reflection_data.get("duration_seconds", 0)
            if attempts > 1:
                log(
                    f"  • {entity_type}: {attempts} attempts in {duration:.2f}s",
                    level="info",
                )

    log(
        f"Successfully processed article #{row_index} in {processing_time:.2f}s",
        level="success",
    )

    return processing_time


def check_article_relevance(
    article_content: str, article_id: str, model_type: str, domain: str
) -> bool:
    """Check if article is relevant to the domain."""
    log("Performing relevance check...", level="processing")
    try:
        if model_type == "ollama":
            relevance_result = ollama_check_relevance(
                article_content, model="qwq", domain=domain
            )
        else:
            relevance_result = gemini_check_relevance(article_content, domain=domain)

        if not relevance_result.is_relevant:
            log("Skipping article as it's not relevant", level="warning")
            log(f"Reason: {relevance_result.reason}", level="debug")
            return False
        else:
            log("Article is relevant", level="success")
            return True
    except Exception as e:
        error = RelevanceCheckError(
            "Relevance check failed",
            "unknown",
            article_id,
            {"original_error": str(e), "model_type": model_type},
        )
        handle_article_processing_error(article_id, "relevance_check", error)
        log(
            "Proceeding with extraction despite relevance check error",
            level="warning",
        )
        return True  # Default to relevant if check fails


def write_results_and_statistics(
    processed_rows: List[Dict],
    entities: Dict[str, Dict],
    args: argparse.Namespace,
    processed_count: int,
    skipped_relevance_count: int,
    skipped_already_processed: int,
    article_count: int,
) -> None:
    """Write results to files and log statistics."""
    # Write updated articles to a temp parquet file
    temp_file = args.articles_path + ".tmp.parquet"
    try:
        log("Writing updated articles to parquet file...", level="processing")
        new_table = pa.Table.from_pylist(processed_rows)
        pq.write_table(new_table, temp_file)
        os.replace(temp_file, args.articles_path)
        log("Successfully updated articles parquet file", level="success")
    except Exception as e:
        log("Could not write updated articles to parquet", level="error", exception=e)
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Write final entity sets to their parquet files
    log("Saving updated entity tables...", level="processing")
    write_entities_to_files(entities)
    log("Entity tables successfully saved", level="success")

    # Print summary information
    log("\n[bold]Processing complete[/bold]", level="info")

    # Create a summary table in the log
    log(f"Articles read: {article_count}", level="info")
    log(f"Articles processed: {processed_count}", level="info")
    log(f"Articles skipped (relevance): {skipped_relevance_count}", level="info")
    log(
        f"Articles skipped (already processed): {skipped_already_processed}",
        level="info",
    )

    log("\nFinal entity counts:", level="info")
    log(f"• People: {len(entities['people'])}", level="info")
    log(f"• Organizations: {len(entities['organizations'])}", level="info")
    log(f"• Locations: {len(entities['locations'])}", level="info")
    log(f"• Events: {len(entities['events'])}", level="info")

    # Add reflection statistics if we processed any articles
    if processed_count > 0:
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

        if total_reflection_attempts > 0:
            avg_reflections = total_reflection_attempts / processed_count
            log(f"\nReflection statistics:", level="info")
            log(
                f"• Total reflection attempts: {total_reflection_attempts}",
                level="info",
            )
            log(
                f"• Average reflection attempts per article: {avg_reflections:.2f}",
                level="info",
            )


def main():
    """Main processing function - now broken down into smaller, focused functions."""
    log("Starting script...")

    # Setup arguments and configuration
    args = setup_arguments_and_config()

    # Load existing entities
    log("Loading existing entities...", level="processing")
    entities = load_existing_entities()

    # Determine model configuration
    model_type = "ollama" if args.local else "gemini"
    specific_model = OLLAMA_MODEL if args.local else CLOUD_MODEL

    # Load and validate articles
    rows = load_and_validate_articles(args.articles_path)
    if not rows:
        return

    # Initialize counters
    article_count = len(rows)
    processed_count = 0
    skipped_relevance_count = 0
    skipped_already_processed = 0
    processed_rows = []
    row_index = 0

    # Process each article
    for row in rows:
        if row_index >= args.limit:
            # We've hit the limit; keep the rest unmodified
            processed_rows.append(row)
            row_index += 1
            continue

        row_index += 1
        log(f"\n[bold]Processing article #{row_index}[/bold]")

        # Extract article information
        article_id = row.get("id", f"article_{row_index}")
        article_title = row.get("title", "")
        article_url = row.get("url", "")
        article_published_date = row.get("published_date", "")
        article_content = row.get("content", "")

        # Initialize processing metadata
        if "processing_metadata" not in row:
            row["processing_metadata"] = {}
        processing_metadata = row["processing_metadata"]

        # Initialize enhanced reflection metadata tracking
        processing_metadata["reflection_attempts"] = {}
        processing_metadata["reflection_summary"] = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
        }

        # Skip if already processed and not forced
        if processing_metadata.get("processed") and not args.force_reprocess:
            log("Article already processed, skipping...", level="warning")
            skipped_already_processed += 1
            processed_rows.append(row)
            continue

        if not article_content:
            log("Article has no content, skipping extraction", level="warning")
            processed_rows.append(row)
            continue

        # Mark that we started processing
        processing_metadata["processing_started"] = datetime.now().isoformat()
        processing_metadata["model_type"] = model_type
        processing_metadata["specific_model"] = specific_model

        # Relevance check
        if args.relevance_check:
            if not check_article_relevance(
                article_content, article_id, model_type, args.domain
            ):
                processing_metadata["processed"] = False
                processing_metadata["reason"] = "Not relevant"
                skipped_relevance_count += 1
                processed_rows.append(row)
                continue

        extraction_timestamp = datetime.now().isoformat()

        # Extract all entity types
        extracted_entities = extract_entities_from_article(
            article_content,
            article_id,
            model_type,
            args.domain,
            processing_metadata,
            args.verbose,
        )

        # Merge all entities
        merge_all_entities(
            extracted_entities,
            entities,
            article_id,
            article_title,
            article_url,
            article_published_date,
            article_content,
            extraction_timestamp,
            model_type,
            processing_metadata,
        )

        # Update processing metadata
        update_processing_metadata(
            processing_metadata,
            extracted_entities,
            extraction_timestamp,
            model_type,
            args.verbose,
            row_index,
        )

        processed_rows.append(row)
        processed_count += 1

    # Write results and statistics
    write_results_and_statistics(
        processed_rows,
        entities,
        args,
        processed_count,
        skipped_relevance_count,
        skipped_already_processed,
        article_count,
    )


if __name__ == "__main__":
    main()
