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
from typing import Dict

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


def main():
    log("Starting script...")

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

    # Load existing entities
    log("Loading existing entities...", level="processing")
    entities = load_existing_entities()

    model_type = "ollama" if args.local else "gemini"
    specific_model = OLLAMA_MODEL if args.local else CLOUD_MODEL

    # Check if articles parquet exists
    if not os.path.exists(args.articles_path):
        log(f"ERROR: Articles file not found at {args.articles_path}", level="error")
        return

    # Read entire Parquet into memory
    try:
        table = pq.read_table(args.articles_path)
        rows = table.to_pylist()
    except Exception as e:
        error = ArticleLoadError(
            f"Failed to read articles from {args.articles_path}",
            {"file_path": args.articles_path, "original_error": str(e)},
        )
        log("Failed to read Parquet file", level="error", exception=error)
        return

    article_count = len(rows)
    processed_count = 0
    skipped_relevance_count = 0
    skipped_already_processed = 0

    log(f"Loaded {article_count} articles from {args.articles_path}", level="success")

    processed_rows = []
    row_index = 0

    for row in rows:
        if row_index >= args.limit:
            # We've hit the limit; keep the rest unmodified
            processed_rows.append(row)
            row_index += 1
            continue

        row_index += 1
        log(f"\n[bold]Processing article #{row_index}[/bold]")

        article_id = row.get("id", f"article_{row_index}")
        article_title = row.get("title", "")
        article_url = row.get("url", "")
        article_published_date = row.get("published_date", "")
        article_content = row.get("content", "")

        # Initialize or check processing_metadata
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
            log("Performing relevance check...", level="processing")
            try:
                if args.local:
                    relevance_result = ollama_check_relevance(
                        article_content, model="qwq", domain=args.domain
                    )
                else:
                    relevance_result = gemini_check_relevance(
                        article_content, domain=args.domain
                    )

                if not relevance_result.is_relevant:
                    log("Skipping article as it's not relevant", level="warning")
                    log(f"Reason: {relevance_result.reason}", level="debug")
                    processing_metadata["processed"] = False
                    processing_metadata["reason"] = relevance_result.reason
                    skipped_relevance_count += 1
                    processed_rows.append(row)
                    continue
                else:
                    log("Article is relevant", level="success")
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

        extraction_timestamp = datetime.now().isoformat()

        # Extract people with enhanced reflection tracking
        try:
            log("Extracting people...", level="processing")
            start_time = datetime.now()

            if args.local:
                extracted_people = ollama_extract_people(
                    article_content, model="qwq", domain=args.domain
                )
            else:
                extracted_people = gemini_extract_people(
                    article_content, domain=args.domain
                )

            # Track reflection attempts for people
            # Try to extract reflection history if available
            reflection_history = []
            reflection_attempts = 1

            # Check if the response has reflection history attribute
            if hasattr(extracted_people, "reflection_history"):
                reflection_history = extracted_people.reflection_history
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

                if args.verbose and reflection_history:
                    log(f"Reflection history for people extraction:", level="debug")
                    for i, reflection in enumerate(reflection_history):
                        passed = reflection.get("passed", False)
                        feedback = reflection.get("feedback", "No feedback")
                        log(
                            f"  Attempt {i + 1}: {'✓' if passed else '✗'} {feedback}",
                            level="debug",
                        )

            # For entity extraction modules that return merged results
            if (
                isinstance(extracted_people, dict)
                and "reflection_history" in extracted_people
            ):
                reflection_history = extracted_people["reflection_history"]
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

            duration = (datetime.now() - start_time).total_seconds()
            log(
                f"Extracted {len(extracted_people)} people in {duration:.2f}s",
                level="success",
            )

            if reflection_attempts > 1:
                log(
                    f"Required {reflection_attempts} reflection iterations",
                    level="debug",
                )

            # Update reflection metadata
            processing_metadata["reflection_attempts"]["people"] = {
                "attempts": reflection_attempts,
                "success": bool(extracted_people),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
            }

            # Update summary counts
            processing_metadata["reflection_summary"]["total_attempts"] += (
                reflection_attempts
            )
            processing_metadata["reflection_summary"]["successful_attempts"] += 1

        except Exception as e:
            error = EntityExtractionError(
                "People extraction failed",
                "people",
                article_id,
                {"original_error": str(e), "model_type": model_type},
            )
            handle_article_processing_error(article_id, "people_extraction", error)
            extracted_people = []
            processing_metadata["reflection_attempts"]["people"] = {
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
                "attempts": 1,
                "success": False,
            }
            processing_metadata["reflection_summary"]["failed_attempts"] += 1

        # Extract organizations with enhanced reflection tracking
        try:
            log("Extracting organizations...", level="processing")
            start_time = datetime.now()

            if args.local:
                extracted_orgs = ollama_extract_organizations(
                    article_content, model="qwq", domain=args.domain
                )
            else:
                extracted_orgs = gemini_extract_organizations(
                    article_content, domain=args.domain
                )

            # Track reflection attempts for organizations
            reflection_history = []
            reflection_attempts = 1

            # Check if the response has reflection history attribute
            if hasattr(extracted_orgs, "reflection_history"):
                reflection_history = extracted_orgs.reflection_history
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

                if args.verbose and reflection_history:
                    log(
                        f"Reflection history for organizations extraction:",
                        level="debug",
                    )
                    for i, reflection in enumerate(reflection_history):
                        passed = reflection.get("passed", False)
                        feedback = reflection.get("feedback", "No feedback")
                        log(
                            f"  Attempt {i + 1}: {'✓' if passed else '✗'} {feedback}",
                            level="debug",
                        )

            # For entity extraction modules that return merged results
            if (
                isinstance(extracted_orgs, dict)
                and "reflection_history" in extracted_orgs
            ):
                reflection_history = extracted_orgs["reflection_history"]
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

            duration = (datetime.now() - start_time).total_seconds()
            log(
                f"Extracted {len(extracted_orgs)} organizations in {duration:.2f}s",
                level="success",
            )

            if reflection_attempts > 1:
                log(
                    f"Required {reflection_attempts} reflection iterations",
                    level="debug",
                )

            # Update reflection metadata
            processing_metadata["reflection_attempts"]["organizations"] = {
                "attempts": reflection_attempts,
                "success": bool(extracted_orgs),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
            }

            # Update summary counts
            processing_metadata["reflection_summary"]["total_attempts"] += (
                reflection_attempts
            )
            processing_metadata["reflection_summary"]["successful_attempts"] += 1

        except Exception as e:
            error = EntityExtractionError(
                "Organizations extraction failed",
                "organizations",
                article_id,
                {"original_error": str(e), "model_type": model_type},
            )
            handle_article_processing_error(
                article_id, "organizations_extraction", error
            )
            extracted_orgs = []
            processing_metadata["reflection_attempts"]["organizations"] = {
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
                "attempts": 1,
                "success": False,
            }
            processing_metadata["reflection_summary"]["failed_attempts"] += 1

        # Extract locations with enhanced reflection tracking
        try:
            log("Extracting locations...", level="processing")
            start_time = datetime.now()

            if args.local:
                extracted_locs = ollama_extract_locations(
                    article_content, model="qwq", domain=args.domain
                )
            else:
                extracted_locs = gemini_extract_locations(
                    article_content, domain=args.domain
                )

            # Track reflection attempts for locations
            reflection_history = []
            reflection_attempts = 1

            # Check if the response has reflection history attribute
            if hasattr(extracted_locs, "reflection_history"):
                reflection_history = extracted_locs.reflection_history
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

                if args.verbose and reflection_history:
                    log(f"Reflection history for locations extraction:", level="debug")
                    for i, reflection in enumerate(reflection_history):
                        passed = reflection.get("passed", False)
                        feedback = reflection.get("feedback", "No feedback")
                        log(
                            f"  Attempt {i + 1}: {'✓' if passed else '✗'} {feedback}",
                            level="debug",
                        )

            # For entity extraction modules that return merged results
            if (
                isinstance(extracted_locs, dict)
                and "reflection_history" in extracted_locs
            ):
                reflection_history = extracted_locs["reflection_history"]
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

            duration = (datetime.now() - start_time).total_seconds()
            log(
                f"Extracted {len(extracted_locs)} locations in {duration:.2f}s",
                level="success",
            )

            if reflection_attempts > 1:
                log(
                    f"Required {reflection_attempts} reflection iterations",
                    level="debug",
                )

            # Update reflection metadata
            processing_metadata["reflection_attempts"]["locations"] = {
                "attempts": reflection_attempts,
                "success": bool(extracted_locs),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
            }

            # Update summary counts
            processing_metadata["reflection_summary"]["total_attempts"] += (
                reflection_attempts
            )
            processing_metadata["reflection_summary"]["successful_attempts"] += 1

        except Exception as e:
            error = EntityExtractionError(
                "Locations extraction failed",
                "locations",
                article_id,
                {"original_error": str(e), "model_type": model_type},
            )
            handle_article_processing_error(article_id, "locations_extraction", error)
            extracted_locs = []
            processing_metadata["reflection_attempts"]["locations"] = {
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
                "attempts": 1,
                "success": False,
            }
            processing_metadata["reflection_summary"]["failed_attempts"] += 1

        # Extract events with enhanced reflection tracking
        try:
            log("Extracting events...", level="processing")
            start_time = datetime.now()

            if args.local:
                extracted_events = ollama_extract_events(
                    article_content, model="qwq", domain=args.domain
                )
            else:
                extracted_events = gemini_extract_events(
                    article_content, domain=args.domain
                )

            # Track reflection attempts for events
            reflection_history = []
            reflection_attempts = 1

            # Check if the response has reflection history attribute
            if hasattr(extracted_events, "reflection_history"):
                reflection_history = extracted_events.reflection_history
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

                if args.verbose and reflection_history:
                    log(f"Reflection history for events extraction:", level="debug")
                    for i, reflection in enumerate(reflection_history):
                        passed = reflection.get("passed", False)
                        feedback = reflection.get("feedback", "No feedback")
                        log(
                            f"  Attempt {i + 1}: {'✓' if passed else '✗'} {feedback}",
                            level="debug",
                        )

            # For entity extraction modules that return merged results
            if (
                isinstance(extracted_events, dict)
                and "reflection_history" in extracted_events
            ):
                reflection_history = extracted_events["reflection_history"]
                reflection_attempts = (
                    len(reflection_history) if reflection_history else 1
                )

            duration = (datetime.now() - start_time).total_seconds()
            log(
                f"Extracted {len(extracted_events)} events in {duration:.2f}s",
                level="success",
            )

            if reflection_attempts > 1:
                log(
                    f"Required {reflection_attempts} reflection iterations",
                    level="debug",
                )

            # Update reflection metadata
            processing_metadata["reflection_attempts"]["events"] = {
                "attempts": reflection_attempts,
                "success": bool(extracted_events),
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": duration,
            }

            # Update summary counts
            processing_metadata["reflection_summary"]["total_attempts"] += (
                reflection_attempts
            )
            processing_metadata["reflection_summary"]["successful_attempts"] += 1

        except Exception as e:
            error = EntityExtractionError(
                "Events extraction failed",
                "events",
                article_id,
                {"original_error": str(e), "model_type": model_type},
            )
            handle_article_processing_error(article_id, "events_extraction", error)
            extracted_events = []
            processing_metadata["reflection_attempts"]["events"] = {
                "error": str(error),
                "timestamp": datetime.now().isoformat(),
                "attempts": 1,
                "success": False,
            }
            processing_metadata["reflection_summary"]["failed_attempts"] += 1

        # Convert Pydantic to dict if needed
        def to_dict_list(items):
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

        people_dicts = to_dict_list(extracted_people)
        org_dicts = to_dict_list(extracted_orgs)
        loc_dicts = to_dict_list(extracted_locs)
        event_dicts = to_dict_list(extracted_events)

        log("Merging extracted entities...", level="processing")

        # Merge people
        try:
            merge_start = datetime.now()
            merge_people(
                people_dicts,
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
                f"Merged {len(people_dicts)} people in {merge_duration:.2f}s",
                level="success",
            )
        except Exception as e:
            log("Error merging people", level="error", exception=e)
            processing_metadata["error_people"] = str(e)

        # Merge organizations
        try:
            merge_start = datetime.now()
            merge_organizations(
                org_dicts,
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
                f"Merged {len(org_dicts)} organizations in {merge_duration:.2f}s",
                level="success",
            )
        except Exception as e:
            log("Error merging organizations", level="error", exception=e)
            processing_metadata["error_organizations"] = str(e)

        # Merge locations
        try:
            merge_start = datetime.now()
            merge_locations(
                loc_dicts,
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
                f"Merged {len(loc_dicts)} locations in {merge_duration:.2f}s",
                level="success",
            )
        except Exception as e:
            log("Error merging locations", level="error", exception=e)
            processing_metadata["error_locations"] = str(e)

        # Merge events
        try:
            merge_start = datetime.now()
            merge_events(
                event_dicts,
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
                f"Merged {len(event_dicts)} events in {merge_duration:.2f}s",
                level="success",
            )
        except Exception as e:
            log("Error merging events", level="error", exception=e)
            processing_metadata["error_events"] = str(e)

        # Because reflection histories are stored in each entity, we can
        # also keep a record here if desired.
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
            "people": len(people_dicts),
            "organizations": len(org_dicts),
            "locations": len(loc_dicts),
            "events": len(event_dicts),
            "total": len(people_dicts)
            + len(org_dicts)
            + len(loc_dicts)
            + len(event_dicts),
        }
        processing_metadata["processing_time_seconds"] = processing_time

        processed_rows.append(row)
        processed_count += 1

        # Log reflection summary
        log(f"Reflection summary for article #{row_index}:", level="info")
        log(f"  Total reflection attempts: {total_reflection_attempts}", level="info")

        # Only log detailed reflection stats in verbose mode or if there were multiple attempts
        if (
            args.verbose or total_reflection_attempts > 4
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


if __name__ == "__main__":
    main()
