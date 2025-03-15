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
from rich import print
from rich.console import Console

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
from src.locations import gemini_extract_locations, ollama_extract_locations
from src.merge import merge_events, merge_locations, merge_organizations, merge_people
from src.organizations import gemini_extract_organizations, ollama_extract_organizations
from src.people import gemini_extract_people, ollama_extract_people
from src.relevance import gemini_check_relevance, ollama_check_relevance
from src.utils import write_entity_to_file

console = Console()


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


def reload_entities() -> Dict[str, Dict]:
    """
    Reload all entity files to get the latest state.
    """
    return load_existing_entities()


def main():
    print("Starting script...")

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
    args = parser.parse_args()

    console.print(
        f"[bold blue]Arguments:[/] limit={args.limit}, local={args.local}, relevance_check={args.relevance_check}, force_reprocess={args.force_reprocess}"
    )

    ensure_dir(OUTPUT_DIR)

    # Load existing entities
    console.print("[green]Loading existing entities...[/green]")
    entities = load_existing_entities()

    model_type = "ollama" if args.local else "gemini"
    specific_model = OLLAMA_MODEL if args.local else CLOUD_MODEL

    # Check if articles parquet exists
    if not os.path.exists(args.articles_path):
        console.print(
            f"[red]ERROR: Articles file not found at {args.articles_path}[/red]"
        )
        return

    # Read entire Parquet into memory
    try:
        table = pq.read_table(args.articles_path)
        rows = table.to_pylist()
    except Exception as e:
        console.print(f"[red]Failed to read Parquet file: {e}[/red]")
        return

    article_count = len(rows)
    processed_count = 0
    skipped_relevance_count = 0
    skipped_already_processed = 0

    console.print(f"Loaded {article_count} articles from {args.articles_path}")

    processed_rows = []
    row_index = 0

    for row in rows:
        if row_index >= args.limit:
            # We've hit the limit; keep the rest unmodified
            processed_rows.append(row)
            row_index += 1
            continue

        row_index += 1
        console.print(f"\n[bold]Processing article #{row_index}[/bold]")

        article_id = row.get("id", f"article_{row_index}")
        article_title = row.get("title", "")
        article_url = row.get("url", "")
        article_published_date = row.get("published_date", "")
        article_content = row.get("content", "")

        # Initialize or check processing_metadata
        if "processing_metadata" not in row:
            row["processing_metadata"] = {}
        processing_metadata = row["processing_metadata"]

        # Initialize reflection metadata
        processing_metadata["reflection_attempts"] = {}

        # Skip if already processed and not forced
        if processing_metadata.get("processed") and not args.force_reprocess:
            console.print("[yellow]Article already processed, skipping...[/yellow]")
            skipped_already_processed += 1
            processed_rows.append(row)
            continue

        if not article_content:
            console.print(
                "[yellow]Article has no content, skipping extraction[/yellow]"
            )
            processed_rows.append(row)
            continue

        # Mark that we started processing
        processing_metadata["processing_started"] = datetime.now().isoformat()
        processing_metadata["model_type"] = model_type
        processing_metadata["specific_model"] = specific_model

        # Relevance check
        if args.relevance_check:
            console.print("[cyan]Performing relevance check...[/cyan]")
            try:
                if args.local:
                    relevance_result = ollama_check_relevance(
                        article_content, model="qwq"
                    )
                else:
                    relevance_result = gemini_check_relevance(article_content)
                if not relevance_result.is_relevant:
                    console.print("[red]Skipping article as it's not relevant[/red]")
                    processing_metadata["processed"] = False
                    processing_metadata["reason"] = relevance_result.reason
                    skipped_relevance_count += 1
                    processed_rows.append(row)
                    continue
                else:
                    console.print("[green]Article is relevant[/green]")
            except Exception as e:
                console.print(f"[red]Error during relevance check: {e}[/red]")
                console.print(
                    "[yellow]Proceeding with extraction despite error[/yellow]"
                )

        extraction_timestamp = datetime.now().isoformat()

        # Extract people with reflection tracking
        try:
            console.print("[blue]Extracting people...[/blue]")
            if args.local:
                extracted_people = ollama_extract_people(article_content, model="qwq")
            else:
                extracted_people = gemini_extract_people(article_content)

            # Track reflection attempts for people - just record success for now
            # since we don't have direct access to reflection history
            processing_metadata["reflection_attempts"]["people"] = {
                "attempts": 1,  # Basic tracking for now
                "success": bool(extracted_people),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            console.print(f"[red]Error extracting people: {e}[/red]")
            extracted_people = []
            processing_metadata["reflection_attempts"]["people"] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        # Extract organizations
        try:
            console.print("[blue]Extracting organizations...[/blue]")
            if args.local:
                extracted_orgs = ollama_extract_organizations(
                    article_content, model="qwq"
                )
            else:
                extracted_orgs = gemini_extract_organizations(article_content)

            processing_metadata["reflection_attempts"]["organizations"] = {
                "attempts": 1,
                "success": bool(extracted_orgs),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            console.print(f"[red]Error extracting organizations: {e}[/red]")
            extracted_orgs = []
            processing_metadata["reflection_attempts"]["organizations"] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        # Extract locations
        try:
            console.print("[blue]Extracting locations...[/blue]")
            if args.local:
                extracted_locs = ollama_extract_locations(article_content, model="qwq")
            else:
                extracted_locs = gemini_extract_locations(article_content)

            processing_metadata["reflection_attempts"]["locations"] = {
                "attempts": 1,
                "success": bool(extracted_locs),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            console.print(f"[red]Error extracting locations: {e}[/red]")
            extracted_locs = []
            processing_metadata["reflection_attempts"]["locations"] = {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        # Extract events
        try:
            console.print("[blue]Extracting events...[/blue]")
            if args.local:
                extracted_events = ollama_extract_events(article_content, model="qwq")
            else:
                extracted_events = gemini_extract_events(article_content)

            processing_metadata["reflection_attempts"]["events"] = {
                "attempts": 1,
                "success": bool(extracted_events),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            console.print(f"[red]Error extracting events: {e}[/red]")
            extracted_events = []

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

        console.print("[magenta]\nMerging extracted entities...[/magenta]")
        try:
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

            processing_metadata["processed"] = True
            processing_metadata["processing_completed"] = datetime.now().isoformat()
            processing_metadata["entities_extracted"] = {
                "people": len(people_dicts),
                "organizations": len(org_dicts),
                "locations": len(loc_dicts),
                "events": len(event_dicts),
            }

            processed_rows.append(row)
            processed_count += 1
            console.print(f"[green]Successfully processed article #{row_index}[/green]")
        except Exception as e:
            console.print(f"[red]Error merging entities: {e}[/red]")
            processing_metadata["error"] = str(e)
            processed_rows.append(row)

    # Write updated articles to a temp parquet file
    temp_file = args.articles_path + ".tmp.parquet"
    try:
        new_table = pa.Table.from_pylist(processed_rows)
        pq.write_table(new_table, temp_file)
        os.replace(temp_file, args.articles_path)
    except Exception as e:
        console.print(f"[red]Could not write updated articles to parquet: {e}[/red]")
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Write final entity sets to their parquet files
    console.print("[green]\nSaving updated entity tables...[/green]")
    write_entities_to_files(entities)

    console.print(
        f"\n[bold]Processing complete[/bold]. "
        f"Articles read: {article_count}, "
        f"processed: {processed_count}, "
        f"skipped (relevance): {skipped_relevance_count}, "
        f"skipped (already processed): {skipped_already_processed}"
    )
    console.print(f"\nFinal entity counts:")
    console.print(f"- People: {len(entities['people'])}")
    console.print(f"- Organizations: {len(entities['organizations'])}")
    console.print(f"- Locations: {len(entities['locations'])}")
    console.print(f"- Events: {len(entities['events'])}")


if __name__ == "__main__":
    main()
