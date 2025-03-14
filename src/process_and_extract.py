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
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.embeddings import embed_text

console = Console()

from src.constants import (
    ARTICLES_PATH,
    CLOUD_EMBEDDING_MODEL,
    EVENTS_OUTPUT_PATH,
    GEMINI_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LOCATIONS_OUTPUT_PATH,
    OLLAMA_MODEL,
    ORGANIZATIONS_OUTPUT_PATH,
    OUTPUT_DIR,
    PEOPLE_OUTPUT_PATH,
)
from src.events import gemini_extract_events, ollama_extract_events
from src.locations import gemini_extract_locations, ollama_extract_locations
from src.organizations import gemini_extract_organizations, ollama_extract_organizations
from src.people import gemini_extract_people, ollama_extract_people
from src.profiles import create_profile, update_profile
from src.relevance import gemini_check_relevance, ollama_check_relevance


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
    Write updated entities back to Parquet files.
    The incoming 'entities' has keys: people, events, locations, organizations
    Each is a dict, so we need to convert them to a list and sort them, etc.
    """
    # People
    people_list = sorted(entities["people"].values(), key=lambda x: x["name"])
    if people_list:
        people_table = pa.Table.from_pylist(people_list)
        pq.write_table(people_table, PEOPLE_OUTPUT_PATH)
    else:
        # If empty, remove any existing file to avoid stale data
        if os.path.exists(PEOPLE_OUTPUT_PATH):
            os.remove(PEOPLE_OUTPUT_PATH)

    # Events
    events_list = sorted(entities["events"].values(), key=lambda x: x["title"])
    if events_list:
        events_table = pa.Table.from_pylist(events_list)
        pq.write_table(events_table, EVENTS_OUTPUT_PATH)
    else:
        if os.path.exists(EVENTS_OUTPUT_PATH):
            os.remove(EVENTS_OUTPUT_PATH)

    # Locations
    locations_list = sorted(entities["locations"].values(), key=lambda x: x["name"])
    if locations_list:
        locations_table = pa.Table.from_pylist(locations_list)
        pq.write_table(locations_table, LOCATIONS_OUTPUT_PATH)
    else:
        if os.path.exists(LOCATIONS_OUTPUT_PATH):
            os.remove(LOCATIONS_OUTPUT_PATH)

    # Organizations
    organizations_list = sorted(
        entities["organizations"].values(), key=lambda x: x["name"]
    )
    if organizations_list:
        organizations_table = pa.Table.from_pylist(organizations_list)
        pq.write_table(organizations_table, ORGANIZATIONS_OUTPUT_PATH)
    else:
        if os.path.exists(ORGANIZATIONS_OUTPUT_PATH):
            os.remove(ORGANIZATIONS_OUTPUT_PATH)


def write_entity_to_file(
    entity_type: str, entity_key: Any, entity_data: Dict[str, Any]
):
    """
    Write a single entity to its respective Parquet file. This function now uses
    an append/update approach for simplicity:
      - reads existing data from the file
      - updates or appends the entity
      - writes it all back

    Args:
        entity_type: "people", "events", "locations", or "organizations"
        entity_key: Key to identify entity (name for people, tuple for others)
        entity_data: Entity data to write
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if entity_type == "people":
        output_path = PEOPLE_OUTPUT_PATH
    elif entity_type == "events":
        output_path = EVENTS_OUTPUT_PATH
    elif entity_type == "locations":
        output_path = LOCATIONS_OUTPUT_PATH
    elif entity_type == "organizations":
        output_path = ORGANIZATIONS_OUTPUT_PATH
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

    entities = []
    entity_found = False

    if os.path.exists(output_path):
        table = pq.read_table(output_path)
        entities = table.to_pylist()

        for i, entity in enumerate(entities):
            if entity_type == "people" and entity.get("name") == entity_key:
                entities[i] = entity_data  # Update existing entity
                entity_found = True
                break
            elif (
                entity_type == "events"
                and (entity.get("title"), entity.get("start_date", "")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break
            elif (
                entity_type in ["locations", "organizations"]
                and (entity.get("name"), entity.get("type", "")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break

    if not entity_found:
        entities.append(entity_data)  # Append new entity

    # Sort entities appropriately
    if entity_type == "people":
        entities.sort(key=lambda x: x["name"])
    elif entity_type == "events":
        entities.sort(key=lambda x: x["title"])
    elif entity_type == "locations":
        entities.sort(key=lambda x: x["name"])
    elif entity_type == "organizations":
        entities.sort(key=lambda x: x["name"])

    # Write all entities back to file
    new_table = pa.Table.from_pylist(entities)
    pq.write_table(new_table, output_path)


def reload_entities() -> Dict[str, Dict]:
    """
    Reload all entity files to get the latest state.
    """
    return load_existing_entities()


def merge_people(
    extracted_people: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    for p in extracted_people:
        person_name = p.get("name", "")
        if not person_name:
            continue

        entity_updated = False

        if person_name in entities["people"]:
            existing_person = entities["people"][person_name]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_person["articles"]
            )
            if not article_exists:
                existing_person["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile with new information
                if "profile" in existing_person:
                    console.print(
                        f"\n[yellow]Updating profile for person:[/] {person_name}"
                    )
                    existing_person["profile"] = update_profile(
                        "person",
                        person_name,
                        existing_person["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    embedding_model = (
                        LOCAL_EMBEDDING_MODEL
                        if model_type == "ollama"
                        else CLOUD_EMBEDDING_MODEL
                    )
                    existing_person["profile_embedding"] = embed_text(
                        existing_person["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_person["profile"]["text"]),
                            title=f"Updated Profile: {person_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for person:[/] {person_name}"
                    )
                    existing_person["profile"] = create_profile(
                        "person", person_name, article_content, article_id, model_type
                    )
                    console.print(
                        Panel(
                            Markdown(existing_person["profile"]["text"]),
                            title=f"New Profile: {person_name}",
                            border_style="green",
                        )
                    )

            # Update extraction timestamp to earliest
            existing_timestamp = existing_person.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_person["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("people", person_name, existing_person)
                entities["people"][person_name] = existing_person
                console.print(
                    f"[blue]Updated person entity saved to file:[/] {person_name}"
                )
        else:
            # Create new entry with initial profile
            console.print(f"\n[green]Creating profile for new person:[/] {person_name}")
            profile = create_profile(
                "person", person_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {person_name}",
                    border_style="green",
                )
            )

            new_person = {
                "name": person_name,
                "type": p.get("type", ""),
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "profile_embedding": embed_text(profile["text"]),
                "extraction_timestamp": extraction_timestamp,
            }

            entities["people"][person_name] = new_person
            write_entity_to_file("people", person_name, new_person)
            console.print(f"[green]New person entity saved to file:[/] {person_name}")


def merge_locations(
    extracted_locations: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    for loc in extracted_locations:
        loc_name = loc.get("name", "")
        loc_type = loc.get("type", "")
        if not loc_name:
            continue
        location_key = (loc_name, loc_type)

        entity_updated = False

        if location_key in entities["locations"]:
            existing_loc = entities["locations"][location_key]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_loc["articles"]
            )
            if not article_exists:
                existing_loc["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_loc:
                    console.print(
                        f"\n[yellow]Updating profile for location:[/] {loc_name}"
                    )
                    existing_loc["profile"] = update_profile(
                        "location",
                        loc_name,
                        existing_loc["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    embedding_model = (
                        LOCAL_EMBEDDING_MODEL
                        if model_type == "ollama"
                        else CLOUD_EMBEDDING_MODEL
                    )
                    existing_loc["profile_embedding"] = embed_text(
                        existing_loc["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_loc["profile"]["text"]),
                            title=f"Updated Profile: {loc_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for location:[/] {loc_name}"
                    )
                    existing_loc["profile"] = create_profile(
                        "location", loc_name, article_content, article_id, model_type
                    )
                    console.print(
                        Panel(
                            Markdown(existing_loc["profile"]["text"]),
                            title=f"New Profile: {loc_name}",
                            border_style="green",
                        )
                    )

            existing_timestamp = existing_loc.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_loc["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("locations", location_key, existing_loc)
                entities["locations"][location_key] = existing_loc
                console.print(
                    f"[blue]Updated location entity saved to file:[/] {loc_name}"
                )
        else:
            # Create new entry
            console.print(f"\n[green]Creating profile for new location:[/] {loc_name}")
            profile = create_profile(
                "location", loc_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {loc_name}",
                    border_style="green",
                )
            )

            new_location = {
                "name": loc_name,
                "type": loc_type,
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
            }

            entities["locations"][location_key] = new_location
            write_entity_to_file("locations", location_key, new_location)
            console.print(f"[green]New location entity saved to file:[/] {loc_name}")


def merge_organizations(
    extracted_orgs: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    for org in extracted_orgs:
        org_name = org.get("name", "")
        org_type = org.get("type", "")
        if not org_name:
            continue
        org_key = (org_name, org_type)

        entity_updated = False

        if org_key in entities["organizations"]:
            existing_org = entities["organizations"][org_key]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_org["articles"]
            )
            if not article_exists:
                existing_org["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_org:
                    console.print(
                        f"\n[yellow]Updating profile for organization:[/] {org_name}"
                    )
                    existing_org["profile"] = update_profile(
                        "organization",
                        org_name,
                        existing_org["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    embedding_model = (
                        LOCAL_EMBEDDING_MODEL
                        if model_type == "ollama"
                        else CLOUD_EMBEDDING_MODEL
                    )
                    existing_org["profile_embedding"] = embed_text(
                        existing_org["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_org["profile"]["text"]),
                            title=f"Updated Profile: {org_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for organization:[/] {org_name}"
                    )
                    existing_org["profile"] = create_profile(
                        "organization",
                        org_name,
                        article_content,
                        article_id,
                        model_type,
                    )
                    console.print(
                        Panel(
                            Markdown(existing_org["profile"]["text"]),
                            title=f"New Profile: {org_name}",
                            border_style="green",
                        )
                    )

            existing_timestamp = existing_org.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_org["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("organizations", org_key, existing_org)
                entities["organizations"][org_key] = existing_org
                console.print(
                    f"[blue]Updated organization entity saved to file:[/] {org_name}"
                )
        else:
            # Create new entry
            console.print(
                f"\n[green]Creating profile for new organization:[/] {org_name}"
            )
            profile = create_profile(
                "organization", org_name, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {org_name}",
                    border_style="green",
                )
            )

            new_org = {
                "name": org_name,
                "type": org_type,
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
            }

            entities["organizations"][org_key] = new_org
            write_entity_to_file("organizations", org_key, new_org)
            console.print(
                f"[green]New organization entity saved to file:[/] {org_name}"
            )


def merge_events(
    extracted_events: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
):
    for e in extracted_events:
        event_title = e.get("title", "")
        event_start_date = e.get("start_date", "")
        event_key = (event_title, event_start_date)
        if not event_title:
            continue

        entity_updated = False

        if event_key in entities["events"]:
            existing_event = entities["events"][event_key]
            article_exists = any(
                a.get("article_id") == article_id for a in existing_event["articles"]
            )
            if not article_exists:
                existing_event["articles"].append(
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                )
                entity_updated = True

                # Update profile
                if "profile" in existing_event:
                    console.print(
                        f"\n[yellow]Updating profile for event:[/] {event_title}"
                    )
                    existing_event["profile"] = update_profile(
                        "event",
                        event_title,
                        existing_event["profile"],
                        article_content,
                        article_id,
                        model_type,
                    )
                    # Embed updated profile text using appropriate model
                    embedding_model = (
                        LOCAL_EMBEDDING_MODEL
                        if model_type == "ollama"
                        else CLOUD_EMBEDDING_MODEL
                    )
                    existing_event["profile_embedding"] = embed_text(
                        existing_event["profile"]["text"],
                        model_name=embedding_model,
                    )

                    console.print(
                        Panel(
                            Markdown(existing_event["profile"]["text"]),
                            title=f"Updated Profile: {event_title}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(
                        f"\n[green]Creating initial profile for event:[/] {event_title}"
                    )
                    existing_event["profile"] = create_profile(
                        "event", event_title, article_content, article_id, model_type
                    )
                    console.print(
                        Panel(
                            Markdown(existing_event["profile"]["text"]),
                            title=f"New Profile: {event_title}",
                            border_style="green",
                        )
                    )

            existing_timestamp = existing_event.get(
                "extraction_timestamp", extraction_timestamp
            )
            if existing_timestamp != min(existing_timestamp, extraction_timestamp):
                existing_event["extraction_timestamp"] = min(
                    existing_timestamp, extraction_timestamp
                )
                entity_updated = True

            if entity_updated:
                write_entity_to_file("events", event_key, existing_event)
                entities["events"][event_key] = existing_event
                console.print(
                    f"[blue]Updated event entity saved to file:[/] {event_title}"
                )
        else:
            # Create new entry
            console.print(f"\n[green]Creating profile for new event:[/] {event_title}")
            profile = create_profile(
                "event", event_title, article_content, article_id, model_type
            )
            console.print(
                Panel(
                    Markdown(profile["text"]),
                    title=f"New Profile: {event_title}",
                    border_style="green",
                )
            )

            new_event = {
                "title": event_title,
                "description": e.get("description", ""),
                "event_type": e.get("event_type", ""),
                "start_date": event_start_date,
                "end_date": e.get("end_date", ""),
                "is_fuzzy_date": e.get("is_fuzzy_date", False),
                "tags": e.get("tags", []),
                "profile": profile,
                "articles": [
                    {
                        "article_id": article_id,
                        "article_title": article_title,
                        "article_url": article_url,
                        "article_published_date": article_published_date,
                    }
                ],
                "extraction_timestamp": extraction_timestamp,
            }

            entities["events"][event_key] = new_event
            write_entity_to_file("events", event_key, new_event)
            console.print(f"[green]New event entity saved to file:[/] {event_title}")


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
    specific_model = OLLAMA_MODEL if args.local else GEMINI_MODEL

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

        # Extract people
        try:
            console.print("[blue]Extracting people...[/blue]")
            if args.local:
                extracted_people = ollama_extract_people(article_content, model="qwq")
            else:
                extracted_people = gemini_extract_people(article_content)
        except Exception as e:
            console.print(f"[red]Error extracting people: {e}[/red]")
            extracted_people = []

        # Extract organizations
        try:
            console.print("[blue]Extracting organizations...[/blue]")
            if args.local:
                extracted_orgs = ollama_extract_organizations(
                    article_content, model="qwq"
                )
            else:
                extracted_orgs = gemini_extract_organizations(article_content)
        except Exception as e:
            console.print(f"[red]Error extracting organizations: {e}[/red]")
            extracted_orgs = []

        # Extract locations
        try:
            console.print("[blue]Extracting locations...[/blue]")
            if args.local:
                extracted_locs = ollama_extract_locations(article_content, model="qwq")
            else:
                extracted_locs = gemini_extract_locations(article_content)
        except Exception as e:
            console.print(f"[red]Error extracting locations: {e}[/red]")
            extracted_locs = []

        # Extract events
        try:
            console.print("[blue]Extracting events...[/blue]")
            if args.local:
                extracted_events = ollama_extract_events(article_content, model="qwq")
            else:
                extracted_events = gemini_extract_events(article_content)
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
