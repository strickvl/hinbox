#!/usr/bin/env python3
"""
Script merging logic from run.py and extract_entities.py, iterating over articles in
data/raw_sources/miami_herald_articles.jsonl, extracting entities (people, events,
locations, organizations) via the Gemini or local approach, then merging results
into the data/entities/*.jsonl files as in extract_entities.py.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# We'll not forcibly re-check relevance unless desired, but we keep the option in code.
# Use constants from constants.py
from src.constants import (
    ARTICLES_PATH,
    EVENTS_OUTPUT_PATH,
    GEMINI_MODEL,
    LOCATIONS_OUTPUT_PATH,
    OLLAMA_MODEL,
    ORGANIZATIONS_OUTPUT_PATH,
    OUTPUT_DIR,
    PEOPLE_OUTPUT_PATH,
)
from src.events import gemini_extract_events, ollama_extract_events
from src.locations import gemini_extract_locations, ollama_extract_locations
from src.organizations import (
    gemini_extract_organizations,
    ollama_extract_organizations,
)
from src.people import gemini_extract_people, ollama_extract_people
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
    people_table = pa.Table.from_pylist(people_list)
    pq.write_table(people_table, PEOPLE_OUTPUT_PATH)

    # Events
    events_list = sorted(
        entities["events"].values(), key=lambda x: x["title"]
    )  # Sort by title
    events_table = pa.Table.from_pylist(events_list)
    pq.write_table(events_table, EVENTS_OUTPUT_PATH)

    # Locations
    locations_list = sorted(entities["locations"].values(), key=lambda x: x["name"])
    locations_table = pa.Table.from_pylist(locations_list)
    pq.write_table(locations_table, LOCATIONS_OUTPUT_PATH)

    # Organizations
    organizations_list = sorted(
        entities["organizations"].values(), key=lambda x: x["name"]
    )
    organizations_table = pa.Table.from_pylist(organizations_list)
    pq.write_table(organizations_table, ORGANIZATIONS_OUTPUT_PATH)


def write_entity_to_file(
    entity_type: str, entity_key: Any, entity_data: Dict[str, Any]
):
    """
    Write a single entity to its respective Parquet file. This function now uses
    an append-only approach for simplicity, reading the existing data, adding the new
    or updated entity, and rewriting the entire file.

    Args:
        entity_type: "people", "events", "locations", or "organizations"
        entity_key: Key to identify entity (name for people, tuple for others)
        entity_data: Entity data to write
    """
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
    This ensures we have the most up-to-date data when checking for duplicates.
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
    """
    For each person in `extracted_people`, check if they exist in `entities["people"]`.
    If yes, append the article if not present and update profile.
    If no, create new entry with initial profile.
    """
    from src.profiles import create_profile, update_profile

    # Reload entities to get latest state
    current_entities = reload_entities()
    entities["people"] = current_entities["people"]

    for p in extracted_people:
        person_name = p.get("name", "")
        if not person_name:
            continue

        entity_updated = False

        if person_name in entities["people"]:
            existing_person = entities["people"][person_name]
            # Check if article already listed
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

            # Write the updated entity to file if it was modified
            if entity_updated:
                write_entity_to_file("people", person_name, existing_person)
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
                "extraction_timestamp": extraction_timestamp,
            }

            entities["people"][person_name] = new_person

            # Write the new entity to file
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
    """Merge location logic with profile handling."""
    from src.profiles import create_profile, update_profile

    # Reload entities to get latest state
    current_entities = reload_entities()
    entities["locations"] = current_entities["locations"]

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

                # Update profile with new information
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

            # Write the updated entity to file if it was modified
            if entity_updated:
                write_entity_to_file("locations", location_key, existing_loc)
                console.print(
                    f"[blue]Updated location entity saved to file:[/] {loc_name}"
                )
        else:
            # Create new entry with profile
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

            # Write the new entity to file
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
    """Merge organization logic with profile handling."""
    from src.profiles import create_profile, update_profile

    # Reload entities to get latest state
    current_entities = reload_entities()
    entities["organizations"] = current_entities["organizations"]

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

                # Update profile with new information
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

            # Write the updated entity to file if it was modified
            if entity_updated:
                write_entity_to_file("organizations", org_key, existing_org)
                console.print(
                    f"[blue]Updated organization entity saved to file:[/] {org_name}"
                )
        else:
            # Create new entry with profile
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

            # Write the new entity to file
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
    """Merge events logic with profile handling."""
    from src.profiles import create_profile, update_profile

    # Reload entities to get latest state
    current_entities = reload_entities()
    entities["events"] = current_entities["events"]

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

                # Update profile with new information
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

            # Write the updated entity to file if it was modified
            if entity_updated:
                write_entity_to_file("events", event_key, existing_event)
                console.print(
                    f"[blue]Updated event entity saved to file:[/] {event_title}"
                )
        else:
            # Create new entry with profile
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

            # Write the new entity to file
            write_entity_to_file("events", event_key, new_event)
            console.print(f"[green]New event entity saved to file:[/] {event_title}")


def main():
    print("Starting script...")  # Basic debug

    parser = argparse.ArgumentParser(
        description="Process articles, extract entities, and merge into data/entities/*.jsonl"
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
        help="If specified, we do a relevance check on each article. If not relevant, skip extraction.",
    )
    parser.add_argument(
        "--articles-path",
        type=str,
        default=ARTICLES_PATH,
        help="Path to the raw articles JSONL file",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Process articles even if they've been processed before",
    )
    args = parser.parse_args()

    console.print(
        f"[bold blue]Arguments parsed:[/] limit={args.limit}, local={args.local}, relevance_check={args.relevance_check}, force_reprocess={args.force_reprocess}"
    )

    # Make sure the output directory for entity JSONL files exists
    ensure_dir(OUTPUT_DIR)
    console.print(f"Ensured output directory exists: {OUTPUT_DIR}")  # Debug directory

    # Load existing entities into dictionaries
    console.print("Loading existing entities...")
    entities = load_existing_entities()
    console.print(
        f"Loaded entities: {len(entities['people'])} people, {len(entities['events'])} events"
    )

    # Read articles from the specified path
    article_count = 0
    processed_count = 0
    skipped_relevance_count = 0
    skipped_already_processed = 0

    console.print(f"Opening articles file: {args.articles_path}")  # Debug file opening
    model_type = "ollama" if args.local else "gemini"
    specific_model = OLLAMA_MODEL if args.local else GEMINI_MODEL
    try:
        if not os.path.exists(args.articles_path):
            console.print(f"ERROR: Articles file not found at {args.articles_path}")
            return

        # Create a temporary file to write processed articles
        temp_file = args.articles_path + ".tmp"

        with (
            open(args.articles_path, "r") as input_file,
            open(temp_file, "w") as output_file,
        ):
            for line in input_file:
                if args.limit is not None and article_count >= args.limit:
                    # Copy remaining unprocessed articles to temp file
                    output_file.write(line)
                    continue

                article_count += 1
                console.print(f"\nProcessing article #{article_count}")

                try:
                    article = json.loads(line)
                    article_id = article.get("id", f"article_{article_count}")
                    article_title = article.get("title", "")
                    article_url = article.get("url", "")
                    article_published_date = article.get("published_date", "")
                    article_content = article.get("content", "")

                    # Check if article has already been processed
                    processing_metadata = article.get("processing_metadata", {})
                    if not args.force_reprocess and processing_metadata.get(
                        "processed"
                    ):
                        console.print(
                            "[yellow]Article already processed, skipping...[/]"
                        )
                        skipped_already_processed += 1
                        output_file.write(line)  # Write unchanged article
                        continue

                    console.print(f"Article title: {article_title}")
                except Exception as e:
                    console.print(f"Could not parse JSONL line: {e}")
                    output_file.write(line)  # Write problematic line unchanged
                    continue

                if not article_content:
                    console.print("Warning: Article has no content, skipping")
                    output_file.write(line)  # Write unchanged article
                    continue

                # Initialize or update processing metadata
                if "processing_metadata" not in article:
                    article["processing_metadata"] = {}

                processing_metadata = article["processing_metadata"]
                processing_metadata.update(
                    {
                        "processing_started": datetime.now().isoformat(),
                        "model_type": model_type,
                        "specific_model": specific_model,
                    }
                )

                # Perform relevance check if requested
                if args.relevance_check:
                    console.print("Performing relevance check...")
                    try:
                        if args.local:
                            relevance_result = ollama_check_relevance(
                                article_content, model="qwq"
                            )
                        else:
                            relevance_result = gemini_check_relevance(article_content)

                        console.print(
                            f"Relevance check result: {'[green]RELEVANT[/]' if relevance_result.is_relevant else '[red]NOT RELEVANT[/]'}"
                        )
                        console.print(f"Reason: {relevance_result.reason}")

                        if not relevance_result.is_relevant:
                            console.print("Skipping article as it's not relevant")
                            skipped_relevance_count += 1
                            output_file.write(json.dumps(article) + "\n")
                            continue
                    except Exception as e:
                        console.print(f"[red]Error during relevance check: {e}[/]")
                        console.print(
                            "Proceeding with extraction despite relevance check failure"
                        )

                print("Starting entity extraction...")
                extraction_timestamp = datetime.now().isoformat()

                # Extract people
                try:
                    print("Extracting people...")
                    if args.local:
                        extracted_people = ollama_extract_people(
                            article_content, model="qwq"
                        )
                    else:
                        extracted_people = gemini_extract_people(article_content)
                    print(f"Found {len(extracted_people)} people")
                except Exception as e:
                    print(f"Error extracting people: {e}")
                    extracted_people = []

                # Extract organizations
                try:
                    print("\nExtracting organizations...")
                    if args.local:
                        extracted_orgs = ollama_extract_organizations(
                            article_content, model="qwq"
                        )
                    else:
                        extracted_orgs = gemini_extract_organizations(article_content)
                    print(f"Found {len(extracted_orgs)} organizations")
                except Exception as e:
                    print(f"Error extracting organizations: {e}")
                    extracted_orgs = []

                # Extract locations
                try:
                    print("\nExtracting locations...")
                    if args.local:
                        extracted_locs = ollama_extract_locations(
                            article_content, model="qwq"
                        )
                    else:
                        extracted_locs = gemini_extract_locations(article_content)
                    print(f"Found {len(extracted_locs)} locations")
                except Exception as e:
                    print(f"Error extracting locations: {e}")
                    extracted_locs = []

                # Extract events
                try:
                    print("\nExtracting events...")
                    if args.local:
                        extracted_events = ollama_extract_events(
                            article_content, model="qwq"
                        )
                    else:
                        extracted_events = gemini_extract_events(article_content)
                    print(f"Found {len(extracted_events)} events")
                except Exception as e:
                    print(f"Error extracting events: {e}")
                    extracted_events = []

                # Now merge all extracted entities
                print("\nMerging extracted entities...")
                try:
                    # Convert Pydantic models to dictionaries
                    people_dicts = [
                        p.model_dump() if hasattr(p, "model_dump") else p.dict()
                        for p in extracted_people
                    ]
                    org_dicts = [
                        o.model_dump() if hasattr(o, "model_dump") else o.dict()
                        for o in extracted_orgs
                    ]
                    loc_dicts = [
                        l.model_dump() if hasattr(l, "model_dump") else l.dict()
                        for l in extracted_locs
                    ]
                    event_dicts = [
                        e.model_dump() if hasattr(e, "model_dump") else e.dict()
                        for e in extracted_events
                    ]

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

                    # Update processing metadata after successful processing
                    processing_metadata.update(
                        {
                            "processed": True,
                            "processing_completed": datetime.now().isoformat(),
                            "entities_extracted": {
                                "people": len(extracted_people),
                                "organizations": len(extracted_orgs),
                                "locations": len(extracted_locs),
                                "events": len(extracted_events),
                            },
                        }
                    )

                    # Write processed article immediately
                    output_file.write(json.dumps(article) + "\n")

                    processed_count += 1
                    console.print(f"Successfully processed article #{article_count}")
                except Exception as e:
                    console.print(f"Error merging entities: {e}")
                    # Write the article with error information
                    processing_metadata["error"] = str(e)
                    output_file.write(json.dumps(article) + "\n")

        # Replace original file with processed file
        os.replace(temp_file, args.articles_path)

    except Exception as e:
        console.print(f"Error processing articles: {e}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            os.unlink(temp_file)

    # Write final results - this is now redundant since we write incrementally,
    # but we'll keep it as a final verification step
    console.print("\nVerifying all entities are saved to files...")
    write_entities_to_files(entities)

    console.print(
        f"\nProcessing complete. Articles read: {article_count}, "
        f"processed: {processed_count}, "
        f"skipped due to relevance check: {skipped_relevance_count}, "
        f"skipped already processed: {skipped_already_processed}"
    )
    console.print(f"Final entity counts:")
    console.print(f"- People: {len(entities['people'])}")
    console.print(f"- Organizations: {len(entities['organizations'])}")
    console.print(f"- Locations: {len(entities['locations'])}")
    console.print(f"- Events: {len(entities['events'])}")


if __name__ == "__main__":
    main()
