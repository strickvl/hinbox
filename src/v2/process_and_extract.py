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

from pydantic import BaseModel
from rich import print

from src.v2.events import gemini_extract_events, ollama_extract_events
from src.v2.locations import gemini_extract_locations, ollama_extract_locations
from src.v2.organizations import (
    gemini_extract_organizations,
    ollama_extract_organizations,
)
from src.v2.people import gemini_extract_people, ollama_extract_people
from src.v2.relevance import gemini_check_relevance, ollama_check_relevance

# We'll not forcibly re-check relevance unless desired, but we keep the option in code.

# Some constants, consistent with the existing codebase
ARTICLES_PATH = "data/raw_sources/miami_herald_articles.jsonl"
OUTPUT_DIR = "data/entities"

PEOPLE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "people.jsonl")
EVENTS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "events.jsonl")
LOCATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "locations.jsonl")
ORGANIZATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "organizations.jsonl")


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)


def ensure_dir(directory: str):
    """
    Ensure that a directory exists, creating it if necessary.
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def load_existing_entities() -> Dict[str, Dict]:
    """
    Load existing entities from JSONL files if they exist.
    Returns a dict with keys: people, events, locations, organizations
    Each is a dictionary keyed by the relevant unique tuple or name.
    """
    people = {}
    events = {}
    locations = {}
    organizations = {}

    # People
    if os.path.exists(PEOPLE_OUTPUT_PATH):
        with open(PEOPLE_OUTPUT_PATH, "r") as f:
            for line in f:
                person = json.loads(line)
                people[person["name"]] = person

    # Events
    if os.path.exists(EVENTS_OUTPUT_PATH):
        with open(EVENTS_OUTPUT_PATH, "r") as f:
            for line in f:
                event = json.loads(line)
                # We key events by (title, start_date) as in extract_entities.py
                events[(event["title"], event.get("start_date", ""))] = event

    # Locations
    if os.path.exists(LOCATIONS_OUTPUT_PATH):
        with open(LOCATIONS_OUTPUT_PATH, "r") as f:
            for line in f:
                location = json.loads(line)
                # We key locations by (name, type)
                locations[(location["name"], location.get("type", ""))] = location

    # Organizations
    if os.path.exists(ORGANIZATIONS_OUTPUT_PATH):
        with open(ORGANIZATIONS_OUTPUT_PATH, "r") as f:
            for line in f:
                org = json.loads(line)
                # We key organizations by (name, type)
                organizations[(org["name"], org.get("type", ""))] = org

    return {
        "people": people,
        "events": events,
        "locations": locations,
        "organizations": organizations,
    }


def write_entities_to_files(entities: Dict[str, Dict]):
    """
    Write updated entities back to JSONL files.
    The incoming 'entities' has keys: people, events, locations, organizations
    Each is a dict, so we need to convert them to a list and sort them, etc.
    """
    # People
    with open(PEOPLE_OUTPUT_PATH, "w") as f:
        # sort by name
        for person_key in sorted(entities["people"].keys()):
            person_data = entities["people"][person_key]
            f.write(json.dumps(person_data, cls=DateTimeEncoder) + "\n")

    # Events
    with open(EVENTS_OUTPUT_PATH, "w") as f:
        # sort by event title
        sorted_keys = sorted(entities["events"].keys(), key=lambda k: k[0])
        for event_key in sorted_keys:
            event_data = entities["events"][event_key]
            f.write(json.dumps(event_data, cls=DateTimeEncoder) + "\n")

    # Locations
    with open(LOCATIONS_OUTPUT_PATH, "w") as f:
        sorted_keys = sorted(entities["locations"].keys(), key=lambda k: k[0])
        for loc_key in sorted_keys:
            loc_data = entities["locations"][loc_key]
            f.write(json.dumps(loc_data, cls=DateTimeEncoder) + "\n")

    # Organizations
    with open(ORGANIZATIONS_OUTPUT_PATH, "w") as f:
        sorted_keys = sorted(entities["organizations"].keys(), key=lambda k: k[0])
        for org_key in sorted_keys:
            org_data = entities["organizations"][org_key]
            f.write(json.dumps(org_data, cls=DateTimeEncoder) + "\n")


def merge_people(
    extracted_people: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    extraction_timestamp: str,
):
    """
    For each person in `extracted_people`, check if they exist in `entities["people"]`.
    If yes, append the article if not present.
    If no, create new entry.
    We do it exactly like extract_entities.py does.
    """
    for p in extracted_people:
        person_name = p.get("name", "")
        if not person_name:
            continue

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
            # Update extraction timestamp to earliest
            existing_timestamp = existing_person.get(
                "extraction_timestamp", extraction_timestamp
            )
            existing_person["extraction_timestamp"] = min(
                existing_timestamp, extraction_timestamp
            )
        else:
            # Add new
            entities["people"][person_name] = {
                "name": person_name,
                "type": p.get("type", ""),
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


def merge_events(
    extracted_events: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    extraction_timestamp: str,
):
    """
    Merge events logic. Key by (title, start_date).
    """
    for e in extracted_events:
        event_title = e.get("title", "")
        event_start_date = e.get("start_date", "")
        event_key = (event_title, event_start_date)
        if not event_title:
            continue

        if event_key in entities["events"]:
            existing_event = entities["events"][event_key]
            # Check if article already in there
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
            existing_timestamp = existing_event.get(
                "extraction_timestamp", extraction_timestamp
            )
            existing_event["extraction_timestamp"] = min(
                existing_timestamp, extraction_timestamp
            )
        else:
            # Add new
            entities["events"][event_key] = {
                "title": event_title,
                "description": e.get("description", ""),
                "event_type": e.get("event_type", ""),
                "start_date": event_start_date,
                "end_date": e.get("end_date", ""),
                "is_fuzzy_date": e.get("is_fuzzy_date", False),
                "tags": e.get("tags", []),
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


def merge_locations(
    extracted_locations: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    extraction_timestamp: str,
):
    """
    Merge location logic. Key by (name, type).
    """
    for loc in extracted_locations:
        loc_name = loc.get("name", "")
        loc_type = loc.get("type", "")
        if not loc_name:
            continue
        location_key = (loc_name, loc_type)
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
            existing_timestamp = existing_loc.get(
                "extraction_timestamp", extraction_timestamp
            )
            existing_loc["extraction_timestamp"] = min(
                existing_timestamp, extraction_timestamp
            )
        else:
            entities["locations"][location_key] = {
                "name": loc_name,
                "type": loc_type,
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


def merge_organizations(
    extracted_orgs: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    extraction_timestamp: str,
):
    """
    Merge organization logic. Key by (name, type).
    """
    for org in extracted_orgs:
        org_name = org.get("name", "")
        org_type = org.get("type", "")
        if not org_name:
            continue
        org_key = (org_name, org_type)
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
            existing_timestamp = existing_org.get(
                "extraction_timestamp", extraction_timestamp
            )
            existing_org["extraction_timestamp"] = min(
                existing_timestamp, extraction_timestamp
            )
        else:
            entities["organizations"][org_key] = {
                "name": org_name,
                "type": org_type,
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
    args = parser.parse_args()
    
    print(f"Arguments parsed: limit={args.limit}, local={args.local}")  # Debug args

    # Make sure the output directory for entity JSONL files exists
    ensure_dir(OUTPUT_DIR)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")  # Debug directory

    # Load existing entities into dictionaries
    print("Loading existing entities...")
    entities = load_existing_entities()
    print(f"Loaded entities: {len(entities['people'])} people, {len(entities['events'])} events")

    # Read articles from the specified path
    article_count = 0
    processed_count = 0

    print(f"Opening articles file: {args.articles_path}")  # Debug file opening
    try:
        if not os.path.exists(args.articles_path):
            print(f"ERROR: Articles file not found at {args.articles_path}")
            return
            
        with open(args.articles_path, "r") as f:
            print("Successfully opened articles file")
            for line in f:
                if args.limit is not None and article_count >= args.limit:
                    print(f"Reached limit of {args.limit} articles")
                    break

                article_count += 1
                print(f"\nProcessing article #{article_count}")

                try:
                    article = json.loads(line)
                    article_id = article.get("id", f"article_{article_count}")
                    article_title = article.get("title", "")
                    article_url = article.get("url", "")
                    article_published_date = article.get("published_date", "")
                    article_content = article.get("content", "")
                    
                    print(f"Article title: {article_title}")
                except Exception as e:
                    print(f"Could not parse JSONL line: {e}")
                    continue

                if not article_content:
                    print("Warning: Article has no content, skipping")
                    continue

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
                    people_dicts = [p.model_dump() if hasattr(p, 'model_dump') else p.dict() for p in extracted_people]
                    org_dicts = [o.model_dump() if hasattr(o, 'model_dump') else o.dict() for o in extracted_orgs]
                    loc_dicts = [l.model_dump() if hasattr(l, 'model_dump') else l.dict() for l in extracted_locs]
                    event_dicts = [e.model_dump() if hasattr(e, 'model_dump') else e.dict() for e in extracted_events]
                    
                    merge_people(
                        people_dicts,
                        entities,
                        article_id,
                        article_title,
                        article_url,
                        article_published_date,
                        extraction_timestamp,
                    )
                    merge_organizations(
                        org_dicts,
                        entities,
                        article_id,
                        article_title,
                        article_url,
                        article_published_date,
                        extraction_timestamp,
                    )
                    merge_locations(
                        loc_dicts,
                        entities,
                        article_id,
                        article_title,
                        article_url,
                        article_published_date,
                        extraction_timestamp,
                    )
                    merge_events(
                        event_dicts,
                        entities,
                        article_id,
                        article_title,
                        article_url,
                        article_published_date,
                        extraction_timestamp,
                    )
                    processed_count += 1
                    print(f"Successfully processed article #{article_count}")
                except Exception as e:
                    print(f"Error merging entities: {e}")

    except Exception as e:
        print(f"Error processing articles: {e}")

    # Write final results
    print("\nWriting extracted entities to files...")
    write_entities_to_files(entities)
    
    print(f"\nProcessing complete. Articles read: {article_count}, processed: {processed_count}")
    print(f"Final entity counts:")
    print(f"- People: {len(entities['people'])}")
    print(f"- Organizations: {len(entities['organizations'])}")
    print(f"- Locations: {len(entities['locations'])}")
    print(f"- Events: {len(entities['events'])}")

if __name__ == "__main__":
    main()
