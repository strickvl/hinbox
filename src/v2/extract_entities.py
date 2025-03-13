#!/usr/bin/env python3
"""
Extract entities from processed articles and create separate JSONL files for each entity type.

This script reads the processed articles from a JSONL file and extracts people, events,
locations, and organizations into separate JSONL files.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Define paths
PROCESSED_ARTICLES_PATH = "data/processed/processed_articles.jsonl"
OUTPUT_DIR = "data/entities"

# Define output paths
PEOPLE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "people.jsonl")
EVENTS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "events.jsonl")
LOCATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "locations.jsonl")
ORGANIZATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "organizations.jsonl")


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): The directory path to check
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def process_articles(
    input_path: str, limit: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process articles from a JSONL file and extract entities.

    Args:
        input_path (str): Path to the input JSONL file
        limit (Optional[int]): Maximum number of articles to process

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary containing lists of extracted entities
    """
    people = []
    events = []
    locations = []
    organizations = []

    article_count = 0

    with open(input_path, "r") as f:
        for line in f:
            if limit is not None and article_count >= limit:
                break

            article = json.loads(line)
            article_id = article.get("id", f"article_{article_count}")
            article_title = article.get("title", "")
            article_url = article.get("url", "")
            article_published_date = article.get("published_date", "")

            # Check if the article has metadata and is relevant
            metadata = article.get("metadata", {})
            relevance_check = metadata.get("relevance_check", {})
            is_relevant = relevance_check.get("is_relevant", False)

            if is_relevant:
                # Get extraction timestamp
                extraction_timestamp = metadata.get(
                    "metadata_extraction_timestamp", datetime.now().isoformat()
                )

                # Process people
                for person in metadata.get("people", []):
                    people.append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                            "name": person.get("name", ""),
                            "type": person.get("type", ""),
                            "extraction_timestamp": extraction_timestamp,
                        }
                    )

                # Process events
                for event in metadata.get("events", []):
                    events.append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                            "title": event.get("title", ""),
                            "description": event.get("description", ""),
                            "event_type": event.get("event_type", ""),
                            "start_date": event.get("start_date", ""),
                            "end_date": event.get("end_date", ""),
                            "is_fuzzy_date": event.get("is_fuzzy_date", False),
                            "tags": event.get("tags", []),
                            "extraction_timestamp": extraction_timestamp,
                        }
                    )

                # Process locations
                for location in metadata.get("locations", []):
                    locations.append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                            "name": location.get("name", ""),
                            "type": location.get("type", ""),
                            "extraction_timestamp": extraction_timestamp,
                        }
                    )

                # Process organizations
                for organization in metadata.get("organizations", []):
                    organizations.append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                            "name": organization.get("name", ""),
                            "type": organization.get("type", ""),
                            "extraction_timestamp": extraction_timestamp,
                        }
                    )

            article_count += 1

    return {
        "people": people,
        "events": events,
        "locations": locations,
        "organizations": organizations,
    }


def write_entities_to_files(entities: Dict[str, List[Dict[str, Any]]]):
    """
    Write extracted entities to separate JSONL files.

    Args:
        entities (Dict[str, List[Dict[str, Any]]]): Dictionary containing lists of extracted entities
    """
    # Write people to file
    with open(PEOPLE_OUTPUT_PATH, "w") as f:
        for person in entities["people"]:
            f.write(json.dumps(person, cls=DateTimeEncoder) + "\n")

    # Write events to file
    with open(EVENTS_OUTPUT_PATH, "w") as f:
        for event in entities["events"]:
            f.write(json.dumps(event, cls=DateTimeEncoder) + "\n")

    # Write locations to file
    with open(LOCATIONS_OUTPUT_PATH, "w") as f:
        for location in entities["locations"]:
            f.write(json.dumps(location, cls=DateTimeEncoder) + "\n")

    # Write organizations to file
    with open(ORGANIZATIONS_OUTPUT_PATH, "w") as f:
        for organization in entities["organizations"]:
            f.write(json.dumps(organization, cls=DateTimeEncoder) + "\n")


def main():
    """Main function to extract entities from processed articles."""
    parser = argparse.ArgumentParser(
        description="Extract entities from processed articles and create separate JSONL files"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=PROCESSED_ARTICLES_PATH,
        help=f"Path to the input JSONL file (default: {PROCESSED_ARTICLES_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to store output files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of articles to process (default: process all)",
    )
    args = parser.parse_args()

    # Update paths if necessary
    global \
        PEOPLE_OUTPUT_PATH, \
        EVENTS_OUTPUT_PATH, \
        LOCATIONS_OUTPUT_PATH, \
        ORGANIZATIONS_OUTPUT_PATH
    if args.output_dir != OUTPUT_DIR:
        PEOPLE_OUTPUT_PATH = os.path.join(args.output_dir, "people.jsonl")
        EVENTS_OUTPUT_PATH = os.path.join(args.output_dir, "events.jsonl")
        LOCATIONS_OUTPUT_PATH = os.path.join(args.output_dir, "locations.jsonl")
        ORGANIZATIONS_OUTPUT_PATH = os.path.join(args.output_dir, "organizations.jsonl")

    # Ensure output directory exists
    ensure_dir(args.output_dir)

    # Process articles and extract entities
    print(f"Processing articles from {args.input}")
    entities = process_articles(args.input, args.limit)

    # Write entities to files
    write_entities_to_files(entities)

    # Print summary
    print(f"Extracted {len(entities['people'])} people, saved to {PEOPLE_OUTPUT_PATH}")
    print(f"Extracted {len(entities['events'])} events, saved to {EVENTS_OUTPUT_PATH}")
    print(
        f"Extracted {len(entities['locations'])} locations, saved to {LOCATIONS_OUTPUT_PATH}"
    )
    print(
        f"Extracted {len(entities['organizations'])} organizations, saved to {ORGANIZATIONS_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
