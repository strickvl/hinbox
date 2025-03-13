import argparse
import json
import os
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Union

import litellm
from pydantic import BaseModel
from rich import print

from src.v2.events import gemini_extract_events, ollama_extract_events
from src.v2.locations import (
    gemini_extract_locations,
    ollama_extract_locations,
)
from src.v2.organizations import (
    gemini_extract_organizations,
    ollama_extract_organizations,
)
from src.v2.people import (
    gemini_extract_people,
    ollama_extract_people,
)
from src.v2.relevance import gemini_check_relevance, ollama_check_relevance
from src.v2.tags import gemini_extract_tags, ollama_extract_tags

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


ARTICLES_PATH = "data/raw_sources/miami_herald_articles.jsonl"

# Use a relative path for the output file
OUTPUT_PATH = "data/processed/processed_articles.jsonl"


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects and Enums."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.dict()
        return super().default(obj)


def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory (str): The directory path to check
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def model_to_dict(obj: Any) -> Union[Dict, List, Any]:
    """
    Convert a Pydantic model or a list of models to a dictionary or list of dictionaries.

    Args:
        obj: The object to convert

    Returns:
        The converted object
    """
    if hasattr(obj, "dict"):
        return obj.dict()
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, list):
        return [model_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: model_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj


def normalize_text(text: str) -> str:
    """
    Normalize text by removing accents and other diacritical marks.

    Args:
        text (str): The text to normalize

    Returns:
        str: The normalized text
    """
    if not text:
        return ""

    # Normalize to NFKD form (compatibility decomposition)
    # This separates base characters from diacritical marks
    normalized = unicodedata.normalize("NFKD", text)

    # Remove diacritical marks by keeping only ASCII characters
    result = "".join(c for c in normalized if not unicodedata.combining(c))

    return result


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract information about Guantánamo from articles"
    )
    parser.add_argument(
        "--local", action="store_true", help="Use only local models (spaCy and Ollama)"
    )
    parser.add_argument(
        "--people",
        action="store_true",
        help="Only extract and print people information",
    )
    parser.add_argument(
        "--places",
        action="store_true",
        help="Only extract and print location information",
    )
    parser.add_argument(
        "--orgs",
        action="store_true",
        help="Only extract and print organization information",
    )
    parser.add_argument(
        "--events",
        action="store_true",
        help="Only extract and print event information",
    )
    parser.add_argument(
        "--tags",
        action="store_true",
        help="Only extract and print article tags",
    )
    parser.add_argument(
        "--show-article",
        action="store_true",
        help="Show the article",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit the number of articles to process (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help=f"Path to the output file (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    # If no specific extraction is specified, extract all types
    extract_people = args.people or not (
        args.people or args.places or args.orgs or args.events or args.tags
    )
    extract_places = args.places or not (
        args.people or args.places or args.orgs or args.events or args.tags
    )
    extract_orgs = args.orgs or not (
        args.people or args.places or args.orgs or args.events or args.tags
    )
    extract_events = args.events or not (
        args.people or args.places or args.orgs or args.events or args.tags
    )
    extract_tags = args.tags or not (
        args.people or args.places or args.orgs or args.events or args.tags
    )

    # Create the output directory if it doesn't exist
    ensure_dir(os.path.dirname(args.output))

    # Process articles
    processed_articles = []
    article_count = 0

    print(f"Processing up to {args.limit} articles from {ARTICLES_PATH}")

    with open(ARTICLES_PATH, "r") as f:
        for line in f:
            if article_count >= args.limit:
                break

            loaded_entry = json.loads(line)

            # Normalize title and content to remove accents
            if "title" in loaded_entry:
                loaded_entry["title"] = normalize_text(loaded_entry["title"])
            if "content" in loaded_entry:
                loaded_entry["content"] = normalize_text(loaded_entry["content"])

            article_id = loaded_entry.get("id", f"article_{article_count}")
            article_text = f"# Title: {loaded_entry.get('title')}\n\n# Article: {loaded_entry.get('content')}"

            print(
                f"Processing article {article_count+1}/{args.limit}: {loaded_entry.get('title')}"
            )

            if args.show_article:
                print(article_text)

            # First check if the article is relevant to Guantanamo detention/prison
            try:
                # Check relevance before processing
                if args.local:
                    relevance_result = ollama_check_relevance(article_text, model="qwq")
                else:
                    relevance_result = gemini_check_relevance(article_text)

                # Add relevance info to metadata regardless of relevance result
                metadata = {
                    "relevance_check": {
                        "is_relevant": relevance_result.is_relevant,
                        "reason": relevance_result.reason,
                    }
                }

                # Show relevance result
                print(
                    f"Relevance check: {'RELEVANT' if relevance_result.is_relevant else 'NOT RELEVANT'}"
                )
                print(f"Reason: {relevance_result.reason}")

                # Only proceed with extraction if article is relevant
                if relevance_result.is_relevant:
                    # Process the article based on extraction flags
                    try:
                        if extract_people:
                            if args.local:
                                people = ollama_extract_people(
                                    article_text, model="qwq"
                                )
                                metadata["people"] = model_to_dict(people)
                                print(
                                    f"Extracted {len(metadata['people'])} people with Ollama"
                                )
                            else:
                                people = gemini_extract_people(article_text)
                                metadata["people"] = model_to_dict(people)
                                print(
                                    f"Extracted {len(metadata['people'])} people with Gemini"
                                )

                        if extract_places:
                            if args.local:
                                locations = ollama_extract_locations(
                                    article_text, model="qwq"
                                )
                                metadata["locations"] = model_to_dict(locations)
                                print(
                                    f"Extracted {len(metadata['locations'])} locations with Ollama"
                                )
                            else:
                                locations = gemini_extract_locations(article_text)
                                metadata["locations"] = model_to_dict(locations)
                                print(
                                    f"Extracted {len(metadata['locations'])} locations with Gemini"
                                )

                        if extract_orgs:
                            if args.local:
                                orgs = ollama_extract_organizations(
                                    article_text, model="qwq"
                                )
                                metadata["organizations"] = model_to_dict(orgs)
                                print(
                                    f"Extracted {len(metadata['organizations'])} organizations with Ollama"
                                )
                            else:
                                orgs = gemini_extract_organizations(article_text)
                                metadata["organizations"] = model_to_dict(orgs)
                                print(
                                    f"Extracted {len(metadata['organizations'])} organizations with Gemini"
                                )

                        if extract_events:
                            if args.local:
                                events = ollama_extract_events(
                                    article_text, model="qwq"
                                )
                                metadata["events"] = model_to_dict(events)
                                print(
                                    f"Extracted {len(metadata['events'])} events with Ollama"
                                )
                            else:
                                events = gemini_extract_events(article_text)
                                metadata["events"] = model_to_dict(events)
                                print(
                                    f"Extracted {len(metadata['events'])} events with Gemini"
                                )

                        if extract_tags:
                            if args.local:
                                tags_result = ollama_extract_tags(
                                    article_text, model="qwq"
                                )
                                metadata["tags"] = model_to_dict(tags_result.tags)
                                print(
                                    f"Extracted {len(metadata['tags'])} tags with Ollama"
                                )
                            else:
                                tags_result = gemini_extract_tags(article_text)
                                metadata["tags"] = model_to_dict(tags_result.tags)
                                print(
                                    f"Extracted {len(metadata['tags'])} tags with Gemini"
                                )
                    except Exception as e:
                        print(f"Error during extraction: {e}")
                        metadata["extraction_error"] = str(e)
                else:
                    print("Skipping detailed extraction for non-relevant article")

                # Add the metadata to the article
                loaded_entry["metadata"] = metadata
                loaded_entry["metadata_extraction_timestamp"] = (
                    datetime.now().isoformat()
                )

                # Add the processed article to the list
                processed_articles.append(loaded_entry)

            except Exception as e:
                print(f"Error processing article {article_count+1}: {e}")
                # Add the article with error information
                loaded_entry["metadata"] = {"relevance_check_error": str(e)}
                loaded_entry["metadata_extraction_timestamp"] = (
                    datetime.now().isoformat()
                )
                processed_articles.append(loaded_entry)

            article_count += 1

    # Write the processed articles to the output file
    with open(args.output, "w") as f:
        for article in processed_articles:
            f.write(json.dumps(article, cls=DateTimeEncoder) + "\n")

    print(
        f"Processed {len(processed_articles)} articles. Output written to {args.output}"
    )
