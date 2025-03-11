import argparse
import json

import litellm
from rich import print

from src.v2.events import gemini_extract_events, ollama_extract_events
from src.v2.locations import (
    gemini_extract_locations,
    ollama_extract_locations,
    spacy_extract_locations,
)
from src.v2.organizations import (
    gemini_extract_organizations,
    ollama_extract_organizations,
    spacy_extract_organizations,
)
from src.v2.people import (
    gemini_extract_people,
    ollama_extract_people,
    spacy_extract_people,
)
from src.v2.tags import gemini_extract_tags, ollama_extract_tags

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


ARTICLES_PATH = (
    "/home/strickvl/coding/hinbox/data/raw_sources/miami_herald_articles.jsonl"
)


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

    with open(ARTICLES_PATH, "r") as f:
        first_entry = f.readline()
        loaded_entry = json.loads(first_entry)
        article = loaded_entry.get("content")

        # Extract information based on flags
        spacy_locations = spacy_extract_locations(article) if extract_places else None
        spacy_people = spacy_extract_people(article) if extract_people else None
        spacy_orgs = spacy_extract_organizations(article) if extract_orgs else None

        # Run Gemini extraction only if not in local mode
        gemini_locations = None
        gemini_people = None
        gemini_orgs = None
        gemini_events = None
        gemini_tags = None
        if not args.local:
            if extract_places:
                gemini_locations = gemini_extract_locations(article)
            if extract_people:
                gemini_people = gemini_extract_people(article)
            if extract_orgs:
                gemini_orgs = gemini_extract_organizations(article)
            if extract_events:
                gemini_events = gemini_extract_events(article)
            if extract_tags:
                gemini_tags = gemini_extract_tags(article)

        # Run Ollama extraction based on flags
        ollama_locations = (
            ollama_extract_locations(article, model="qwq") if extract_places else None
        )
        ollama_people = (
            ollama_extract_people(article, model="qwq") if extract_people else None
        )
        ollama_orgs = (
            ollama_extract_organizations(article, model="qwq") if extract_orgs else None
        )
        ollama_events = (
            ollama_extract_events(article, model="qwq") if extract_events else None
        )
        ollama_tags = (
            ollama_extract_tags(article, model="qwq") if extract_tags else None
        )

    # Print the article
    print(article)

    # Print results based on flags
    if extract_places:
        print("SpaCy locations:")
        print(spacy_locations)

        if gemini_locations:
            print("Gemini locations:")
            print(gemini_locations)

        print("Ollama locations:")
        print(ollama_locations)

    if extract_people:
        print("SpaCy people:")
        print(spacy_people)

        if gemini_people:
            print("Gemini people:")
            print(gemini_people)

        print("Ollama people:")
        print(ollama_people)

    if extract_orgs:
        print("SpaCy organizations:")
        print(spacy_orgs)

        if gemini_orgs:
            print("Gemini organizations:")
            print(gemini_orgs)

        print("Ollama organizations:")
        print(ollama_orgs)

    if extract_events:
        if gemini_events:
            print("Gemini events:")
            print(gemini_events)

        print("Ollama events:")
        print(ollama_events)

    if extract_tags:
        if gemini_tags:
            print("Gemini tags:")
            print(gemini_tags)

        print("Ollama tags:")
        print(ollama_tags)
