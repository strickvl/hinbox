import argparse
import json

import litellm
from rich import print

from src.v2.locations import (
    gemini_extract_locations,
    ollama_extract_locations,
    spacy_extract_locations,
)

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
    args = parser.parse_args()

    with open(ARTICLES_PATH, "r") as f:
        first_entry = f.readline()
        loaded_entry = json.loads(first_entry)
        article = loaded_entry.get("content")

        # Always run spaCy extraction
        spacy_locations = spacy_extract_locations(article)

        # Run Gemini extraction only if not in local mode
        gemini_locations = None
        if not args.local:
            gemini_locations = gemini_extract_locations(article)

        # Always run Ollama extraction
        ollama_locations = ollama_extract_locations(article, model="qwq")

    print(article)
    print("SpaCy locations:")
    print(spacy_locations)

    if gemini_locations:
        print("Gemini locations:")
        print(gemini_locations)

    print("Ollama locations:")
    print(ollama_locations)
