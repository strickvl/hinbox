from typing import Any, Dict, List

import instructor
import litellm
import spacy
from openai import OpenAI
from pydantic import BaseModel

from src.constants import GEMINI_MODEL, OLLAMA_API_KEY, OLLAMA_API_URL, OLLAMA_MODEL
from src.models import Place, PlaceType

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class ArticleLocations(BaseModel):
    locations: List[Place]


def spacy_extract_locations(text: str) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using spaCy."""
    nlp = spacy.load("en_core_web_lg")

    # Process the text
    doc = nlp(text)

    # Extract location entities (GPE: countries, cities, etc. and LOC: non-GPE locations)
    locations = []
    seen_locations = set()  # Track locations we've already added

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            # Only add if we haven't seen this location name before
            if ent.text not in seen_locations:
                locations.append(Place(name=ent.text, type=PlaceType.OTHER))
                seen_locations.add(ent.text)

    return locations


def gemini_extract_locations(
    text: str, model: str = GEMINI_MODEL
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Gemini."""
    client = instructor.from_litellm(litellm.completion)

    results = client.chat.completions.create(
        model=model,
        response_model=List[Place],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting locations from news articles.

When identifying locations, categorize them using the following place types:
- country: A sovereign state with its own government and territory
- province: An administrative division of a country
- state: A constituent political entity within a country
- district: An administrative division of a city or region
- city: An urban area with a significant population
- prison_location: A specific location within a detention facility like 'Camp Delta' or 'Camp 6'
- other: Any other type of place not covered by the above categories

Only use standard ASCII characters for the names that you extract.

Extract all locations mentioned in the text and categorize them appropriately.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        metadata={
            "project_name": "hinbox",  # for braintrust
            "tags": ["dev"],
        },
    )
    return results


# in testing, best options so far are qwq or mistral-small
def ollama_extract_locations(
    text: str, model: str = OLLAMA_MODEL
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Gemini."""
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    results = client.beta.chat.completions.parse(
        model=model,
        response_format=ArticleLocations,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting locations from
                           news articles.

                When identifying locations, categorize them using the following place types:
                - country: A sovereign state with its own government and territory
                - province: An administrative division of a country
                - state: A constituent political entity within a country
                - district: An administrative division of a city or region
                - city: An urban area with a significant population
                - prison_location: A specific location within a detention
                facility like 'Camp Delta' or 'Camp 6'.
                - other: Any other type of place not covered by the above categories

                Only use standard ASCII characters for the names that you extract.

                Extract all locations mentioned in the text and categorize them appropriately.
                """,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return results.choices[0].message.parsed.locations
