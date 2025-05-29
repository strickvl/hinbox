"""Extract locations from article text."""

from typing import Any, Dict, List

from pydantic import BaseModel

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.models import Place
from src.utils.extraction import (
    LOCATIONS_SYSTEM_PROMPT,
    extract_entities_cloud,
    extract_entities_local,
)


class ArticleLocations(BaseModel):
    locations: List[Place]


def gemini_extract_locations(
    text: str, model: str = CLOUD_MODEL
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Gemini."""
    return extract_entities_cloud(
        text=text,
        system_prompt=LOCATIONS_SYSTEM_PROMPT,
        response_model=List[Place],
        model=model,
        temperature=0,
    )


# in testing, best options so far are qwq or mistral-small
def ollama_extract_locations(
    text: str, model: str = OLLAMA_MODEL
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Ollama."""
    results = extract_entities_local(
        text=text,
        system_prompt=LOCATIONS_SYSTEM_PROMPT,
        response_model=ArticleLocations,
        model=model,
        temperature=0,
    )
    return results.locations
