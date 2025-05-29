"""Extract locations from article text."""

from typing import Any, Dict, List

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import create_list_models, get_location_model
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)


def gemini_extract_locations(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Gemini."""
    Location = get_location_model(domain)
    return extract_entities_cloud(
        text=text,
        system_prompt=get_system_prompt("locations", domain),
        response_model=List[Location],
        model=model,
        temperature=0,
    )


# in testing, best options so far are qwq or mistral-small
def ollama_extract_locations(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Ollama."""
    list_models = create_list_models(domain)
    ArticleLocations = list_models["locations"]

    results = extract_entities_local(
        text=text,
        system_prompt=get_system_prompt("locations", domain),
        response_model=ArticleLocations,
        model=model,
        temperature=0,
    )
    return results.locations
