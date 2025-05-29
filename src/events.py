"""Extract events from article text."""

from typing import Any, Dict, List

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import create_list_models, get_event_model
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)


def gemini_extract_events(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Gemini."""
    Event = get_event_model(domain)
    # Use a simpler approach with List[Event] instead of ArticleEvents
    # This might avoid the issue with default values in nested models
    try:
        return extract_entities_cloud(
            text=text,
            system_prompt=get_system_prompt("events", domain),
            response_model=List[Event],
            model=model,
            temperature=0,
        )
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Return an empty list as fallback
        return []


def ollama_extract_events(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Ollama."""
    list_models = create_list_models(domain)
    ArticleEvents = list_models["events"]

    results = extract_entities_local(
        text=text,
        system_prompt=get_system_prompt("events", domain),
        response_model=ArticleEvents,
        model=model,
        temperature=0,
    )
    return results.events
