"""Extract events from article text."""

from typing import Any, Dict, List

from pydantic import BaseModel

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.models import Event
from src.utils.extraction import (
    EVENTS_SYSTEM_PROMPT,
    extract_entities_cloud,
    extract_entities_local,
)


class ArticleEvents(BaseModel):
    events: List[Event]


def gemini_extract_events(text: str, model: str = CLOUD_MODEL) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Gemini."""
    # Use a simpler approach with List[Event] instead of ArticleEvents
    # This might avoid the issue with default values in nested models
    try:
        return extract_entities_cloud(
            text=text,
            system_prompt=EVENTS_SYSTEM_PROMPT,
            response_model=List[Event],
            model=model,
            temperature=0,
        )
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Return an empty list as fallback
        return []


def ollama_extract_events(text: str, model: str = OLLAMA_MODEL) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Ollama."""
    results = extract_entities_local(
        text=text,
        system_prompt=EVENTS_SYSTEM_PROMPT,
        response_model=ArticleEvents,
        model=model,
        temperature=0,
    )
    return results.events
