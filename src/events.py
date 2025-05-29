"""Extract events from article text."""

from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor


def gemini_extract_events(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Gemini."""
    extractor = EntityExtractor("events", domain)
    try:
        return extractor.extract_cloud(text=text, model=model, temperature=0)
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Return an empty list as fallback
        return []


def ollama_extract_events(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Ollama."""
    extractor = EntityExtractor("events", domain)
    return extractor.extract_local(text=text, model=model, temperature=0)
