"""Extract events from article text."""

from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor
from src.utils.error_handler import handle_extraction_error


def gemini_extract_events(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Gemini."""
    try:
        extractor = EntityExtractor("events", domain)
        return extractor.extract_cloud(text=text, model=model, temperature=0)
    except Exception as e:
        return handle_extraction_error("events", "unknown", e, "gemini_extraction")


def ollama_extract_events(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Ollama."""
    try:
        extractor = EntityExtractor("events", domain)
        return extractor.extract_local(text=text, model=model, temperature=0)
    except Exception as e:
        return handle_extraction_error("events", "unknown", e, "ollama_extraction")
