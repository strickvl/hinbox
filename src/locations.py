"""Extract locations from article text."""

from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor
from src.utils.error_handler import handle_extraction_error


def gemini_extract_locations(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Gemini."""
    try:
        extractor = EntityExtractor("locations", domain)
        return extractor.extract_cloud(text=text, model=model, temperature=0)
    except Exception as e:
        return handle_extraction_error("locations", "unknown", e, "gemini_extraction")


# in testing, best options so far are qwq or mistral-small
def ollama_extract_locations(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Ollama."""
    try:
        extractor = EntityExtractor("locations", domain)
        return extractor.extract_local(text=text, model=model, temperature=0)
    except Exception as e:
        return handle_extraction_error("locations", "unknown", e, "ollama_extraction")
