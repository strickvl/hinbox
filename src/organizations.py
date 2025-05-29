"""Extract organizations from article text."""

from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor


def gemini_extract_organizations(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Gemini."""
    extractor = EntityExtractor("organizations", domain)
    return extractor.extract_cloud(text=text, model=model, temperature=0)


def ollama_extract_organizations(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Ollama."""
    extractor = EntityExtractor("organizations", domain)
    return extractor.extract_local(text=text, model=model, temperature=0)
