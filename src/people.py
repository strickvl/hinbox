"""Extract people from article text."""

from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor


def gemini_extract_people(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Gemini."""
    extractor = EntityExtractor("people", domain)
    return extractor.extract_cloud(text=text, model=model, temperature=0)


def ollama_extract_people(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Ollama."""
    extractor = EntityExtractor("people", domain)
    return extractor.extract_local(text=text, model=model, temperature=0)


if __name__ == "__main__":
    text = "John Doe is a journalist at the New York Times. He is friends with Jane Smith, who is a lawyer at the same newspaper."

    print(gemini_extract_people(text))
    # print(ollama_extract_people(text))
