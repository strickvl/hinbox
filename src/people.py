"""Extract people from article text."""

from typing import Any, Dict, List

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import create_list_models, get_person_model
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)


def gemini_extract_people(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Gemini."""
    Person = get_person_model(domain)
    return extract_entities_cloud(
        text=text,
        system_prompt=get_system_prompt("people", domain),
        response_model=List[Person],
        model=model,
        temperature=0,
    )


def ollama_extract_people(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Ollama."""
    list_models = create_list_models(domain)
    ArticlePeople = list_models["people"]

    results = extract_entities_local(
        text=text,
        system_prompt=get_system_prompt("people", domain),
        response_model=ArticlePeople,
        model=model,
        temperature=0,
    )
    return results.people


if __name__ == "__main__":
    text = "John Doe is a journalist at the New York Times. He is friends with Jane Smith, who is a lawyer at the same newspaper."

    print(gemini_extract_people(text))
    # print(ollama_extract_people(text))
