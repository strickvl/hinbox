"""Extract people from article text."""

from typing import Any, Dict, List

from pydantic import BaseModel

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.models import Person
from src.utils.extraction import (
    PEOPLE_SYSTEM_PROMPT,
    extract_entities_cloud,
    extract_entities_local,
)


class ArticlePeople(BaseModel):
    people: List[Person]


def gemini_extract_people(text: str, model: str = CLOUD_MODEL) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Gemini."""
    return extract_entities_cloud(
        text=text,
        system_prompt=PEOPLE_SYSTEM_PROMPT,
        response_model=List[Person],
        model=model,
        temperature=0,
    )


def ollama_extract_people(text: str, model: str = OLLAMA_MODEL) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Ollama."""
    results = extract_entities_local(
        text=text,
        system_prompt=PEOPLE_SYSTEM_PROMPT,
        response_model=ArticlePeople,
        model=model,
        temperature=0,
    )
    return results.people


if __name__ == "__main__":
    text = "John Doe is a journalist at the New York Times. He is friends with Jane Smith, who is a lawyer at the same newspaper."

    print(gemini_extract_people(text))
    # print(ollama_extract_people(text))
