"""Extract organizations from article text."""

from typing import Any, Dict, List

from pydantic import BaseModel

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.models import Organization
from src.utils.extraction import (
    ORGANIZATIONS_SYSTEM_PROMPT,
    extract_entities_cloud,
    extract_entities_local,
)


class ArticleOrganizations(BaseModel):
    organizations: List[Organization]


def gemini_extract_organizations(
    text: str, model: str = CLOUD_MODEL
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Gemini."""
    return extract_entities_cloud(
        text=text,
        system_prompt=ORGANIZATIONS_SYSTEM_PROMPT,
        response_model=List[Organization],
        model=model,
        temperature=0,
    )


def ollama_extract_organizations(
    text: str, model: str = OLLAMA_MODEL
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Ollama."""
    results = extract_entities_local(
        text=text,
        system_prompt=ORGANIZATIONS_SYSTEM_PROMPT,
        response_model=ArticleOrganizations,
        model=model,
        temperature=0,
    )
    return results.organizations
