"""Extract organizations from article text."""

from typing import Any, Dict, List

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import create_list_models, get_organization_model
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)


def gemini_extract_organizations(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Gemini."""
    Organization = get_organization_model(domain)
    return extract_entities_cloud(
        text=text,
        system_prompt=get_system_prompt("organizations", domain),
        response_model=List[Organization],
        model=model,
        temperature=0,
    )


def ollama_extract_organizations(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Ollama."""
    list_models = create_list_models(domain)
    ArticleOrganizations = list_models["organizations"]

    results = extract_entities_local(
        text=text,
        system_prompt=get_system_prompt("organizations", domain),
        response_model=ArticleOrganizations,
        model=model,
        temperature=0,
    )
    return results.organizations
