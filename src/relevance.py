"""Check article relevance to configured domain topics."""

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.dynamic_models import get_relevance_model
from src.utils.extraction import (
    extract_entities_cloud,
    extract_entities_local,
)


def gemini_check_relevance(
    text: str, model: str = CLOUD_MODEL, domain: str = "guantanamo"
):
    """
    Check if an article is relevant to the domain using Gemini.

    Args:
        text: The article text to check
        model: The Gemini model to use
        domain: The domain configuration to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    ArticleRelevance = get_relevance_model(domain)
    return extract_entities_cloud(
        text=text,
        system_prompt=get_system_prompt("relevance", domain),
        response_model=ArticleRelevance,
        model=model,
        temperature=0,
    )


def ollama_check_relevance(
    text: str, model: str = OLLAMA_MODEL, domain: str = "guantanamo"
):
    """
    Check if an article is relevant to the domain using Ollama.

    Args:
        text: The article text to check
        model: The Ollama model to use
        domain: The domain configuration to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    ArticleRelevance = get_relevance_model(domain)
    return extract_entities_local(
        text=text,
        system_prompt=get_system_prompt("relevance", domain),
        response_model=ArticleRelevance,
        model=model,
        temperature=0,
    )
