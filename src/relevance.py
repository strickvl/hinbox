"""Check article relevance to Guantanamo Bay topics."""

from pydantic import BaseModel, Field

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.utils.extraction import (
    RELEVANCE_SYSTEM_PROMPT,
    extract_entities_cloud,
    extract_entities_local,
)


class ArticleRelevance(BaseModel):
    """Model for determining if an article is relevant to Guantanamo detention/torture."""

    is_relevant: bool = Field(
        description="Whether the article is relevant to Guantanamo detention, prison, torture, naval base, or related issues"
    )
    reason: str = Field(
        description="Brief explanation of why the article is or is not relevant"
    )


def gemini_check_relevance(text: str, model: str = CLOUD_MODEL) -> ArticleRelevance:
    """
    Check if an article is relevant to Guantanamo detention/prison using Gemini.

    Args:
        text: The article text to check
        model: The Gemini model to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    return extract_entities_cloud(
        text=text,
        system_prompt=RELEVANCE_SYSTEM_PROMPT,
        response_model=ArticleRelevance,
        model=model,
        temperature=0,
    )


def ollama_check_relevance(text: str, model: str = OLLAMA_MODEL) -> ArticleRelevance:
    """
    Check if an article is relevant to Guantanamo detention/prison using Ollama.

    Args:
        text: The article text to check
        model: The Ollama model to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    return extract_entities_local(
        text=text,
        system_prompt=RELEVANCE_SYSTEM_PROMPT,
        response_model=ArticleRelevance,
        model=model,
        temperature=0,
    )
