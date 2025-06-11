"""Extract people entities from article text using cloud and local LLM models.

This module provides functions to extract person entities from text content using
either cloud-based (Gemini) or local (Ollama) language models. The extraction
follows domain-specific prompts and categories defined in the configuration.
"""

from typing import Any, Dict, List, Optional

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor
from src.utils.error_handler import handle_extraction_error


def gemini_extract_people(
    text: str,
    model: str = CLOUD_MODEL,
    domain: str = "guantanamo",
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Gemini cloud models.

    Uses the EntityExtractor to identify and extract people mentioned in the text
    according to the domain-specific configuration and prompts.

    Args:
        text: The input text content to extract people from
        model: Gemini model name to use for extraction
        domain: Domain configuration to use for extraction prompts and categories
        langfuse_session_id: Optional Langfuse session ID for request tracing
        langfuse_trace_id: Optional Langfuse trace ID for request tracing

    Returns:
        List of dictionaries containing extracted person entities with fields
        like name, role, description, and other person-specific attributes

    Raises:
        Exception: Various exceptions during LLM API calls or response parsing

    Note:
        Returns empty list on extraction errors after logging the error details.
    """
    try:
        extractor = EntityExtractor("people", domain)
        return extractor.extract_cloud(
            text=text,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        return handle_extraction_error("people", "unknown", e, "gemini_extraction")


def ollama_extract_people(
    text: str,
    model: str = OLLAMA_MODEL,
    domain: str = "guantanamo",
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Ollama local models.

    Uses the EntityExtractor to identify and extract people mentioned in the text
    according to the domain-specific configuration and prompts via local Ollama server.

    Args:
        text: The input text content to extract people from
        model: Ollama model name to use for extraction
        domain: Domain configuration to use for extraction prompts and categories
        langfuse_session_id: Optional Langfuse session ID for request tracing
        langfuse_trace_id: Optional Langfuse trace ID for request tracing

    Returns:
        List of dictionaries containing extracted person entities with fields
        like name, role, description, and other person-specific attributes

    Raises:
        Exception: Various exceptions during local model API calls or response parsing

    Note:
        Returns empty list on extraction errors after logging the error details.
        Requires Ollama server to be running and accessible.
    """
    try:
        extractor = EntityExtractor("people", domain)
        return extractor.extract_local(
            text=text,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        return handle_extraction_error("people", "unknown", e, "ollama_extraction")


if __name__ == "__main__":
    text = "John Doe is a journalist at the New York Times. He is friends with Jane Smith, who is a lawyer at the same newspaper."

    print(gemini_extract_people(text))
    # print(ollama_extract_people(text))
