"""Extract location entities from article text using cloud and local LLM models.

This module provides functions to extract location entities from text content using
either cloud-based (Gemini) or local (Ollama) language models. Locations include
geographic places, facilities, buildings, and other place-based entities according to
domain-specific categorization.
"""

from typing import Any, Dict, List, Optional

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor
from src.utils.error_handler import handle_extraction_error


def gemini_extract_locations(
    text: str,
    model: str = CLOUD_MODEL,
    domain: str = "guantanamo",
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Gemini cloud models.

    Uses the EntityExtractor to identify and extract locations mentioned in the text
    according to the domain-specific configuration and prompts.

    Args:
        text: The input text content to extract locations from
        model: Gemini model name to use for extraction
        domain: Domain configuration to use for extraction prompts and categories
        langfuse_session_id: Optional Langfuse session ID for request tracing
        langfuse_trace_id: Optional Langfuse trace ID for request tracing

    Returns:
        List of dictionaries containing extracted location entities with fields
        like name, type, description, and other location-specific attributes

    Raises:
        Exception: Various exceptions during LLM API calls or response parsing

    Note:
        Returns empty list on extraction errors after logging the error details.
        Locations are classified into types based on domain configuration.
    """
    try:
        extractor = EntityExtractor("locations", domain)
        return extractor.extract_cloud(
            text=text,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        return handle_extraction_error("locations", "unknown", e, "gemini_extraction")


def ollama_extract_locations(
    text: str,
    model: str = OLLAMA_MODEL,
    domain: str = "guantanamo",
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract location entities from the provided text using Ollama local models.

    Uses the EntityExtractor to identify and extract locations mentioned in the text
    according to the domain-specific configuration and prompts via local Ollama server.

    Args:
        text: The input text content to extract locations from
        model: Ollama model name to use for extraction
        domain: Domain configuration to use for extraction prompts and categories
        langfuse_session_id: Optional Langfuse session ID for request tracing
        langfuse_trace_id: Optional Langfuse trace ID for request tracing

    Returns:
        List of dictionaries containing extracted location entities with fields
        like name, type, description, and other location-specific attributes

    Raises:
        Exception: Various exceptions during local model API calls or response parsing

    Note:
        Returns empty list on extraction errors after logging the error details.
        Requires Ollama server to be running and accessible.
        In testing, qwq and mistral-small models have shown best performance.
        Locations are classified into types based on domain configuration.
    """
    try:
        extractor = EntityExtractor("locations", domain)
        return extractor.extract_local(
            text=text,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        return handle_extraction_error("locations", "unknown", e, "ollama_extraction")
