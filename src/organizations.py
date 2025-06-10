"""Extract organizations from article text."""

from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.extractors import EntityExtractor
from src.utils.error_handler import handle_extraction_error


def gemini_extract_organizations(
    text: str,
    model: str = CLOUD_MODEL,
    domain: str = "guantanamo",
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Gemini."""
    try:
        extractor = EntityExtractor("organizations", domain)
        return extractor.extract_cloud(
            text=text,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        return handle_extraction_error(
            "organizations", "unknown", e, "gemini_extraction"
        )


def ollama_extract_organizations(
    text: str,
    model: str = OLLAMA_MODEL,
    domain: str = "guantanamo",
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Ollama."""
    try:
        extractor = EntityExtractor("organizations", domain)
        return extractor.extract_local(
            text=text,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        return handle_extraction_error(
            "organizations", "unknown", e, "ollama_extraction"
        )
