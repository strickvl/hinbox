"""Thin wrappers for entity merge operations.

Exposes convenience functions that delegate to the generic EntityMerger
implementation for each entity type. This module intentionally contains no
embedding logic and only imports SIMILARITY_THRESHOLD for default parameters.
"""

from typing import Any, Dict, List, Optional

from src.constants import SIMILARITY_THRESHOLD


def merge_people_generic(
    extracted_people: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
    *,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> None:
    """Merge people entities using the generic EntityMerger."""
    from src.engine.mergers import EntityMerger

    merger = EntityMerger("people")
    merger.merge_entities(
        extracted_people,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_id=langfuse_trace_id,
    )


def merge_organizations_generic(
    extracted_orgs: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
    *,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> None:
    """Merge organization entities using the generic EntityMerger."""
    from src.engine.mergers import EntityMerger

    merger = EntityMerger("organizations")
    merger.merge_entities(
        extracted_orgs,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_id=langfuse_trace_id,
    )


def merge_locations_generic(
    extracted_locations: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
    *,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> None:
    """Merge location entities using the generic EntityMerger."""
    from src.engine.mergers import EntityMerger

    merger = EntityMerger("locations")
    merger.merge_entities(
        extracted_locations,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_id=langfuse_trace_id,
    )


def merge_events_generic(
    extracted_events: List[Dict[str, Any]],
    entities: Dict[str, Dict],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    article_content: str,
    extraction_timestamp: str,
    model_type: str = "gemini",
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    domain: str = "guantanamo",
    *,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> None:
    """Merge event entities using the generic EntityMerger."""
    from src.engine.mergers import EntityMerger

    merger = EntityMerger("events")
    merger.merge_entities(
        extracted_events,
        entities,
        article_id,
        article_title,
        article_url,
        article_published_date,
        article_content,
        extraction_timestamp,
        model_type,
        similarity_threshold,
        domain,
        langfuse_session_id=langfuse_session_id,
        langfuse_trace_id=langfuse_trace_id,
    )
