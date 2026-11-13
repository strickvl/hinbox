"""A new entity row, as the merger creates one (also used by the corrections
step that splits a page, ``src/engine/corrections_merges.py``)."""

from typing import Any, Dict, List, Optional, Tuple, Union

from src.constants import ENABLE_PROFILE_VERSIONING
from src.engine.profiles import VersionedProfile
from src.utils.embeddings.manager import EmbeddingManager


def new_entity_row(
    table: str,
    alternative_field: str,
    entity_dict: Dict[str, Any],
    entity_key: Union[str, Tuple],
    profile: Dict[str, Any],
    versioned_profile: VersionedProfile,
    profile_embedding: List[float],
    reflection_history: List[Any],
    article_id: str,
    article_title: str,
    article_url: str,
    article_published_date: Any,
    extraction_timestamp: str,
    *,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    embedding_fingerprint: Optional[str] = None,
    search_embedding: Optional[List[float]] = None,
    search_embedding_model: Optional[str] = None,
    search_embedding_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a new entity dictionary with appropriate structure for each entity type."""
    resolved_dim = embedding_dim or (
        len(profile_embedding) if profile_embedding else None
    )
    search_dim = search_embedding_dim or (
        len(search_embedding) if search_embedding else None
    )
    base_entity: Dict[str, Any] = {
        "profile": profile,
        "profile_versions": versioned_profile.model_dump()
        if ENABLE_PROFILE_VERSIONING
        else None,
        "articles": [
            {
                "article_id": article_id,
                "article_title": article_title,
                "article_url": article_url,
                "article_published_date": article_published_date,
            }
        ],
        "profile_embedding": profile_embedding,
        "profile_embedding_model": embedding_model,
        "profile_embedding_dim": resolved_dim,
        "profile_embedding_fingerprint": embedding_fingerprint
        or EmbeddingManager.make_fingerprint(embedding_model, resolved_dim),
        "search_embedding": search_embedding or [],
        "search_embedding_model": search_embedding_model,
        "search_embedding_dim": search_dim,
        "search_embedding_fingerprint": EmbeddingManager.make_fingerprint(
            search_embedding_model, search_dim
        )
        if search_embedding
        else None,
        "extraction_timestamp": extraction_timestamp,
        alternative_field: [],
        "aliases": entity_dict.get("aliases", []),
        "reflection_history": reflection_history or [],
    }

    if table == "people":
        base_entity["name"] = entity_key
        base_entity["type"] = entity_dict.get("type", "")  # picks starter headings
    elif table in ("organizations", "locations"):
        base_entity["name"] = entity_key[0]
        base_entity["type"] = entity_key[1]
    elif table == "events":
        base_entity["title"] = entity_key[0]
        base_entity["start_date"] = entity_key[1]
        base_entity["description"] = entity_dict.get("description", "")
        base_entity["event_type"] = entity_dict.get("event_type", "")
        base_entity["end_date"] = entity_dict.get("end_date", "")
        base_entity["is_fuzzy_date"] = entity_dict.get("is_fuzzy_date", False)
        base_entity["tags"] = entity_dict.get("tags", [])

    return base_entity
