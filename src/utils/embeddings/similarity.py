"""Decoupled helpers for embedding similarity and manager retrieval.

These utilities are intentionally independent of src/merge.py to allow reuse
across modules without coupling to merge logic.
"""

import os
from importlib.util import find_spec
from typing import Dict, List, Tuple

import numpy as np

from src.utils.embeddings.manager import EmbeddingManager, EmbeddingMode

# Cache managers per (domain, mode) to avoid re-initialization overhead
# and ensure mode changes within a long-lived process are respected.
_MANAGER_CACHE: Dict[Tuple[str, str], EmbeddingManager] = {}


def _resolve_mode_for_cache(domain: str) -> str:
    """Cheaply resolve the effective embedding mode without constructing a manager.

    Mirrors the resolution logic in EmbeddingManager._resolve_requested_mode +
    _resolve_auto but avoids loading domain config or constructing providers.
    """
    # 1. Check environment variable first
    env_mode = os.getenv("EMBEDDING_MODE")
    if env_mode:
        try:
            mode = EmbeddingMode(env_mode.lower())
        except ValueError:
            mode = None
        if mode is not None:
            if mode == EmbeddingMode.AUTO:
                return (
                    EmbeddingMode.LOCAL.value
                    if find_spec("sentence_transformers") is not None
                    else EmbeddingMode.CLOUD.value
                )
            return mode.value

    # 2. Try loading domain config for mode
    try:
        import src.config_loader as config_loader

        config = config_loader.DomainConfig(domain).load_config()
        mode_str = config.get("embeddings", {}).get("mode", "cloud")
        mode = EmbeddingMode(mode_str)
    except Exception:
        mode = EmbeddingMode.CLOUD

    if mode == EmbeddingMode.AUTO:
        return (
            EmbeddingMode.LOCAL.value
            if find_spec("sentence_transformers") is not None
            else EmbeddingMode.CLOUD.value
        )
    return mode.value


def get_embedding_manager(domain: str = "guantanamo") -> EmbeddingManager:
    """Get a cached EmbeddingManager instance for the specified domain.

    The cache key includes the resolved mode so that environment variable
    changes (EMBEDDING_MODE) in a long-running process create a new manager
    rather than reusing one initialised under the old mode.
    """
    resolved_mode = _resolve_mode_for_cache(domain)
    cache_key = (domain, resolved_mode)

    cached = _MANAGER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    manager = EmbeddingManager(domain=domain)
    _MANAGER_CACHE[cache_key] = manager
    return manager


def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors using NumPy.

    Returns:
        A similarity score in [-1.0, 1.0]. Returns 0.0 if either vector is empty,
        has zero norm, or if the vectors are of different lengths.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


__all__ = ["compute_similarity", "get_embedding_manager", "EmbeddingManager"]
