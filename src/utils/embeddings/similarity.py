"""Decoupled helpers for embedding similarity and manager retrieval.

These utilities are intentionally independent of src/merge.py to allow reuse
across modules without coupling to merge logic.
"""

from typing import Dict, List

import numpy as np

from src.utils.embeddings.manager import EmbeddingManager

# Cache managers per domain to avoid re-initialization overhead
_MANAGER_CACHE: Dict[str, EmbeddingManager] = {}


def get_embedding_manager(domain: str = "guantanamo") -> EmbeddingManager:
    """Get a cached EmbeddingManager instance for the specified domain."""
    manager = _MANAGER_CACHE.get(domain)
    if manager is None:
        manager = EmbeddingManager(domain=domain)
        _MANAGER_CACHE[domain] = manager
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
