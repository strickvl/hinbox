"""Embedding utilities - imports from new implementation."""

# Import everything from new implementation
from src.utils.embeddings.manager import (
    EmbeddingManager,
    EmbeddingMode,
    get_default_manager,
)

# Re-export for compatibility
__all__ = ["EmbeddingManager", "EmbeddingMode", "get_default_manager"]
