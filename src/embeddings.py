"""Module for computing embeddings of profile text."""

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

# Initialize logger
logger = logging.getLogger(__name__)

# Global model instance to avoid reloading
_model = None


def get_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """Get or initialize the SentenceTransformer model.

    Args:
        model_name: The name of the model to load

    Returns:
        The loaded SentenceTransformer model
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
    return _model


def embed_text(
    text: str, model_name: Optional[str] = None, is_local: bool = True
) -> List[float]:
    """Return the embedding of the given text as a list of float32.

    Args:
        text: The text to embed
        model_name: The specific model to use for embedding. If None, uses the default
                   SentenceTransformer model (all-MiniLM-L6-v2)
        is_local: Kept for backward compatibility but no longer used

    Returns:
        A list of float32 values representing the embedding
    """
    if not text.strip():
        return []

    # just use all-MiniLM-L6-v2
    st_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = get_model(st_model_name)

    embeddings = model.encode(text)

    # Convert to list of float values
    return embeddings.tolist()
