"""Module for computing embeddings of profile text."""

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.logging_config import get_logger

# Initialize logger
logger = get_logger("utils.embeddings")

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


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score between 0 and 1
    """
    import numpy as np

    if not embedding1 or not embedding2:
        return 0.0

    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))
