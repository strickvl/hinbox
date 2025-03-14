"""Module for computing embeddings of profile text."""

from typing import List, Optional

import numpy as np
from litellm import embedding

from src.constants import CLOUD_EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL


def embed_text(
    text: str, model_name: Optional[str] = None, is_local: bool = True
) -> List[float]:
    """Return the embedding of the given text as a list of float32.

    Args:
        text: The text to embed
        model_name: The specific model to use for embedding. If None, uses LOCAL_EMBEDDING_MODEL
                   or CLOUD_EMBEDDING_MODEL based on is_local
        is_local: Whether to use a local embedding model (True) or cloud-based model (False).
                 Only used if model_name is None.

    Returns:
        A list of float32 values representing the embedding
    """
    if not text.strip():
        return []

    # Determine which model to use
    if model_name is None:
        model_name = LOCAL_EMBEDDING_MODEL if is_local else CLOUD_EMBEDDING_MODEL

    # Use LiteLLM for embedding with the selected model
    response = embedding(model=model_name, input=[text])
    embedding_vector = np.array(response["data"][0]["embedding"], dtype=np.float32)

    return embedding_vector.tolist()
