"""Module for computing embeddings of profile text."""

from typing import List, Optional

import numpy as np
from litellm import embedding

# Default model names
DEFAULT_LOCAL_MODEL = "ollama/nomic-embed-text"  # Example local model
DEFAULT_CLOUD_MODEL = "openai/text-embedding-ada-002"  # Example cloud model


def embed_text(
    text: str, is_local: bool = True, model_name: Optional[str] = None
) -> List[float]:
    """Return the embedding of the given text as a list of float32.

    Args:
        text: The text to embed
        is_local: Whether to use a local embedding model (True) or cloud-based model (False)
        model_name: The specific model to use for embedding
                    If None, uses DEFAULT_LOCAL_MODEL or DEFAULT_CLOUD_MODEL based on is_local

    Returns:
        A list of float32 values representing the embedding
    """
    if not text.strip():
        return []

    # Determine which model to use
    if model_name is None:
        model_name = DEFAULT_LOCAL_MODEL if is_local else DEFAULT_CLOUD_MODEL

    # Use LiteLLM for embedding with the selected model
    response = embedding(model=model_name, input=[text])
    embedding_vector = np.array(response["data"][0]["embedding"], dtype=np.float32)

    return embedding_vector.tolist()


# if __name__ == "__main__":
#     print(embed_text("Hello, world!"))
#     print(embed_text("Hello, world!", is_local=False))
