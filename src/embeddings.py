"""Module for computing embeddings of profile text."""

from typing import List


from sentence_transformers import SentenceTransformer
import numpy as np

# Example local model; you can adjust to your needs
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> List[float]:
    """Return the embedding of the given text as a list of float32."""
    if not text.strip():
        return []
    embedding = _model.encode(text)
    # Convert to float32 to reduce space & for direct Parquet storage
    embedding = embedding.astype(np.float32)
    return embedding.tolist()
