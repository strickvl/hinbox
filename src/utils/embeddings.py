"""Module for computing embeddings of profile text."""

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config_loader import DomainConfig
from src.constants import (
    CLOUD_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
)
from src.exceptions import EmbeddingError
from src.logging_config import get_logger
from src.utils.error_handler import ErrorHandler

# Initialize logger
logger = get_logger("utils.embeddings")


class EmbeddingManager:
    """Centralized embedding management with domain-specific model support."""

    def __init__(self, model_type: str = "local", domain: str = "guantanamo"):
        """Initialize the EmbeddingManager.

        Args:
            model_type: 'cloud', 'local', or 'default'
            domain: Domain name for configuration lookup
        """
        self.model_type = model_type
        self.domain = domain
        self._model_cache: Dict[str, SentenceTransformer] = {}
        self.error_handler = ErrorHandler(
            "embedding_manager", {"model_type": model_type, "domain": domain}
        )

    def get_model_for_domain(self, domain: str = None) -> SentenceTransformer:
        """Get domain-specific embedding model if configured.

        Args:
            domain: Domain to get model for, defaults to instance domain

        Returns:
            Loaded SentenceTransformer model

        Raises:
            EmbeddingError: If model loading fails
        """
        domain = domain or self.domain

        try:
            # Try to get domain-specific embedding model from config
            try:
                config = DomainConfig(domain)
                domain_config = config.load_config()
                model_name = domain_config.get("embedding_model")
                if model_name:
                    return self._get_or_load_model(model_name)
            except Exception:
                # Fall back to system default if domain config fails
                pass

            # Use model type to determine which model to load
            if self.model_type == "cloud":
                model_name = CLOUD_EMBEDDING_MODEL
            elif self.model_type == "local":
                model_name = LOCAL_EMBEDDING_MODEL
            else:
                model_name = DEFAULT_EMBEDDING_MODEL

            return self._get_or_load_model(model_name)

        except Exception as e:
            error = EmbeddingError(
                f"Failed to load embedding model for domain {domain}",
                {
                    "domain": domain,
                    "model_type": self.model_type,
                    "original_error": str(e),
                },
            )
            self.error_handler.log_error(error, "error")
            raise error

    def _get_or_load_model(self, model_name: str) -> SentenceTransformer:
        """Get or load a specific model with caching.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded SentenceTransformer model
        """
        if model_name not in self._model_cache:
            logger.info(f"Loading embedding model: {model_name}")
            try:
                # Handle different model name formats
                if model_name.startswith("jina_ai/"):
                    # Convert jina_ai/ prefix to jinaai/ for sentence-transformers
                    st_model_name = model_name.replace("jina_ai/", "jinaai/")
                elif model_name.startswith("huggingface/"):
                    # Remove huggingface/ prefix
                    st_model_name = model_name.replace("huggingface/", "")
                else:
                    st_model_name = model_name

                self._model_cache[model_name] = SentenceTransformer(st_model_name)
                logger.info(f"Successfully loaded embedding model: {model_name}")
            except Exception:
                # Fall back to default model if specific model fails
                logger.warning(
                    f"Failed to load {model_name}, falling back to default: {DEFAULT_EMBEDDING_MODEL}"
                )
                if DEFAULT_EMBEDDING_MODEL not in self._model_cache:
                    self._model_cache[DEFAULT_EMBEDDING_MODEL] = SentenceTransformer(
                        DEFAULT_EMBEDDING_MODEL
                    )
                self._model_cache[model_name] = self._model_cache[
                    DEFAULT_EMBEDDING_MODEL
                ]

        return self._model_cache[model_name]

    def embed_text(self, text: str, domain: str = None) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            domain: Domain to use for model selection

        Returns:
            List of float values representing the embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text.strip():
            return []

        try:
            model = self.get_model_for_domain(domain)
            embeddings = model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            error = EmbeddingError(
                "Failed to generate text embedding",
                {
                    "text_length": len(text),
                    "domain": domain or self.domain,
                    "original_error": str(e),
                },
            )
            self.error_handler.log_error(error, "error")
            raise error

    def batch_embed(
        self, texts: List[str], domain: str = None, batch_size: int = 32
    ) -> List[List[float]]:
        """Efficiently embed multiple texts at once.

        Args:
            texts: List of texts to embed
            domain: Domain to use for model selection
            batch_size: Number of texts to process in each batch

        Returns:
            List of embeddings, one for each input text

        Raises:
            EmbeddingError: If batch embedding fails
        """
        if not texts:
            return []

        try:
            model = self.get_model_for_domain(domain)

            # Filter out empty texts and track their positions
            non_empty_texts = []
            empty_positions = []
            for i, text in enumerate(texts):
                if text.strip():
                    non_empty_texts.append(text)
                else:
                    empty_positions.append(i)

            if not non_empty_texts:
                return [[] for _ in texts]

            # Generate embeddings for non-empty texts
            embeddings = model.encode(non_empty_texts, batch_size=batch_size)
            embedding_list = embeddings.tolist()

            # Reconstruct full results with empty embeddings in correct positions
            results = []
            embedding_idx = 0
            for i in range(len(texts)):
                if i in empty_positions:
                    results.append([])
                else:
                    results.append(embedding_list[embedding_idx])
                    embedding_idx += 1

            return results

        except Exception as e:
            error = EmbeddingError(
                "Failed to generate batch embeddings",
                {
                    "text_count": len(texts),
                    "batch_size": batch_size,
                    "domain": domain or self.domain,
                    "original_error": str(e),
                },
            )
            self.error_handler.log_error(error, "error")
            raise error

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0

        try:
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

        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0


# Global default manager instance
_default_manager = None


def get_default_manager() -> EmbeddingManager:
    """Get the default embedding manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager(model_type="default")
    return _default_manager
