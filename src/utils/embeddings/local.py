"""Local embedding provider using sentence-transformers."""

from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.logging_config import get_logger

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingResult

logger = get_logger("utils.embeddings.local")


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            # Handle model name transformations
            model_name = self._transform_model_name(self.config.model_name)
            logger.info(f"Loading local embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded local embedding model: {model_name}")
        return self._model

    def _transform_model_name(self, model_name: str) -> str:
        """Transform model names to sentence-transformers format."""
        if model_name.startswith("huggingface/"):
            # Remove huggingface/ prefix
            return model_name.replace("huggingface/", "")
        elif model_name.startswith("jina_ai/"):
            # Convert jina_ai/ prefix to jinaai/ for sentence-transformers
            return model_name.replace("jina_ai/", "jinaai/")
        return model_name

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text locally."""
        if not text.strip():
            return []

        model = self._get_model()
        embedding = model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple texts locally."""
        # Filter empty texts
        non_empty_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]

        if not non_empty_texts:
            return EmbeddingResult(
                embeddings=[[] for _ in texts], model=self.config.model_name
            )

        model = self._get_model()
        indices, valid_texts = zip(*non_empty_texts)

        # Generate embeddings
        embeddings = model.encode(
            list(valid_texts),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )

        # Reconstruct full results
        results = [[] for _ in texts]
        for idx, embedding in zip(indices, embeddings):
            results[idx] = embedding.tolist()

        return EmbeddingResult(
            embeddings=results,
            model=self.config.model_name,
            usage={"texts": len(valid_texts)},
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        model = self._get_model()
        return {
            "type": "local",
            "model_name": self.config.model_name,
            "embedding_dimension": model.get_sentence_embedding_dimension(),
            "max_sequence_length": model.max_seq_length,
        }
