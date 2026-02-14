"""Local embedding provider using sentence-transformers.

This module avoids importing heavy dependencies (sentence-transformers/torch)
at import time so environments without proper NumPy/PyTorch support can still
run the application when local embeddings are not used. The actual
sentence-transformers import happens lazily inside `_get_model()`.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.logging_config import get_logger

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingResult

if TYPE_CHECKING:  # for type checkers only; avoids runtime import
    pass  # pragma: no cover

logger = get_logger("utils.embeddings.local")


def _resolve_device(device: str) -> Optional[str]:
    """Resolve the device string for SentenceTransformer.

    Returns None for "auto" so sentence-transformers can auto-detect the best
    available device (CUDA > MPS > CPU). Explicit values are passed through.
    """
    if device == "auto":
        return None  # let sentence-transformers choose
    return device


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        # Use Any for runtime to avoid requiring the class at import time
        self._model: Optional[Any] = None

    def _get_model(self):
        """Lazy load the model.

        Raises RuntimeError with actionable guidance if sentence-transformers
        (and its PyTorch dependency) is not installed.
        """
        if self._model is None:
            # Handle model name transformations
            model_name = self._transform_model_name(self.config.model_name)
            logger.info(f"Loading local embedding model: {model_name}")

            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Local embeddings require the optional dependency "
                    "'sentence-transformers' (and PyTorch).\n\n"
                    "  Install:  pip install 'hinbox[local-embeddings]'\n"
                    "  Or switch: set EMBEDDING_MODE=cloud (env var)\n"
                    "             or embeddings.mode: cloud (config YAML)\n"
                ) from e

            resolved_device = _resolve_device(self.config.device)
            if resolved_device is not None:
                self._model = SentenceTransformer(model_name, device=resolved_device)
            else:
                self._model = SentenceTransformer(model_name)

            logger.info(
                f"Successfully loaded local embedding model: {model_name} "
                f"(device={self._model.device})"
            )
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
                embeddings=[[] for _ in texts],
                model=self.config.model_name,
                dimension=None,
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

        # Determine dimension from the model or first non-empty embedding
        dim = model.get_sentence_embedding_dimension()

        return EmbeddingResult(
            embeddings=results,
            model=self.config.model_name,
            dimension=dim,
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
            "device": str(model.device),
        }
