"""Cloud embedding provider using LiteLLM."""

import asyncio
from typing import Any, Dict, List

import litellm
from langfuse import observe
from litellm import embedding

from src.logging_config import get_logger

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingResult

logger = get_logger("utils.embeddings.cloud")


class CloudEmbeddingProvider(EmbeddingProvider):
    """Cloud embedding provider using LiteLLM."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._setup_litellm()

    def _setup_litellm(self):
        """Configure LiteLLM for embeddings."""
        litellm.suppress_debug_info = True
        litellm.callbacks = ["braintrust"]

    @observe()
    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text using cloud API."""
        if not text.strip():
            return []

        result = await self.embed_batch([text])
        return result.embeddings[0]

    @observe()
    async def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple texts using cloud API."""
        # Filter empty texts
        non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return EmbeddingResult(
                embeddings=[[] for _ in texts], model=self.config.model_name
            )

        try:
            # Call cloud API with batching
            response = await self._call_with_retry(non_empty_texts)

            # Extract embeddings
            embeddings_map = {
                non_empty_indices[i]: data["embedding"]
                for i, data in enumerate(response.data)
            }

            # Reconstruct full results
            results = []
            for i in range(len(texts)):
                results.append(embeddings_map.get(i, []))

            return EmbeddingResult(
                embeddings=results,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )

        except Exception as e:
            logger.error(f"Cloud embedding failed: {e}")
            raise

    async def _call_with_retry(self, texts: List[str]):
        """Call cloud API with exponential backoff retry."""
        loop = asyncio.get_event_loop()

        for attempt in range(self.config.max_retries):
            try:
                # Run the synchronous embedding call in an executor
                response = await loop.run_in_executor(
                    None,
                    lambda: embedding(
                        model=self.config.model_name,
                        input=texts,
                        metadata={
                            **self.config.metadata,
                            "batch_size": len(texts),
                            "attempt": attempt + 1,
                        },
                    ),
                )
                return response
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise

                wait_time = (2**attempt) * 1.0
                logger.warning(
                    f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)

    def get_model_info(self) -> Dict[str, Any]:
        """Get cloud model information."""
        return {
            "type": "cloud",
            "model_name": self.config.model_name,
            "provider": self.config.model_name.split("/")[0],
            "max_batch_size": 100,  # Jina AI limit
        }
