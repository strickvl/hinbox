"""Cloud embedding provider using OpenAI-compatible APIs."""

import asyncio
from typing import Any, Dict, List

from openai import OpenAI

from src.logging_config import get_logger
from src.utils.provider_routing import resolve_embedding_target

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingResult

logger = get_logger("utils.embeddings.cloud")


class CloudEmbeddingProvider(EmbeddingProvider):
    """Cloud embedding provider using OpenAI-compatible endpoints (Jina, OpenAI, etc.)."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text using cloud API."""
        if not text.strip():
            return []

        result = await self.embed_batch([text])
        return result.embeddings[0]

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
                non_empty_indices[i]: item.embedding
                for i, item in enumerate(response.data)
            }

            # Reconstruct full results
            results = []
            for i in range(len(texts)):
                results.append(embeddings_map.get(i, []))

            # Determine dimension from the first non-empty embedding
            dim = next((len(emb) for emb in results if emb), None)

            return EmbeddingResult(
                embeddings=results,
                model=self.config.model_name,  # stable configured name
                dimension=dim,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                metadata={"resolved_model": response.model},
            )

        except Exception as e:
            logger.error(f"Cloud embedding failed: {e}")
            raise

    async def _call_with_retry(self, texts: List[str]):
        """Call cloud API with exponential backoff retry."""
        loop = asyncio.get_running_loop()
        target = resolve_embedding_target(self.config.model_name)
        client = OpenAI(
            base_url=target.base_url,
            api_key=target.api_key or "not-set",
        )

        for attempt in range(self.config.max_retries):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.embeddings.create(
                        model=target.api_model,
                        input=texts,
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
