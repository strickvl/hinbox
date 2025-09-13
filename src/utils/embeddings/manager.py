"""Unified embedding manager for all embedding operations."""

import asyncio
import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

import src.config_loader as config_loader
from src.constants import CLOUD_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL
from src.logging_config import get_logger

from .base import EmbeddingConfig, EmbeddingProvider
from .cloud import CloudEmbeddingProvider

if TYPE_CHECKING:  # import only for type hints
    pass  # pragma: no cover

logger = get_logger("utils.embeddings.manager")


class EmbeddingMode(str, Enum):
    """Embedding generation mode."""

    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"  # Try cloud first, fallback to local if it fails


class EmbeddingManager:
    """Unified manager for all embedding operations."""

    def __init__(
        self,
        mode: Optional[EmbeddingMode] = None,
        cloud_config: Optional[EmbeddingConfig] = None,
        local_config: Optional[EmbeddingConfig] = None,
        domain: str = "guantanamo",
    ):
        self.domain = domain

        # Load configuration from domain config file
        domain_config = self._load_domain_config()
        embedding_config = domain_config.get("embeddings", {})

        # Determine mode from environment, parameter, or config
        if mode is None:
            # Check environment variable first
            env_mode = os.getenv("EMBEDDING_MODE")
            if env_mode:
                try:
                    self.mode = EmbeddingMode(env_mode.lower())
                    logger.info(f"Using embedding mode from environment: {self.mode}")
                except ValueError:
                    logger.warning(
                        f"Invalid EMBEDDING_MODE '{env_mode}', falling back to config"
                    )
                    mode_str = embedding_config.get("mode", "local")
                    self.mode = EmbeddingMode(mode_str)
            else:
                # Use config if no environment variable
                mode_str = embedding_config.get("mode", "local")
                self.mode = EmbeddingMode(mode_str)
        else:
            self.mode = mode

        # Initialize providers based on mode
        self.cloud_provider: Optional[CloudEmbeddingProvider] = None
        # Use Any for runtime attribute to avoid import side effects
        self.local_provider: Optional[Any] = None

        if self.mode in [EmbeddingMode.CLOUD, EmbeddingMode.HYBRID]:
            self.cloud_provider = CloudEmbeddingProvider(
                cloud_config or self._get_cloud_config_from_domain(embedding_config)
            )

        if self.mode in [EmbeddingMode.LOCAL, EmbeddingMode.HYBRID]:
            # Lazy import to avoid importing sentence-transformers/torch unless needed
            from .local import LocalEmbeddingProvider  # type: ignore

            self.local_provider = LocalEmbeddingProvider(
                local_config or self._get_local_config_from_domain(embedding_config)
            )

    def _load_domain_config(self) -> Dict[str, Any]:
        """Load domain-specific configuration."""
        try:
            config = config_loader.DomainConfig(self.domain)
            return config.load_config()
        except Exception:
            return {}

    def _get_cloud_config_from_domain(
        self, embedding_config: Dict[str, Any]
    ) -> EmbeddingConfig:
        """Get cloud embedding configuration from domain config."""
        cloud_config = embedding_config.get("cloud", {})
        return EmbeddingConfig(
            model_name=cloud_config.get("model", CLOUD_EMBEDDING_MODEL),
            batch_size=cloud_config.get("batch_size", 100),
            max_retries=cloud_config.get("max_retries", 3),
            timeout=cloud_config.get("timeout", 30),
            metadata={"project": "hinbox", "domain": self.domain},
        )

    def _get_local_config_from_domain(
        self, embedding_config: Dict[str, Any]
    ) -> EmbeddingConfig:
        """Get local embedding configuration from domain config."""
        local_config = embedding_config.get("local", {})
        return EmbeddingConfig(
            model_name=local_config.get("model", DEFAULT_EMBEDDING_MODEL),
            batch_size=local_config.get("batch_size", 32),
        )

    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Embed a single text with mode-appropriate provider."""
        provider = self._get_provider()

        try:
            return await provider.embed_single(text)
        except Exception as e:
            if self.mode == EmbeddingMode.HYBRID and provider == self.cloud_provider:
                logger.warning(f"Cloud embedding failed, falling back to local: {e}")
                return await self.local_provider.embed_single(text)
            raise

    async def embed_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Embed multiple texts efficiently."""
        provider = self._get_provider()

        try:
            result = await provider.embed_batch(texts)
            return result.embeddings
        except Exception as e:
            if self.mode == EmbeddingMode.HYBRID and provider == self.cloud_provider:
                logger.warning(
                    f"Cloud batch embedding failed, falling back to local: {e}"
                )
                result = await self.local_provider.embed_batch(texts)
                return result.embeddings
            raise

    def _get_provider(self) -> EmbeddingProvider:
        """Get the appropriate provider based on mode."""
        if self.mode == EmbeddingMode.CLOUD:
            if not self.cloud_provider:
                raise ValueError("Cloud provider not initialized")
            return self.cloud_provider
        elif self.mode == EmbeddingMode.LOCAL:
            if not self.local_provider:
                raise ValueError("Local provider not initialized")
            return self.local_provider
        else:  # HYBRID - prefer cloud
            if not self.cloud_provider:
                raise ValueError("Cloud provider not initialized for hybrid mode")
            return self.cloud_provider

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if not embedding1 or not embedding2:
            return 0.0

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    # Synchronous wrappers for backward compatibility
    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embed_text."""
        return asyncio.run(self.embed_text(text))

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_batch."""
        return asyncio.run(self.embed_batch(texts))


# Global default manager instance
_default_manager = None


def get_default_manager() -> EmbeddingManager:
    """Get the default embedding manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager()
    return _default_manager
