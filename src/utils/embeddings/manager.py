"""Unified embedding manager for all embedding operations."""

import asyncio
import os
from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import src.config_loader as config_loader
from src.constants import CLOUD_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL
from src.logging_config import get_logger

from .base import EmbeddingConfig, EmbeddingProvider, EmbeddingResult
from .cloud import CloudEmbeddingProvider

if TYPE_CHECKING:  # import only for type hints
    pass  # pragma: no cover

logger = get_logger("utils.embeddings.manager")


class EmbeddingMode(str, Enum):
    """Embedding generation mode."""

    AUTO = "auto"  # Auto-detect: local if available, else cloud
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"  # Try cloud first, fallback to local if it fails


def _local_backend_available() -> bool:
    """Check if sentence-transformers (and thus PyTorch) is importable.

    Uses find_spec to avoid actually importing torch — this is a cheap check
    that works even on platforms where torch wheels aren't installed.
    """
    return find_spec("sentence_transformers") is not None


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

        # Determine mode: explicit param > env var > domain config
        requested_mode = self._resolve_requested_mode(mode, embedding_config)
        # If AUTO, resolve to a concrete mode based on available backends
        self.mode = self._resolve_auto(requested_mode, embedding_config)

        # Initialize providers based on resolved mode
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

    def _resolve_requested_mode(
        self,
        mode_arg: Optional[EmbeddingMode],
        embedding_config: Dict[str, Any],
    ) -> EmbeddingMode:
        """Resolve the requested mode from param > env > config."""
        if mode_arg is not None:
            return mode_arg

        # Check environment variable first
        env_mode = os.getenv("EMBEDDING_MODE")
        if env_mode:
            try:
                resolved = EmbeddingMode(env_mode.lower())
                logger.info(f"Using embedding mode from environment: {resolved}")
                return resolved
            except ValueError:
                logger.warning(
                    f"Invalid EMBEDDING_MODE '{env_mode}', falling back to config"
                )

        # Use config (default to "cloud" if not specified — safe for all platforms)
        mode_str = embedding_config.get("mode", "cloud")
        return EmbeddingMode(mode_str)

    def _resolve_auto(
        self,
        requested: EmbeddingMode,
        embedding_config: Dict[str, Any],
    ) -> EmbeddingMode:
        """Resolve AUTO to a concrete mode based on available backends.

        Policy:
        - If sentence-transformers is available → LOCAL (free, fast)
        - Otherwise → CLOUD (works everywhere)
        """
        if requested != EmbeddingMode.AUTO:
            return requested

        if _local_backend_available():
            logger.info("AUTO mode: sentence-transformers available → using LOCAL")
            return EmbeddingMode.LOCAL
        else:
            logger.info(
                "AUTO mode: sentence-transformers not available → using CLOUD"
            )
            return EmbeddingMode.CLOUD

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
            device=local_config.get("device", "auto"),
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

    async def embed_text_result(self, text: str) -> EmbeddingResult:
        """Embed a single text and return full EmbeddingResult with metadata."""
        provider = self._get_provider()

        try:
            result = await provider.embed_batch([text])
            return result
        except Exception as e:
            if self.mode == EmbeddingMode.HYBRID and provider == self.cloud_provider:
                logger.warning(f"Cloud embedding failed, falling back to local: {e}")
                return await self.local_provider.embed_batch([text])
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

    def get_active_model_name(self) -> str:
        """Return the model name of the currently active (primary) provider."""
        provider = self._get_provider()
        return provider.config.model_name

    # Synchronous wrappers for backward compatibility
    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embed_text."""
        return asyncio.run(self.embed_text(text))

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_batch."""
        return asyncio.run(self.embed_batch(texts))

    def embed_text_result_sync(self, text: str) -> EmbeddingResult:
        """Synchronous wrapper for embed_text_result — returns full metadata."""
        return asyncio.run(self.embed_text_result(text))


# Global default manager instance
_default_manager = None


def get_default_manager() -> EmbeddingManager:
    """Get the default embedding manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager()
    return _default_manager
