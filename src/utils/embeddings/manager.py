"""Unified embedding manager for all embedding operations."""

import asyncio
import os
from enum import Enum
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import src.config_loader as config_loader
from src.constants import CLOUD_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL
from src.logging_config import get_logger
from src.utils.cache_utils import LRUCache, sha256_text

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

        # Embedding LRU cache keyed by (fingerprint, text_hash)
        cache_cfg = self._load_cache_config()
        emb_cache_cfg = cache_cfg.get("embeddings", {})
        lru_max = emb_cache_cfg.get("lru_max_items", 4096)
        self._cache_enabled: bool = cache_cfg.get("enabled", True) and lru_max > 0
        self._embed_lru: LRUCache = LRUCache(max_items=lru_max)
        # Resolved fingerprint per provider kind (populated on first embed call)
        self._provider_fingerprint: Dict[str, Optional[str]] = {}

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
            logger.info("AUTO mode: sentence-transformers not available → using CLOUD")
            return EmbeddingMode.CLOUD

    def _load_domain_config(self) -> Dict[str, Any]:
        """Load domain-specific configuration."""
        try:
            config = config_loader.DomainConfig(self.domain)
            return config.load_config()
        except Exception:
            return {}

    def _load_cache_config(self) -> Dict[str, Any]:
        """Load cache configuration from domain config."""
        try:
            dc = config_loader.DomainConfig(self.domain)
            return dc.get_cache_config()
        except Exception:
            return {"enabled": True, "embeddings": {"lru_max_items": 4096}}

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

    def _provider_kind(self) -> str:
        """Return a string tag for the current primary provider (for cache key segregation)."""
        if self.mode == EmbeddingMode.LOCAL:
            return "local"
        return "cloud"

    def _try_cache_lookup(self, text: str) -> Optional[List[float]]:
        """Look up a single text in the LRU cache. Returns None on miss."""
        if not self._cache_enabled:
            return None
        fp = self._provider_fingerprint.get(self._provider_kind())
        if fp is None:
            return None
        key = (fp, sha256_text(text))
        return self._embed_lru.get(key)

    def _cache_store(self, text: str, vec: List[float], fingerprint: str) -> None:
        """Store a single (text → vec) mapping in the LRU cache."""
        if not self._cache_enabled:
            return
        key = (fingerprint, sha256_text(text))
        self._embed_lru.set(key, vec)

    def _update_fingerprint_from_result(self, result: EmbeddingResult) -> Optional[str]:
        """Compute and cache the provider fingerprint from an EmbeddingResult."""
        fp = self.fingerprint_from_result(result)
        if fp:
            self._provider_fingerprint[self._provider_kind()] = fp
        return fp

    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Embed a single text with mode-appropriate provider."""
        if use_cache:
            cached = self._try_cache_lookup(text)
            if cached is not None:
                return cached

        provider = self._get_provider()
        try:
            result = await provider.embed_batch([text])
            fp = self._update_fingerprint_from_result(result)
            vec = result.embeddings[0] if result.embeddings else []
            if fp and use_cache:
                self._cache_store(text, vec, fp)
            return vec
        except Exception as e:
            if self.mode == EmbeddingMode.HYBRID and provider == self.cloud_provider:
                logger.warning(f"Cloud embedding failed, falling back to local: {e}")
                return await self.local_provider.embed_single(text)
            raise

    async def embed_batch(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Embed multiple texts efficiently with per-text cache lookups."""
        if not texts:
            return []

        # Split into hits and misses
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices: List[int] = []

        if use_cache and self._cache_enabled:
            for i, t in enumerate(texts):
                cached = self._try_cache_lookup(t)
                if cached is not None:
                    results[i] = cached
                else:
                    miss_indices.append(i)
        else:
            miss_indices = list(range(len(texts)))

        if not miss_indices:
            return [r for r in results if r is not None]

        # Embed only misses
        miss_texts = [texts[i] for i in miss_indices]
        provider = self._get_provider()

        try:
            batch_result = await provider.embed_batch(miss_texts)
        except Exception as e:
            if self.mode == EmbeddingMode.HYBRID and provider == self.cloud_provider:
                logger.warning(
                    f"Cloud batch embedding failed, falling back to local: {e}"
                )
                batch_result = await self.local_provider.embed_batch(miss_texts)
            else:
                raise

        fp = self._update_fingerprint_from_result(batch_result)

        for j, idx in enumerate(miss_indices):
            vec = batch_result.embeddings[j] if j < len(batch_result.embeddings) else []
            results[idx] = vec
            if fp and use_cache:
                self._cache_store(texts[idx], vec, fp)

        return [r for r in results if r is not None]

    async def embed_text_result(self, text: str) -> EmbeddingResult:
        """Embed a single text and return full EmbeddingResult with metadata."""
        provider = self._get_provider()

        try:
            result = await provider.embed_batch([text])
            self._update_fingerprint_from_result(result)
            # Cache the vector
            fp = self._provider_fingerprint.get(self._provider_kind())
            if fp and result.embeddings:
                self._cache_store(text, result.embeddings[0], fp)
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

    @property
    def cache_stats(self) -> dict:
        """Return LRU cache hit/miss statistics for diagnostics."""
        return self._embed_lru.stats

    @staticmethod
    def fingerprint_from_result(result: EmbeddingResult) -> Optional[str]:
        """Build a stable fingerprint string from an EmbeddingResult.

        Format: "{model}:{dimension}" — used to detect when an entity's
        stored embedding was produced by a different model/dimension than
        the currently active one.
        """
        if not result.model:
            return None
        dim = result.dimension
        if dim is None and result.embeddings:
            dim = len(result.embeddings[0])
        return f"{result.model}:{dim}" if dim is not None else None

    @staticmethod
    def make_fingerprint(model: Optional[str], dim: Optional[int]) -> Optional[str]:
        """Build a fingerprint from raw model name and dimension."""
        if not model or dim is None:
            return None
        return f"{model}:{dim}"

    # --- Synchronous wrappers ---
    # Use a single cached event loop to avoid asyncio.run() overhead per call.

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return a persistent event loop for sync wrappers."""
        if not hasattr(self, "_loop") or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def embed_text_sync(self, text: str) -> List[float]:
        """Synchronous wrapper for embed_text."""
        return self._get_loop().run_until_complete(self.embed_text(text))

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous wrapper for embed_batch."""
        return self._get_loop().run_until_complete(self.embed_batch(texts))

    def embed_text_result_sync(self, text: str) -> EmbeddingResult:
        """Synchronous wrapper for embed_text_result — returns full metadata."""
        return self._get_loop().run_until_complete(self.embed_text_result(text))

    def embed_batch_result_sync(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple texts synchronously, returning full metadata.

        This is the preferred method for batching — sentence-transformers
        encodes a batch in a single GPU/CPU pass, much faster than N × single.
        Uses the LRU cache to skip already-embedded texts.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=None, dimension=None)

        # Split into cache hits and misses
        all_vecs: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices: List[int] = []

        if self._cache_enabled:
            for i, t in enumerate(texts):
                cached = self._try_cache_lookup(t)
                if cached is not None:
                    all_vecs[i] = cached
                else:
                    miss_indices.append(i)
        else:
            miss_indices = list(range(len(texts)))

        provider = self._get_provider()

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]

            async def _batch():
                try:
                    return await provider.embed_batch(miss_texts)
                except Exception as e:
                    if (
                        self.mode == EmbeddingMode.HYBRID
                        and provider == self.cloud_provider
                    ):
                        logger.warning(
                            f"Cloud batch embedding failed, falling back to local: {e}"
                        )
                        return await self.local_provider.embed_batch(miss_texts)
                    raise

            batch_result = self._get_loop().run_until_complete(_batch())
            fp = self._update_fingerprint_from_result(batch_result)

            for j, idx in enumerate(miss_indices):
                vec = (
                    batch_result.embeddings[j]
                    if j < len(batch_result.embeddings)
                    else []
                )
                all_vecs[idx] = vec
                if fp:
                    self._cache_store(texts[idx], vec, fp)

            # Build result with full metadata from the provider call
            final_embeddings = [v for v in all_vecs if v is not None]
            return EmbeddingResult(
                embeddings=final_embeddings,
                model=batch_result.model,
                dimension=batch_result.dimension,
            )
        else:
            # All hits — reconstruct metadata from cached fingerprint
            final_embeddings = [v for v in all_vecs if v is not None]
            fp = self._provider_fingerprint.get(self._provider_kind())
            model_name = None
            dim = None
            if fp and ":" in fp:
                parts = fp.rsplit(":", 1)
                model_name = parts[0]
                try:
                    dim = int(parts[1])
                except ValueError:
                    dim = len(final_embeddings[0]) if final_embeddings else None
            return EmbeddingResult(
                embeddings=final_embeddings,
                model=model_name,
                dimension=dim,
            )


# Global default manager instance
_default_manager = None


def get_default_manager() -> EmbeddingManager:
    """Get the default embedding manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = EmbeddingManager()
    return _default_manager
