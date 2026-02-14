"""Tests for CLI --local privacy enforcement on embeddings."""

import os
from unittest.mock import patch

import pytest

from src.utils.embeddings.similarity import (
    _MANAGER_CACHE,
    ensure_local_embeddings_available,
    get_embedding_manager,
    reset_embedding_manager_cache,
)


class TestResetEmbeddingManagerCache:
    """Test the cache reset helper used during privacy enforcement."""

    def test_reset_clears_cache(self):
        """reset_embedding_manager_cache must empty the internal cache."""
        # Seed the cache with a dummy entry
        _MANAGER_CACHE[("dummy", "cloud")] = "placeholder"
        assert len(_MANAGER_CACHE) > 0

        reset_embedding_manager_cache()
        assert len(_MANAGER_CACHE) == 0

    def test_reset_is_idempotent(self):
        """Calling reset on an already-empty cache must not raise."""
        reset_embedding_manager_cache()
        reset_embedding_manager_cache()
        assert len(_MANAGER_CACHE) == 0


class TestEnsureLocalEmbeddingsAvailable:
    """Test the startup preflight check."""

    def test_raises_when_sentence_transformers_missing(self):
        """Must raise RuntimeError with install guidance when package is absent."""
        with patch("src.utils.embeddings.similarity.find_spec", return_value=None):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                ensure_local_embeddings_available()

    def test_passes_when_sentence_transformers_present(self):
        """Must not raise when the package is available."""
        with patch(
            "src.utils.embeddings.similarity.find_spec",
            return_value="<module spec>",
        ):
            # Should not raise
            ensure_local_embeddings_available()


class TestPrivacyEnforcementIntegration:
    """Test that EMBEDDING_MODE=local env var forces local mode on the manager."""

    def test_env_override_forces_local_on_cached_manager(self):
        """After setting EMBEDDING_MODE=local, get_embedding_manager must return LOCAL."""
        reset_embedding_manager_cache()
        try:
            with patch.dict(os.environ, {"EMBEDDING_MODE": "local"}):
                manager = get_embedding_manager(domain="guantanamo")
                from src.utils.embeddings.manager import EmbeddingMode

                assert manager.mode == EmbeddingMode.LOCAL
        finally:
            reset_embedding_manager_cache()
