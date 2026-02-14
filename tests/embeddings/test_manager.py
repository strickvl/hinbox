"""Tests for embedding manager."""

import os
from unittest.mock import patch

import pytest

from src.utils.embeddings.base import EmbeddingResult
from src.utils.embeddings.manager import (
    EmbeddingManager,
    EmbeddingMode,
)
from src.utils.embeddings.similarity import compute_similarity


class TestEmbeddingManager:
    """Test embedding manager functionality."""

    @pytest.fixture
    def mock_domain_config(self):
        """Mock domain configuration."""
        return {
            "embeddings": {
                "mode": "local",
                "cloud": {
                    "model": "jina_ai/jina-embeddings-v3",
                    "batch_size": 100,
                },
                "local": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "batch_size": 32,
                },
            }
        }

    def test_init_with_mode(self):
        """Test initialization with explicit mode."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)
        assert manager.mode == EmbeddingMode.LOCAL
        assert manager.local_provider is not None
        assert manager.cloud_provider is None

    def test_init_auto_mode_with_local_available(self):
        """Test AUTO mode resolves to LOCAL when sentence-transformers is available."""
        with patch(
            "src.utils.embeddings.manager._local_backend_available", return_value=True
        ):
            manager = EmbeddingManager(mode=EmbeddingMode.AUTO)
            assert manager.mode == EmbeddingMode.LOCAL
            assert manager.local_provider is not None
            assert manager.cloud_provider is None

    def test_init_auto_mode_without_local(self, mock_domain_config):
        """Test AUTO mode resolves to CLOUD when sentence-transformers is not available."""
        with patch(
            "src.utils.embeddings.manager._local_backend_available", return_value=False
        ):
            with patch.object(
                EmbeddingManager,
                "_load_domain_config",
                return_value=mock_domain_config,
            ):
                manager = EmbeddingManager(mode=EmbeddingMode.AUTO)
                assert manager.mode == EmbeddingMode.CLOUD
                assert manager.cloud_provider is not None
                assert manager.local_provider is None

    def test_init_with_env_var(self, mock_domain_config):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"EMBEDDING_MODE": "cloud"}):
            with patch.object(
                EmbeddingManager, "_load_domain_config", return_value=mock_domain_config
            ):
                manager = EmbeddingManager()
                assert manager.mode == EmbeddingMode.CLOUD
                assert manager.cloud_provider is not None
                assert manager.local_provider is None

    def test_init_with_invalid_env_var(self, mock_domain_config):
        """Test initialization with invalid environment variable."""
        with patch.dict(os.environ, {"EMBEDDING_MODE": "invalid"}):
            with patch.object(
                EmbeddingManager, "_load_domain_config", return_value=mock_domain_config
            ):
                manager = EmbeddingManager()
                # Should fall back to config
                assert manager.mode == EmbeddingMode.LOCAL

    def test_init_hybrid_mode(self, mock_domain_config):
        """Test initialization in hybrid mode."""
        mock_domain_config["embeddings"]["mode"] = "hybrid"
        with patch.object(
            EmbeddingManager, "_load_domain_config", return_value=mock_domain_config
        ):
            manager = EmbeddingManager()
            assert manager.mode == EmbeddingMode.HYBRID
            assert manager.cloud_provider is not None
            assert manager.local_provider is not None

    @pytest.mark.asyncio
    async def test_embed_text_local_mode(self):
        """Test single text embedding in local mode."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)

        mock_embedding = [0.1, 0.2, 0.3]
        with patch.object(
            manager.local_provider, "embed_single", return_value=mock_embedding
        ) as mock_embed:
            result = await manager.embed_text("test text")

            assert result == mock_embedding
            mock_embed.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_embed_text_cloud_mode(self):
        """Test single text embedding in cloud mode."""
        manager = EmbeddingManager(mode=EmbeddingMode.CLOUD)

        mock_embedding = [0.4, 0.5, 0.6]
        with patch.object(
            manager.cloud_provider, "embed_single", return_value=mock_embedding
        ) as mock_embed:
            result = await manager.embed_text("test text")

            assert result == mock_embedding
            mock_embed.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_embed_text_hybrid_mode_success(self):
        """Test hybrid mode with successful cloud call."""
        manager = EmbeddingManager(mode=EmbeddingMode.HYBRID)

        mock_embedding = [0.7, 0.8, 0.9]
        with patch.object(
            manager.cloud_provider, "embed_single", return_value=mock_embedding
        ) as mock_cloud:
            with patch.object(manager.local_provider, "embed_single") as mock_local:
                result = await manager.embed_text("test text")

                assert result == mock_embedding
                mock_cloud.assert_called_once_with("test text")
                mock_local.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_text_hybrid_mode_fallback(self):
        """Test hybrid mode fallback to local on cloud failure."""
        manager = EmbeddingManager(mode=EmbeddingMode.HYBRID)

        mock_local_embedding = [0.1, 0.2, 0.3]
        with patch.object(
            manager.cloud_provider, "embed_single", side_effect=Exception("Cloud error")
        ) as mock_cloud:
            with patch.object(
                manager.local_provider,
                "embed_single",
                return_value=mock_local_embedding,
            ) as mock_local:
                result = await manager.embed_text("test text")

                assert result == mock_local_embedding
                mock_cloud.assert_called_once_with("test text")
                mock_local.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_embed_batch_local_mode(self):
        """Test batch embedding in local mode."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)

        mock_result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            model="test-model",
        )

        with patch.object(
            manager.local_provider, "embed_batch", return_value=mock_result
        ) as mock_embed:
            result = await manager.embed_batch(["text1", "text2", "text3"])

            assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            mock_embed.assert_called_once_with(["text1", "text2", "text3"])

    def test_compute_similarity(self):
        """Test similarity computation."""
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        similarity = compute_similarity(vec1, vec1)
        assert similarity == pytest.approx(1.0)

        # Test orthogonal vectors
        vec2 = [0.0, 1.0, 0.0]
        similarity = compute_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)

        # Test opposite vectors
        vec3 = [-1.0, 0.0, 0.0]
        similarity = compute_similarity(vec1, vec3)
        assert similarity == pytest.approx(-1.0)

        # Test empty vectors
        assert compute_similarity([], [1, 2, 3]) == 0.0
        assert compute_similarity([1, 2, 3], []) == 0.0
        assert compute_similarity([], []) == 0.0

        # Test zero vectors
        zero_vec = [0.0, 0.0, 0.0]
        assert compute_similarity(zero_vec, vec1) == 0.0

    def test_sync_wrappers(self):
        """Test synchronous wrapper methods."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)

        # Mock the async methods
        mock_embedding = [0.1, 0.2, 0.3]
        mock_batch = [[0.1, 0.2], [0.3, 0.4]]

        with patch.object(manager, "embed_text", return_value=mock_embedding):
            result = manager.embed_text_sync("test")
            assert result == mock_embedding

        with patch.object(manager, "embed_batch", return_value=mock_batch):
            result = manager.embed_batch_sync(["test1", "test2"])
            assert result == mock_batch

    def test_embed_text_result_sync(self):
        """Test embed_text_result_sync returns full EmbeddingResult with metadata."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)

        mock_result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            dimension=3,
        )
        with patch.object(manager, "embed_text_result", return_value=mock_result):
            result = manager.embed_text_result_sync("test")
            assert result.model == "test-model"
            assert result.dimension == 3
            assert result.embeddings == [[0.1, 0.2, 0.3]]

    def test_get_active_model_name(self):
        """Test get_active_model_name returns primary provider's model."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)
        assert (
            manager.get_active_model_name() == "sentence-transformers/all-MiniLM-L6-v2"
        )

        manager_cloud = EmbeddingManager(mode=EmbeddingMode.CLOUD)
        assert manager_cloud.get_active_model_name() == "jina_ai/jina-embeddings-v3"

    def test_get_provider_error(self):
        """Test error when provider not initialized."""
        manager = EmbeddingManager(mode=EmbeddingMode.LOCAL)
        manager.local_provider = None  # Force None

        with pytest.raises(ValueError, match="Local provider not initialized"):
            manager._get_provider()

        manager = EmbeddingManager(mode=EmbeddingMode.CLOUD)
        manager.cloud_provider = None  # Force None

        with pytest.raises(ValueError, match="Cloud provider not initialized"):
            manager._get_provider()

    def test_load_domain_config_error_handling(self):
        """Test domain config loading error handling."""
        with patch(
            "src.config_loader.DomainConfig", side_effect=Exception("Config error")
        ):
            manager = EmbeddingManager()
            # Should return empty dict on error
            config = manager._load_domain_config()
            assert config == {}

    def test_fingerprint_from_result(self):
        """fingerprint_from_result should produce 'model:dim' format."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="jina_ai/jina-embeddings-v3",
            dimension=3,
        )
        fp = EmbeddingManager.fingerprint_from_result(result)
        assert fp == "jina_ai/jina-embeddings-v3:3"

    def test_fingerprint_from_result_infers_dim(self):
        """When dimension is None, fingerprint should infer from vector length."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            model="test-model",
            dimension=None,
        )
        fp = EmbeddingManager.fingerprint_from_result(result)
        assert fp == "test-model:4"

    def test_fingerprint_from_result_no_model(self):
        """No model name should return None."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]],
            model="",
            dimension=2,
        )
        fp = EmbeddingManager.fingerprint_from_result(result)
        assert fp is None

    def test_make_fingerprint(self):
        """make_fingerprint should return 'model:dim' or None."""
        assert EmbeddingManager.make_fingerprint("m", 3) == "m:3"
        assert EmbeddingManager.make_fingerprint(None, 3) is None
        assert EmbeddingManager.make_fingerprint("m", None) is None

    def test_get_configs_from_domain(self):
        """Test configuration extraction from domain config."""
        manager = EmbeddingManager()

        # Test cloud config
        embedding_config = {
            "cloud": {
                "model": "test-cloud-model",
                "batch_size": 50,
                "max_retries": 5,
                "timeout": 60,
            }
        }
        cloud_config = manager._get_cloud_config_from_domain(embedding_config)
        assert cloud_config.model_name == "test-cloud-model"
        assert cloud_config.batch_size == 50
        assert cloud_config.max_retries == 5
        assert cloud_config.timeout == 60

        # Test local config
        embedding_config = {
            "local": {
                "model": "test-local-model",
                "batch_size": 16,
            }
        }
        local_config = manager._get_local_config_from_domain(embedding_config)
        assert local_config.model_name == "test-local-model"
        assert local_config.batch_size == 16

        # Test with empty config (should use defaults)
        cloud_config = manager._get_cloud_config_from_domain({})
        assert cloud_config.model_name == "jina_ai/jina-embeddings-v3"
        assert cloud_config.batch_size == 100
