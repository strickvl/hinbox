"""Tests for cloud embedding provider."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.embeddings.base import EmbeddingConfig, EmbeddingResult
from src.utils.embeddings.cloud import CloudEmbeddingProvider


class TestCloudEmbeddingProvider:
    """Test cloud embedding provider functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return EmbeddingConfig(
            model_name="jina_ai/jina-embeddings-v3",
            batch_size=100,
            max_retries=3,
            timeout=30,
            metadata={"project": "test", "domain": "test"},
        )

    @pytest.fixture
    def provider(self, config):
        """Create cloud provider instance."""
        return CloudEmbeddingProvider(config)

    def test_init(self, provider, config):
        """Test provider initialization."""
        assert provider.config == config

    @pytest.mark.asyncio
    async def test_embed_single_empty_text(self, provider):
        """Test embedding empty text returns empty list."""
        result = await provider.embed_single("")
        assert result == []

        result = await provider.embed_single("   ")
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_single_success(self, provider):
        """Test successful single text embedding."""
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3, 0.4]}]
        mock_response.model = "jina-embeddings-v3"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.total_tokens = 5

        with patch.object(provider, "_call_with_retry", return_value=mock_response):
            result = await provider.embed_single("test text")
            assert result == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, provider):
        """Test batch embedding with empty texts."""
        result = await provider.embed_batch([])
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == []
        assert result.model == "jina_ai/jina-embeddings-v3"

        result = await provider.embed_batch(["", "   ", ""])
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings == [[], [], []]

    @pytest.mark.asyncio
    async def test_embed_batch_mixed(self, provider):
        """Test batch embedding with mix of empty and non-empty texts."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
        mock_response.model = "jina-embeddings-v3"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.total_tokens = 10

        with patch.object(provider, "_call_with_retry", return_value=mock_response):
            texts = ["", "valid text 1", "", "valid text 2", ""]
            result = await provider.embed_batch(texts)

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 5
            assert result.embeddings[0] == []  # empty
            assert result.embeddings[1] == [0.1, 0.2]  # valid text 1
            assert result.embeddings[2] == []  # empty
            assert result.embeddings[3] == [0.3, 0.4]  # valid text 2
            assert result.embeddings[4] == []  # empty

    @pytest.mark.asyncio
    async def test_embed_batch_success(self, provider):
        """Test successful batch embedding."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
            {"embedding": [0.7, 0.8, 0.9]},
        ]
        mock_response.model = "jina-embeddings-v3"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.total_tokens = 15

        with patch.object(provider, "_call_with_retry", return_value=mock_response):
            texts = ["text1", "text2", "text3"]
            result = await provider.embed_batch(texts)

            assert isinstance(result, EmbeddingResult)
            assert len(result.embeddings) == 3
            assert result.embeddings[0] == [0.1, 0.2, 0.3]
            assert result.embeddings[1] == [0.4, 0.5, 0.6]
            assert result.embeddings[2] == [0.7, 0.8, 0.9]
            assert (
                result.model == "jina_ai/jina-embeddings-v3"
            )  # stable configured name
            assert (
                result.metadata["resolved_model"] == "jina-embeddings-v3"
            )  # provider-returned
            assert result.dimension == 3
            assert result.usage == {"prompt_tokens": 15, "total_tokens": 15}

    @pytest.mark.asyncio
    async def test_call_with_retry_success(self, provider):
        """Test successful API call."""
        mock_response = MagicMock()

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = mock_response

            result = await provider._call_with_retry(["test"])

            assert result == mock_response
            mock_embedding.assert_called_once()
            call_args = mock_embedding.call_args[1]
            assert call_args["model"] == "jina_ai/jina-embeddings-v3"
            assert call_args["input"] == ["test"]
            assert call_args["metadata"]["batch_size"] == 1
            assert call_args["metadata"]["attempt"] == 1

    @pytest.mark.asyncio
    async def test_call_with_retry_failure_then_success(self, provider):
        """Test retry logic on failure."""
        mock_response = MagicMock()

        with patch("litellm.embedding") as mock_embedding:
            # First call fails, second succeeds
            mock_embedding.side_effect = [Exception("API Error"), mock_response]

            with patch("asyncio.sleep") as mock_sleep:
                result = await provider._call_with_retry(["test"])

                assert result == mock_response
                assert mock_embedding.call_count == 2
                mock_sleep.assert_called_once_with(1.0)  # 2^0 = 1

    @pytest.mark.asyncio
    async def test_call_with_retry_all_failures(self, provider):
        """Test that all retries are exhausted."""
        provider.config.max_retries = 2

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.side_effect = Exception("API Error")

            with patch("asyncio.sleep") as mock_sleep:
                with pytest.raises(Exception, match="API Error"):
                    await provider._call_with_retry(["test"])

                assert mock_embedding.call_count == 2
                assert mock_sleep.call_count == 1

    def test_get_model_info(self, provider):
        """Test model info retrieval."""
        info = provider.get_model_info()

        assert info["type"] == "cloud"
        assert info["model_name"] == "jina_ai/jina-embeddings-v3"
        assert info["provider"] == "jina_ai"
        assert info["max_batch_size"] == 100
