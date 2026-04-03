"""Tests for provider routing — model prefix → SDK target resolution."""

import os
from unittest.mock import patch

import pytest

from src.utils.provider_routing import (
    _split_model_prefix,
    resolve_chat_target,
    resolve_embedding_target,
)

# ---------------------------------------------------------------------------
# _split_model_prefix
# ---------------------------------------------------------------------------


class TestSplitModelPrefix:
    def test_gemini_prefix(self):
        assert _split_model_prefix("gemini/gemini-2.0-flash") == (
            "gemini",
            "gemini-2.0-flash",
        )

    def test_ollama_prefix(self):
        assert _split_model_prefix("ollama/qwen2.5:32b-instruct-q5_K_M") == (
            "ollama",
            "qwen2.5:32b-instruct-q5_K_M",
        )

    def test_anthropic_prefix(self):
        assert _split_model_prefix("anthropic/claude-sonnet-4-20250514") == (
            "anthropic",
            "claude-sonnet-4-20250514",
        )

    def test_bare_model_treated_as_openai(self):
        assert _split_model_prefix("gpt-4o") == ("openai", "gpt-4o")

    def test_jina_ai_prefix(self):
        assert _split_model_prefix("jina_ai/jina-embeddings-v3") == (
            "jina_ai",
            "jina-embeddings-v3",
        )

    def test_openrouter_prefix(self):
        assert _split_model_prefix("openrouter/anthropic/claude-3.5-sonnet") == (
            "openrouter",
            "anthropic/claude-3.5-sonnet",
        )

    def test_prefix_is_lowered(self):
        assert _split_model_prefix("Gemini/gemini-2.0-flash")[0] == "gemini"


# ---------------------------------------------------------------------------
# resolve_chat_target
# ---------------------------------------------------------------------------


class TestResolveChatTarget:
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"})
    def test_gemini_target(self):
        target = resolve_chat_target("gemini/gemini-2.0-flash")
        assert target.sdk == "openai"
        assert target.provider_label == "gemini"
        assert target.api_model == "gemini-2.0-flash"
        assert "googleapis.com" in target.base_url
        assert target.api_key == "test-gemini-key"
        assert target.is_local is False

    def test_openai_target(self):
        target = resolve_chat_target("openai/gpt-4o")
        assert target.sdk == "openai"
        assert target.provider_label == "openai"
        assert target.api_model == "gpt-4o"
        assert target.base_url is None
        assert target.api_key is None
        assert target.is_local is False

    def test_anthropic_target(self):
        target = resolve_chat_target("anthropic/claude-sonnet-4-20250514")
        assert target.sdk == "anthropic"
        assert target.provider_label == "anthropic"
        assert target.api_model == "claude-sonnet-4-20250514"
        assert target.base_url is None
        assert target.api_key is None
        assert target.is_local is False

    def test_ollama_target(self):
        target = resolve_chat_target("ollama/qwen2.5:32b")
        assert target.sdk == "openai"
        assert target.provider_label == "ollama"
        assert target.api_model == "qwen2.5:32b"
        assert target.base_url is not None  # OLLAMA_API_URL
        assert target.api_key == "ollama"
        assert target.is_local is True

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-or-key"})
    def test_openrouter_target(self):
        target = resolve_chat_target("openrouter/anthropic/claude-3.5-sonnet")
        assert target.sdk == "openai"
        assert target.provider_label == "openrouter"
        assert target.api_model == "anthropic/claude-3.5-sonnet"
        assert "openrouter.ai" in target.base_url
        assert target.api_key == "test-or-key"

    def test_bare_model_treated_as_openai(self):
        target = resolve_chat_target("gpt-4o")
        assert target.sdk == "openai"
        assert target.provider_label == "openai"
        assert target.api_model == "gpt-4o"

    def test_gemini_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove GEMINI_API_KEY if set
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                resolve_chat_target("gemini/gemini-2.0-flash")

    def test_openrouter_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                resolve_chat_target("openrouter/some-model")

    def test_unknown_prefix_fallback(self):
        target = resolve_chat_target("deepseek/deepseek-chat")
        assert target.sdk == "openai"
        assert target.provider_label == "deepseek"
        assert target.api_model == "deepseek-chat"


# ---------------------------------------------------------------------------
# resolve_embedding_target
# ---------------------------------------------------------------------------


class TestResolveEmbeddingTarget:
    @patch.dict(os.environ, {"JINA_API_KEY": "test-jina-key"})
    def test_jina_target(self):
        target = resolve_embedding_target("jina_ai/jina-embeddings-v3")
        assert target.sdk == "openai"
        assert target.provider_label == "jina_ai"
        assert target.api_model == "jina-embeddings-v3"
        assert "jina.ai" in target.base_url
        assert target.api_key == "test-jina-key"

    def test_openai_embedding_target(self):
        target = resolve_embedding_target("openai/text-embedding-3-small")
        assert target.sdk == "openai"
        assert target.provider_label == "openai"
        assert target.api_model == "text-embedding-3-small"
        assert target.base_url is None

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"})
    def test_gemini_embedding_target(self):
        target = resolve_embedding_target("gemini/text-embedding-004")
        assert target.sdk == "openai"
        assert target.provider_label == "gemini"
        assert "googleapis.com" in target.base_url

    def test_jina_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JINA_API_KEY", None)
            with pytest.raises(RuntimeError, match="JINA_API_KEY"):
                resolve_embedding_target("jina_ai/jina-embeddings-v3")
