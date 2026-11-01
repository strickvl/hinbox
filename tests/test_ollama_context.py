"""Prompts too long for a local model's context are refused, not cut off."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils import ollama_context
from src.utils.ollama_context import PromptTooLongError, check_prompt_fits
from src.utils.provider_routing import ProviderTarget

OLLAMA = ProviderTarget(
    sdk="openai",
    provider_label="ollama",
    api_model="local-qwen36-16k:latest",
    base_url="http://beast:11434/v1",
    is_local=True,
)
CLOUD = ProviderTarget(sdk="openai", provider_label="openai", api_model="gpt-6-luna")


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.delenv("HINBOX_OLLAMA_NUM_CTX", raising=False)
    ollama_context._fetch_num_ctx.cache_clear()
    yield
    ollama_context._fetch_num_ctx.cache_clear()


def _show(parameters: str) -> MagicMock:
    reply = MagicMock()
    reply.json.return_value = {"parameters": parameters}
    return reply


def _messages(chars: int):
    return [{"role": "system", "content": "x" * chars}]


def test_long_profile_prompt_is_refused_on_a_16k_model():
    with patch.object(ollama_context.requests, "post") as post:
        post.return_value = _show("temperature 0.7\nnum_ctx 16384\n")
        with pytest.raises(PromptTooLongError, match="16384-token context"):
            check_prompt_fits(OLLAMA, _messages(100_000))  # ~25k tokens
        check_prompt_fits(OLLAMA, _messages(40_000))  # ~10k tokens fits
        # Asked /api/show once, at the server root rather than /v1
        post.assert_called_once()
        assert post.call_args.args[0] == "http://beast:11434/api/show"


def test_unknown_context_or_cloud_model_is_never_refused():
    with patch.object(ollama_context.requests, "post") as post:
        post.side_effect = ConnectionError("beast is off")
        check_prompt_fits(OLLAMA, _messages(1_000_000))
        check_prompt_fits(CLOUD, _messages(1_000_000))
    with patch.object(ollama_context.requests, "post") as post:
        ollama_context._fetch_num_ctx.cache_clear()
        post.return_value = _show("temperature 0.7")  # no num_ctx set
        check_prompt_fits(OLLAMA, _messages(1_000_000))


def test_env_override_sets_the_limit(monkeypatch):
    monkeypatch.setenv("HINBOX_OLLAMA_NUM_CTX", "32768")
    with patch.object(ollama_context.requests, "post") as post:
        check_prompt_fits(OLLAMA, _messages(100_000))
        with pytest.raises(PromptTooLongError):
            check_prompt_fits(OLLAMA, _messages(200_000))
        post.assert_not_called()
