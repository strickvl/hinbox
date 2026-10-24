"""Tests for per-model request parameters and OpenAI key handling."""

from types import SimpleNamespace
from unittest.mock import patch

from src.utils import llm
from src.utils.llm import _count_tool_calls, _create_instructor_client, _sampling_kwargs
from src.utils.provider_routing import resolve_chat_target


def _kwargs(model: str, max_tokens: int = 100, temperature: float = 0.2):
    return _sampling_kwargs(resolve_chat_target(model), max_tokens, temperature)


def test_classic_openai_model_keeps_max_tokens_and_temperature():
    assert _kwargs("openai/gpt-4o") == {"max_tokens": 100, "temperature": 0.2}


def test_openai_reasoning_model_uses_max_completion_tokens_and_no_reasoning():
    assert _kwargs("openai/gpt-6-luna") == {
        "max_completion_tokens": 100,
        "reasoning_effort": "none",
    }
    assert _kwargs("openai/gpt-5.6-luna")["reasoning_effort"] == "none"


def test_openai_models_without_none_effort_get_no_reasoning_effort():
    assert _kwargs("openai/gpt-5") == {"max_completion_tokens": 100}
    assert _kwargs("openai/o3") == {"max_completion_tokens": 100}


def test_claude_5_family_drops_temperature():
    assert _kwargs("anthropic/claude-haiku-5-5") == {"max_tokens": 100}
    assert _kwargs("anthropic/claude-sonnet-4-20250514") == {
        "max_tokens": 100,
        "temperature": 0.2,
    }


def test_ollama_turns_thinking_off_by_default():
    assert _kwargs("ollama/qwen3.6:35b-a3b-q4_K_M")["reasoning_effort"] == "none"


def test_ollama_reasoning_effort_can_be_left_to_the_model():
    with patch.object(llm, "OLLAMA_REASONING_EFFORT", ""):
        assert "reasoning_effort" not in _kwargs("ollama/qwen3.6:35b")


def test_ollama_thinking_gets_a_bigger_token_limit():
    assert _kwargs("ollama/qwen3.6:35b")["max_tokens"] == 100  # thinking off
    for effort in ("", "high"):
        with patch.object(llm, "OLLAMA_REASONING_EFFORT", effort):
            assert _kwargs("ollama/nemotron")["max_tokens"] == 8192
            assert _kwargs("ollama/nemotron", max_tokens=16384)["max_tokens"] == 16384


def test_gemini_is_unchanged(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert _kwargs("gemini/gemini-2.0-flash") == {"max_tokens": 100, "temperature": 0.2}


def test_openai_client_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    with (
        patch.object(llm, "OpenAI") as fake_openai,
        patch.object(llm.instructor, "from_openai"),
    ):
        _create_instructor_client(
            resolve_chat_target("openai/gpt-6-luna"), llm.instructor.Mode.TOOLS
        )
    # None lets the OpenAI SDK pick up OPENAI_API_KEY itself
    assert fake_openai.call_args.kwargs["api_key"] is None


def test_count_tool_calls_reports_zero_for_plain_text_answers():
    err = RuntimeError("Instructor does not support multiple tool calls")
    err.last_completion = SimpleNamespace(  # type: ignore[attr-defined]
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None))]
    )
    assert _count_tool_calls(err) == "0"
    assert _count_tool_calls(RuntimeError("x")) == "an unknown number of"
