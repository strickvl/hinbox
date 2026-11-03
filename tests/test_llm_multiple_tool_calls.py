"""Tests for recovery from Instructor multiple-tool-call responses."""

import json
import os
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import instructor
from pydantic import BaseModel

from src.utils.llm import (
    _recover_multiple_tool_calls,
    cloud_generation,
    local_generation,
)


class FakeEntity(BaseModel):
    name: str
    type: str


def _make_multiple_tool_calls_error(payloads: List[dict]) -> Exception:
    """Build an exception carrying a synthetic ``last_completion`` payload."""
    tool_calls = [
        SimpleNamespace(function=SimpleNamespace(arguments=json.dumps(payload)))
        for payload in payloads
    ]
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))]
    )

    err = RuntimeError(
        "Instructor does not support multiple tool calls, use List[Model] instead"
    )
    err.last_completion = completion  # type: ignore[attr-defined]
    return err


# Env patch needed because resolve_chat_target reads GEMINI_API_KEY for gemini/* models
_GEMINI_ENV = {"GEMINI_API_KEY": "test-key"}


def test_recover_multiple_tool_calls_for_list_model():
    err = _make_multiple_tool_calls_error(
        [
            {"name": "Alice", "type": "detainee"},
            {"name": "Bob", "type": "military_personnel"},
        ]
    )

    recovered = _recover_multiple_tool_calls(
        error=err,
        response_model=List[FakeEntity],
    )

    assert recovered is not None
    assert len(recovered) == 2
    assert all(isinstance(item, FakeEntity) for item in recovered)
    assert recovered[0].name == "Alice"
    assert recovered[1].name == "Bob"


def test_recover_fenced_json_array_when_model_skips_tool_calls():
    # Qwen 3.6 on Ollama sometimes answers in text instead of calling the tool.
    content = '```json\n[\n  {"name": "Cuba", "type": "country"},\n  {"name": "El Paso", "type": "city"}\n]\n```'
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=content))
        ]
    )
    err = RuntimeError(
        "Instructor does not support multiple tool calls, use List[Model] instead"
    )
    err.last_completion = completion  # type: ignore[attr-defined]

    recovered = _recover_multiple_tool_calls(error=err, response_model=List[FakeEntity])

    assert [e.name for e in recovered] == ["Cuba", "El Paso"]


@patch.dict(os.environ, _GEMINI_ENV)
def test_cloud_generation_uses_direct_recovery_before_retrying():
    err = _make_multiple_tool_calls_error(
        [
            {"name": "Alice", "type": "detainee"},
            {"name": "Bob", "type": "military_personnel"},
        ]
    )

    create_mock = MagicMock(side_effect=err)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )

    with patch("src.utils.llm._create_instructor_client", return_value=fake_client):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice and Bob are mentioned."},
            ],
            response_model=List[FakeEntity],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(item, FakeEntity) for item in result)
    assert create_mock.call_count == 1


def _wrapped(**kw):
    """Fake Instructor reply: the requested {"items": [...]} wrapper, filled."""
    return kw["response_model"](
        items=[
            FakeEntity(name="Alice", type="detainee"),
            FakeEntity(name="Bob", type="military_personnel"),
        ]
    )


@patch.dict(os.environ, _GEMINI_ENV)
def test_cloud_generation_asks_for_the_whole_list_in_one_tool_call():
    """Lists go out as one {"items": [...]} tool call, never one call per item.

    In parallel-tools mode GPT-6 Luna returned 1 event per article on 20 of
    20 articles; as one object it returned 2 to 14.
    """
    modes_seen = []
    create_mock = MagicMock(side_effect=_wrapped)

    def _fake_create_client(target, mode):
        modes_seen.append(mode)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

    with patch(
        "src.utils.llm._create_instructor_client", side_effect=_fake_create_client
    ):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice and Bob are mentioned."},
            ],
            response_model=List[FakeEntity],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )

    assert modes_seen == [instructor.Mode.TOOLS]
    assert [e.name for e in result] == ["Alice", "Bob"]
    kwargs = create_mock.call_args.kwargs
    assert set(kwargs["response_model"].model_fields) == {"items"}
    assert kwargs["parallel_tool_calls"] is False


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
def test_anthropic_list_calls_use_single_tool_mode():
    modes_seen = []
    create_mock = MagicMock(side_effect=_wrapped)

    def _fake_create_client(target, mode):
        modes_seen.append(mode)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

    with patch(
        "src.utils.llm._create_instructor_client", side_effect=_fake_create_client
    ):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice and Bob are mentioned."},
            ],
            response_model=List[FakeEntity],
            model="anthropic/claude-haiku-5-5",
        )

    assert modes_seen == [instructor.Mode.ANTHROPIC_TOOLS]
    assert len(result) == 2
    assert "parallel_tool_calls" not in create_mock.call_args.kwargs


def test_recover_list_split_across_several_wrapper_tool_calls():
    err = _make_multiple_tool_calls_error(
        [
            {"items": [{"name": "Alice", "type": "detainee"}]},
            {"items": [{"name": "Bob", "type": "military_personnel"}]},
        ]
    )

    recovered = _recover_multiple_tool_calls(error=err, response_model=List[FakeEntity])

    assert [e.name for e in recovered] == ["Alice", "Bob"]


@patch.dict(os.environ, _GEMINI_ENV)
def test_strategy1_retry_unwraps_the_list():
    calls = {"n": 0}

    def _create(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(
                "Instructor does not support multiple tool calls, use List[Model] instead"
            )
        return _wrapped(**kw)

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    with patch("src.utils.llm._create_instructor_client", return_value=fake_client):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice and Bob are mentioned."},
            ],
            response_model=List[FakeEntity],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )

    assert calls["n"] == 2
    assert [e.name for e in result] == ["Alice", "Bob"]


def test_local_generation_uses_json_schema_mode_for_list_models():
    """Local list calls ask for {"items": [...]} under a JSON schema, then unwrap it."""
    modes_seen = []
    create_mock = MagicMock(
        side_effect=lambda **kw: kw["response_model"](
            items=[FakeEntity(name="Alice", type="detainee")]
        )
    )

    def _fake_create_client(target, mode):
        modes_seen.append(mode)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )

    with patch(
        "src.utils.llm._create_instructor_client", side_effect=_fake_create_client
    ):
        result = local_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice is mentioned."},
            ],
            response_model=List[FakeEntity],
            model="ollama/local-qwen36-32k",
        )

    assert modes_seen == [instructor.Mode.JSON_SCHEMA]
    assert result == [FakeEntity(name="Alice", type="detainee")]
    kwargs = create_mock.call_args.kwargs
    assert set(kwargs["response_model"].model_fields) == {"items"}
    assert "parallel_tool_calls" not in kwargs
