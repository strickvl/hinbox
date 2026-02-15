"""Tests for recovery from Instructor multiple-tool-call responses."""

import json
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import instructor
from pydantic import BaseModel

from src.utils.llm import _recover_multiple_tool_calls, cloud_generation


class FakeEntity(BaseModel):
    name: str
    type: str


class _BadParallelResult:
    def __iter__(self):
        raise TypeError("'NoneType' object is not iterable")


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

    with patch("src.utils.llm.get_litellm_client", return_value=fake_client):
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


def test_cloud_generation_uses_parallel_tools_mode_for_list_models():
    mode_seen = {}
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(
                    return_value=iter([FakeEntity(name="Alice", type="detainee")])
                )
            )
        )
    )

    def _fake_get_client(mode=instructor.Mode.TOOLS):
        mode_seen["mode"] = mode
        return fake_client

    with patch("src.utils.llm.get_litellm_client", side_effect=_fake_get_client):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice is mentioned."},
            ],
            response_model=List[FakeEntity],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )

    assert mode_seen["mode"] == instructor.Mode.PARALLEL_TOOLS
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], FakeEntity)


def test_cloud_generation_falls_back_when_parallel_tools_has_no_tool_calls():
    modes_seen = []
    parallel_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(return_value=_BadParallelResult()))
        )
    )
    tools_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(
                    return_value=[FakeEntity(name="Alice", type="detainee")]
                )
            )
        )
    )

    def _fake_get_client(mode=instructor.Mode.TOOLS):
        modes_seen.append(mode)
        if mode == instructor.Mode.PARALLEL_TOOLS:
            return parallel_client
        return tools_client

    with patch("src.utils.llm.get_litellm_client", side_effect=_fake_get_client):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice is mentioned."},
            ],
            response_model=List[FakeEntity],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )

    assert modes_seen[0] == instructor.Mode.PARALLEL_TOOLS
    assert instructor.Mode.TOOLS in modes_seen[1:]
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], FakeEntity)


def test_strategy1_uses_tools_mode_after_parallel_none_type_failure():
    modes_seen = []
    parallel_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=MagicMock(return_value=_BadParallelResult()))
        )
    )
    tools_create = MagicMock(
        side_effect=[
            RuntimeError(
                "Instructor does not support multiple tool calls, use List[Model] instead"
            ),
            [FakeEntity(name="Alice", type="detainee")],
        ]
    )
    tools_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=tools_create))
    )

    def _fake_get_client(mode=instructor.Mode.TOOLS):
        modes_seen.append(mode)
        if mode == instructor.Mode.PARALLEL_TOOLS:
            return parallel_client
        return tools_client

    with patch("src.utils.llm.get_litellm_client", side_effect=_fake_get_client):
        result = cloud_generation(
            messages=[
                {"role": "system", "content": "Extract entities"},
                {"role": "user", "content": "Alice is mentioned."},
            ],
            response_model=List[FakeEntity],
            model="gemini/gemini-2.0-flash",
            temperature=0,
        )

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], FakeEntity)
    assert modes_seen[0] == instructor.Mode.PARALLEL_TOOLS
    # One TOOLS client for nonetype fallback, another for strategy 1 retry.
    assert modes_seen.count(instructor.Mode.TOOLS) >= 2
