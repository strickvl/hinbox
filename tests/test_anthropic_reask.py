"""Instructor's Anthropic re-ask must answer every tool call Claude made."""

from types import SimpleNamespace

import pytest
from instructor import Mode
from instructor.processing.response import handle_reask_kwargs

import src.utils.llm  # noqa: F401  (installs the re-ask on import)


def _block(kind: str, i: str = "") -> SimpleNamespace:
    data = (
        {"type": kind, "id": i, "name": "Verdicts", "input": {"items": []}}
        if i
        else {"type": kind, "text": "ok"}
    )
    return SimpleNamespace(**data, model_dump=lambda: dict(data))


def _reask(content: list, mode: Mode = Mode.ANTHROPIC_REASONING_TOOLS) -> list:
    kwargs = {"messages": [{"role": "user", "content": "judge these"}]}
    response = SimpleNamespace(content=content)
    return handle_reask_kwargs(
        kwargs, mode, response, ValueError("wanted one tool call")
    )["messages"]


def test_reask_answers_both_tool_calls_when_claude_made_two():
    for mode in (Mode.ANTHROPIC_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS):
        messages = _reask(
            [_block("text"), _block("tool_use", "tu_1"), _block("tool_use", "tu_2")],
            mode,
        )
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert [b["type"] for b in messages[1]["content"]] == [
            "text",
            "tool_use",
            "tool_use",
        ]
        results = messages[2]["content"]
        assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
        assert all(r["type"] == "tool_result" and r["is_error"] for r in results)
        assert "once, not 2 times" in results[0]["content"]


def test_reask_with_one_tool_call_is_instructors_own():
    types = pytest.importorskip("anthropic.types")
    response = types.Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-x",
        stop_reason="tool_use",
        stop_sequence=None,
        content=[
            types.ToolUseBlock(
                id="tu_1", type="tool_use", name="Verdicts", input={"items": []}
            )
        ],
        usage={"input_tokens": 1, "output_tokens": 1},
    )
    kwargs = {"messages": [{"role": "user", "content": "judge these"}]}
    messages = handle_reask_kwargs(
        kwargs, Mode.ANTHROPIC_REASONING_TOOLS, response, ValueError("bad")
    )["messages"]
    assert [r["tool_use_id"] for r in messages[2]["content"]] == ["tu_1"]
    assert "once, not" not in messages[2]["content"][0]["content"]
