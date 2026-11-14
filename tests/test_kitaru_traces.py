"""Trace files turned into Kitaru JSONL sessions (`scripts/kitaru_traces.py`)."""

import json
import os
from typing import Dict, List, Optional

from scripts.kitaru_traces import UNFINISHED, session, step_sessions, write_parts
from src.utils.tracing import _message_attrs


def _row(
    trace: str,
    span: str,
    parent: Optional[str] = None,
    name: str = "step",
    attrs: Optional[Dict] = None,
    error: Optional[str] = None,
    events: Optional[List] = None,
    start: str = "2026-11-09T20:00:00.000000Z",
) -> dict:
    """A span row as Hinbox's JSONL exporter writes it (OpenTelemetry's span JSON)."""
    return {
        "name": name,
        "context": {"trace_id": trace, "span_id": span, "trace_state": "[]"},
        "parent_id": parent,
        "start_time": start,
        "end_time": start.replace("20:00", "20:01"),
        "status": {"status_code": "ERROR", "description": error}
        if error
        else {"status_code": "UNSET"},
        "attributes": {"openinference.span.kind": "CHAIN", **(attrs or {})},
        "events": events or [],
        "resource": {"attributes": {"hinbox.run_id": "r1", "hinbox.env.HINBOX_X": "1"}},
    }


def _llm(trace: str, span: str, parent: str, reply: Dict, reasoning: str = "") -> dict:
    attrs = {
        "openinference.span.kind": "LLM",
        "llm.model_name": "qwen",
        "llm.provider": "ollama",
        "llm.token_count.prompt": 100,
        "llm.token_count.completion": 20,
        "llm.invocation_parameters": json.dumps(
            {"temperature": 0, "response_format": {"big": 1}}
        ),
        "input.value": "{...}",
        **_message_attrs(
            "llm.input_messages",
            [
                {"role": "system", "content": "Edit the profile"},
                {"role": "user", "content": "old profile + article"},
            ],
        ),
        **_message_attrs("llm.output_messages", [reply]),
    }
    if reasoning:
        attrs["hinbox.llm.reasoning"] = reasoning
    return _row(
        trace,
        span,
        parent,
        name="chat qwen",
        attrs=attrs,
        start="2026-11-09T20:00:05.000000Z",
    )


def _article(trace: str = "t1", outcome: str = "processed") -> dict:
    return _row(
        trace,
        f"{trace}-root",
        name="article #0",
        attrs={
            "hinbox.article.title": "Hearing",
            "hinbox.article.outcome": outcome,
            "output.value": outcome,
        },
    )


def _write(path, rows) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _read(path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_a_model_call_reads_as_a_conversation():
    reply = {"role": "assistant", "content": '{"edits": []}'}
    s = session([_article(), _llm("t1", "c1", "t1-root", reply, reasoning="thinking")])
    call = s["nodes"][1]
    assert call["node_type"] == "llm_call"
    assert call["inputs"] == [
        {"role": "system", "content": "Edit the profile"},
        {"role": "user", "content": "old profile + article"},
    ]
    assert call["outputs"] == [reply, {"role": "reasoning", "content": "thinking"}]
    assert (
        call["system_prompt_selector"],
        call["input_text_selector"],
        call["output_text_selector"],
    ) == (
        "/0/content",
        "/1/content",
        "/0/content",
    )
    assert call["reasoning_selectors"] == ["/1/content"]
    assert (call["model"], call["model_provider"]) == ("qwen", "ollama")
    assert call["tokens"] == {"input_tokens": 100, "output_tokens": 20}
    assert call["model_params"] == {"temperature": 0}
    assert not any(k.startswith("llm.input_messages") for k in call["attributes"])


def test_a_tool_call_reply_points_at_its_arguments():
    reply = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "x", "name": "Edits", "arguments": '{"a": 1}'}],
    }
    call = session([_article(), _llm("t1", "c1", "t1-root", reply)])["nodes"][1]
    assert call["outputs"][0]["tool_calls"] == [
        {"id": "x", "function": {"name": "Edits", "arguments": '{"a": 1}'}}
    ]
    assert call["output_text_selector"] == "/0/tool_calls/0/function/arguments"
    # a Claude reply can carry a preamble beside the tool call; the call is the result
    mixed = {**reply, "content": "Here are the edits."}
    call = session([_article(), _llm("t1", "c1", "t1-root", mixed)])["nodes"][1]
    assert call["outputs"][0]["content"] == "Here are the edits."
    assert call["output_text_selector"] == "/0/tool_calls/0/function/arguments"


def test_a_step_shows_its_labels_and_merge_decisions():
    decision = {
        "kind": "MERGE",
        "entity_type": "people",
        "name": "KSM",
        "detail": "sim=0.8",
    }
    merge = _row(
        "t1",
        "m1",
        "t1-root",
        "merge.people",
        events=[{"name": "MERGE KSM", "attributes": decision}],
    )
    check = _row(
        "t1",
        "k1",
        "m1",
        "match_check",
        {
            "hinbox.step": "merge.people/match_check",
            "hinbox.new_name": "KSM",
            "hinbox.is_match": True,
        },
    )
    s = session([_article(), merge, check])
    assert s["nodes"][1]["outputs"] == {"decisions": [decision]}
    assert (s["nodes"][2]["inputs"], s["nodes"][2]["outputs"]) == (
        {"new_name": "KSM"},
        {"is_match": True},
    )
    assert [n["parent_index"] for n in s["nodes"]] == [None, 0, 1]
    assert s["inputs"] == {"article.title": "Hearing"}
    assert s["metadata"] == {"run_id": "r1", "complete": True}


def test_a_step_that_recorded_its_arguments_shows_them_beside_its_labels():
    args = {"existing_profile": {"text": "Old."}, "new_article_text": "Article."}
    update = _row(
        "t1",
        "p1",
        "t1-root",
        "profile.update",
        {
            "hinbox.entity_name": "KSM",
            "hinbox.chars_after": 9,
            "input.value": json.dumps(args),
            "output.value": json.dumps({"profile": {"text": "Old. New."}}),
        },
    )
    s = session([_article(), update])
    node = s["nodes"][1]
    assert node["inputs"] == {**args, "labels": {"entity_name": "KSM"}}
    assert node["outputs"] == {
        "profile": {"text": "Old. New."},
        "labels": {"chars_after": 9},
    }
    assert "input.value" not in node["attributes"]
    # the article span's input.value is its title, not recorded arguments
    assert s["inputs"] == {"article.title": "Hearing"}


def _update(span: str, recorded: bool = True, output: Optional[Dict] = None) -> dict:
    attrs: Dict = {"hinbox.entity_name": "KSM", "hinbox.article_id": "a1"}
    if recorded:
        attrs["input.value"] = json.dumps({"existing_profile": {"text": "Old."}})
        attrs["output.value"] = json.dumps(output or {"profile": {"text": "Old. New."}})
    return _row("t1", span, "m1", "profile.update", attrs)


def test_each_recorded_step_becomes_a_session_of_its_own():
    reply = {"role": "assistant", "content": '{"edits": []}'}
    rows = [
        _article(),
        _row("t1", "m1", "t1-root", "merge.people"),
        _update("p1"),
        _llm("t1", "c1", "p1", reply),
        _update("p2", recorded=False),  # from before steps recorded arguments
        _row("t1", "k1", "m1", "match_check"),
    ]
    (s,) = step_sessions(rows, "profile.update")
    assert s["inputs"] == {"existing_profile": {"text": "Old."}}  # what a replay calls
    assert s["outputs"] == {"profile": {"text": "Old. New."}}
    assert (s["name"], s["status"], s["external_id"]) == (
        "profile.update: KSM",
        "completed",
        "hinbox:t1:p1",
    )
    assert [(n["name"], n["parent_index"]) for n in s["nodes"]] == [
        ("profile.update", None),
        ("chat qwen", 0),
    ]
    assert s["metadata"]["article_session"] == "hinbox:t1"
    assert (s["metadata"]["article_id"], s["metadata"]["article_title"]) == (
        "a1",
        "Hearing",
    )
    failed = _update("p3", output={"error": "profile edit failed: timeout"})
    (f,) = step_sessions([_article(), failed], "profile.update")
    assert (f["status"], f["error"]) == ("failed", "profile edit failed: timeout")


def test_step_parts_count_their_sessions(tmp_path):
    rows = [_article(), _update("p1"), _update("p2"), _update("p3", recorded=False)]
    counts: Dict = {}
    (part,) = write_parts(
        [_write(tmp_path / "run.jsonl", rows)],
        str(tmp_path / "out"),
        10**6,
        "profile.update",
        counts,
    )
    assert [s["external_id"] for s in _read(part)] == ["hinbox:t1:p1", "hinbox:t1:p2"]
    assert counts == {"sessions": 2}


def test_failures_show_on_the_node_and_an_incomplete_article_fails():
    check = _row(
        "t1", "k1", "t1-root", "match_check", error="ValueError: no is_match answer"
    )
    s = session([_article(outcome="incomplete: match check failed"), check])
    assert (s["nodes"][1]["status"], s["nodes"][1]["error"]) == (
        "failed",
        "ValueError: no is_match answer",
    )
    assert (s["status"], s["error"]) == ("failed", "incomplete: match check failed")
    assert session([_article(outcome="relevance_error")])["status"] == "failed"
    for done in ("processed", "not_relevant", "already_processed"):
        assert session([_article(outcome=done)])["status"] == "completed"


def test_a_trace_without_its_top_span_is_kept_as_unfinished():
    # the article span that held these was never written
    s = session(
        [
            _row("t9", "a", "gone", "relevance"),
            _row("t9", "b", "a", "sectioned relevance"),
        ]
    )
    assert (s["status"], s["error"], s["name"]) == (
        "failed",
        UNFINISHED,
        "unfinished (relevance)",
    )
    assert s["metadata"]["complete"] is False
    assert len(s["nodes"]) == 2


def test_parts_hold_whole_sessions_under_the_limit(tmp_path):
    # a run writes several articles' spans interleaved, over two files
    first = [
        _article("a"),
        _article("b"),
        _row("a", "a1", "a-root", attrs={"hinbox.x": "y" * 3000}),
    ]
    second = [_row("b", "b1", "b-root"), _article("c")]
    paths = [
        _write(tmp_path / "one.jsonl", first),
        _write(tmp_path / "two.jsonl", second),
    ]
    one = len(json.dumps(session([first[0], first[2]]))) + 1
    parts = write_parts(paths, str(tmp_path / "out"), one + 10)
    sessions = [[s["external_id"] for s in _read(p)] for p in parts]
    assert sessions == [["hinbox:a"], ["hinbox:b", "hinbox:c"]]
    assert [len(s["nodes"]) for p in parts for s in _read(p)] == [2, 2, 1]
    assert all(os.path.getsize(p) <= one + 10 for p in parts)
