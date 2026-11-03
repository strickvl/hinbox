"""Traces carry enough to rebuild training examples (`scripts/export_training_data.py`)."""

import json
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from scripts.export_training_data import examples, task_of
from src.config_loader import DomainConfig
from src.utils import tracing
from src.utils.provider_routing import ProviderTarget
from src.utils.step_timing import timed_step

_EXPORTER = InMemorySpanExporter()
_provider = trace.get_tracer_provider()
if not isinstance(_provider, TracerProvider):  # set once per process
    _provider = TracerProvider()
    trace.set_tracer_provider(_provider)
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture(autouse=True)
def spans():
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


def _client(replies):
    """A fake OpenAI client answering with ``replies`` in turn, traced."""
    queue = list(replies)

    def create(**kwargs):
        content, calls, finish = queue.pop(0)
        message = SimpleNamespace(
            content=content,
            tool_calls=[
                SimpleNamespace(function=SimpleNamespace(name=n, arguments=a))
                for n, a in calls
            ],
            model_extra={},
        )
        return SimpleNamespace(
            model="gpt-6-luna",
            choices=[SimpleNamespace(message=message, finish_reason=finish)],
            usage=None,
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    target = ProviderTarget(
        sdk="openai",
        api_model="gpt-6-luna",
        base_url=None,
        api_key=None,
        provider_label="openai",
    )
    return tracing.instrument_client(client, target)


def _rows(exporter, **kw):
    spans = [json.loads(s.to_json(indent=None)) for s in exporter.get_finished_spans()]
    return examples(spans, **kw)


def test_eval_match_check_becomes_labelled_chat_example(spans):
    client = _client([("reasoning... is_match: False", [], "stop")])
    messages = [
        {"role": "system", "content": "Decide if two records are the same."},
        {"role": "user", "content": "Camp Justice vs Guantanamo Bay"},
    ]
    with tracing.item_span("eval p1", eval_item_id="p1", eval_label=False):
        with timed_step("match_check"):
            tracing.annotate(new_name="Camp Justice", existing_name="Guantanamo Bay")
            client.chat.completions.create(model="m", messages=messages, temperature=0)
            tracing.annotate(is_match=False)
        tracing.annotate(eval_prediction=False, eval_correct=True)

    (row,) = _rows(spans)
    assert row["messages"][:2] == messages
    assert row["messages"][2] == {
        "role": "assistant",
        "content": "reasoning... is_match: False",
    }
    meta = row["metadata"]
    assert meta["task"] == "match_check"
    assert meta["new_name"] == "Camp Justice"
    assert meta["eval_label"] is False and meta["eval_correct"] is True
    assert meta["is_match"] is False
    assert meta["invocation_parameters"] == {"temperature": 0}
    assert meta["final_attempt"] is True and meta["attempt"] == 1


def test_tool_call_reply_and_schema_kept_and_retries_numbered(spans):
    client = _client(
        [
            (None, [("Items", '{"items": [{"name": "Spath"')], "length"),
            (None, [("Items", '{"items": [{"name": "Vance Spath"}]}')], "tool_calls"),
        ]
    )
    tool = {"type": "function", "function": {"name": "Items", "parameters": {}}}
    msgs = [{"role": "user", "content": "article text"}]
    with timed_step("extraction"):
        with timed_step("extract.people"):
            tracing.annotate(entity_type="people", article_id="a1")
            for _ in range(2):
                client.chat.completions.create(model="m", messages=msgs, tools=[tool])
            tracing.annotate(final_count=1, qc_flags=[])

    (row,) = _rows(spans)  # the truncated first try is dropped
    assert row["tools"] == [tool]
    call = row["messages"][-1]["tool_calls"][0]["function"]
    assert call == {"name": "Items", "arguments": {"items": [{"name": "Vance Spath"}]}}
    meta = row["metadata"]
    assert meta["task"] == "extract.people" and meta["article_id"] == "a1"
    assert meta["attempt"] == 2 and meta["final_attempt"] is True

    both = _rows(spans, keep_failed=True)
    assert [r["metadata"]["failed"] for r in both] == [True, False]


def test_every_prompt_file_a_step_loads_is_kept_with_its_hash(spans):
    cfg = DomainConfig("guantanamo")
    with timed_step("profile.create"):
        gen = cfg.prompt_path("profile_generation", "openai/x")
        ref = cfg.prompt_path("profile_reflection", "openai/x")
        cfg.prompt_path("profile_generation", "openai/x")  # loaded again
    (span,) = [s for s in spans.get_finished_spans() if s.name == "profile.create"]
    files = span.attributes["hinbox.prompt_files"]
    assert [f.split("@")[0] for f in files] == [
        gen.rsplit("/", 1)[-1],
        ref.rsplit("/", 1)[-1],
    ]
    assert all(len(f.split("@")[1]) == 16 for f in files)


def test_annotate_is_a_no_op_without_a_recording_span():
    tracing.annotate(entity_name="nobody")  # must not raise


def test_task_is_the_innermost_step():
    assert task_of("merge.people/profile.update") == "profile.update"
    assert task_of("") == "unknown"


def test_eval_edit_updater_calls_land_under_profile_update(spans, monkeypatch):
    from src.evals import profile_update as pu

    def fake_edit_profile(*args, **kwargs):
        client = _client([("[]", [], "stop")])
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}]
        )
        return {"text": "unchanged"}

    monkeypatch.setattr("src.engine.profile_edits.edit_profile", fake_edit_profile)
    case = {
        "entity_type": "people",
        "entity_name": "Vance Spath",
        "profile": {"text": "unchanged"},
        "article_text": "a",
        "article_id": "a9",
    }
    assert pu.make_edit_updater("openai/x", "guantanamo")(case) == "unchanged"
    (row,) = _rows(spans)
    assert row["metadata"]["task"] == "profile.update"
    assert row["metadata"]["entity_name"] == "Vance Spath"
    assert row["metadata"]["approach"] == "edit"


def test_instructor_retry_turns_keep_tool_call_names_and_ids(spans):
    """Instructor's reask appends an OpenAI-shaped assistant tool call + a tool reply."""
    client = _client([(None, [("Items", '{"items": []}')], "tool_calls")])
    retry = [
        {"role": "user", "content": "article text"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Items", "arguments": '{"items": "oops"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "Items",
            "content": "Validation Error found",
        },
    ]
    with timed_step("extract.people"):
        client.chat.completions.create(model="m", messages=retry)

    (span,) = [s for s in spans.get_finished_spans() if s.name.startswith("chat")]
    a = span.attributes
    base = "llm.input_messages.1.message.tool_calls.0.tool_call"
    assert a[f"{base}.id"] == "call_1"
    assert a[f"{base}.function.name"] == "Items"
    assert a["llm.input_messages.2.message.tool_call_id"] == "call_1"

    (row,) = _rows(spans)  # default: the original prompt + the good reply
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert row["metadata"]["retry_turns_dropped"] == 2

    (row,) = _rows(spans, keep_retry_turns=True)
    # exported as sent, but with arguments decoded like the reply's
    sent_call = row["messages"][1]["tool_calls"][0]
    assert sent_call["id"] == "call_1"
    assert sent_call["function"]["arguments"] == {"items": "oops"}
    reply_args = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(reply_args, dict)
    assert row["messages"][2] == retry[2] and row["messages"][0] == retry[0]


def test_anthropic_tool_declarations_exported_in_openai_form():
    from scripts.export_training_data import _openai_tool

    native = {"name": "Items", "description": "d", "input_schema": {"type": "object"}}
    assert _openai_tool(native) == {
        "type": "function",
        "function": {
            "name": "Items",
            "description": "d",
            "parameters": {"type": "object"},
        },
    }
    openai_form = {"type": "function", "function": {"name": "Items"}}
    assert _openai_tool(openai_form) == openai_form


@pytest.mark.parametrize(
    "second, kept", [([], 1), ([{"name": "Alice Smith", "type": "detainee"}], 2)]
)
def test_extraction_attempts_say_which_reply_was_kept(spans, second, kept):
    from unittest.mock import patch

    from src.engine.article_processor import ArticleProcessor

    processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
    client = _client([(None, [("Items", '{"items": []}')], "tool_calls")] * 2)
    replies = [[], second]

    def run(extractor, content, repair_hint=None):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "a"}]
        )
        return [dict(e) for e in replies.pop(0)]

    with patch.object(processor, "_run_extraction", side_effect=run):
        with timed_step("extract.people"):
            processor.extract_single_entity_type("people", "text", "a1")

    rows = _rows(spans)
    assert [r["metadata"]["extraction_attempt"] for r in rows] == [1, 2]
    assert [r["metadata"]["attempt_selected"] for r in rows] == [kept == 1, kept == 2]
    assert [r["metadata"]["task"] for r in rows] == ["extract.people"] * 2
    assert rows[1]["metadata"]["attempt_count"] == len(second)


def test_calls_under_a_request_that_gave_up_are_dropped(spans):
    """Instructor returns each reply fine, then raises after the last fails validation."""
    client = _client([(None, [("Items", '{"items": "bad"}')], "tool_calls")] * 2)
    msgs = [{"role": "user", "content": "a"}]
    with timed_step("profile.update"):
        try:
            with tracing.request_span("structured Items"):
                for _ in range(2):
                    client.chat.completions.create(model="m", messages=msgs)
                raise ValueError("validation failed twice")
        except ValueError:
            pass  # the caller falls back, so the step itself does not fail
        with tracing.request_span("structured Items"):
            client2 = _client([(None, [("Items", '{"items": []}')], "tool_calls")])
            client2.chat.completions.create(model="m", messages=msgs)

    (row,) = _rows(spans)
    assert row["metadata"]["request_ok"] is True
    assert row["messages"][-1]["tool_calls"][0]["function"]["arguments"] == {
        "items": []
    }
    kept = _rows(spans, keep_failed=True)
    assert [r["metadata"]["failed"] for r in kept] == [True, True, False]
    assert [r["metadata"]["attempt"] for r in kept] == [1, 2, 1]


def test_anthropic_retry_blocks_become_openai_tool_messages():
    from scripts.export_training_data import _decode_arguments, _from_anthropic

    retry = [
        {"role": "user", "content": "article"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Calling."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "Items",
                    "input": {"items": 1},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": "Validation Error",
                }
            ],
        },
    ]
    out = [_decode_arguments(m) for msg in retry for m in _from_anthropic(msg)]
    assert out == [
        {"role": "user", "content": "article"},
        {
            "role": "assistant",
            "content": "Calling.",
            "tool_calls": [
                {
                    "id": "tu_1",
                    "type": "function",
                    "function": {"name": "Items", "arguments": {"items": 1}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tu_1", "content": "Validation Error"},
    ]


def test_unreadable_sectioned_reply_is_exported_as_failed(spans, monkeypatch):
    from src.engine import relevance

    client = _client([("no verdict here", [], "stop")])

    def fake_text_generation(messages, **kwargs):
        resp = client.chat.completions.create(model="m", messages=messages)
        return resp.choices[0].message.content

    monkeypatch.setattr(relevance, "text_generation", fake_text_generation)
    monkeypatch.setattr(
        relevance,
        "extract_entities_cloud",
        lambda **kw: relevance.get_relevance_model("guantanamo")(
            is_relevant=True, reason="r"
        ),
    )
    relevance._check_relevance("article", "openai/x", "guantanamo", local=False)
    assert _rows(spans) == []  # the unreadable reply is not a training target
    (row,) = _rows(spans, keep_failed=True)
    assert row["metadata"]["failed"] is True and row["metadata"]["request_ok"] is False
