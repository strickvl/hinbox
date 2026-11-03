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


def test_prompt_file_and_hash_noted_on_the_step(spans):
    with timed_step("profile.update"):
        path = DomainConfig("guantanamo").prompt_path("profile_edit", "openai/x")
    (span,) = [s for s in spans.get_finished_spans() if s.name == "profile.update"]
    assert span.attributes["hinbox.prompt_file"] == path.rsplit("/", 1)[-1]
    assert len(span.attributes["hinbox.prompt_sha"]) == 16


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
