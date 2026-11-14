"""Tests for OpenTelemetry tracing of steps, articles and model calls."""

import contextvars
import json
import threading
from types import SimpleNamespace

import httpx
import instructor
import pytest
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel

from src.logging_config import DecisionKind, log_decision
from src.utils import tracing
from src.utils.provider_routing import ProviderTarget
from src.utils.step_timing import timed_step

_EXPORTER = InMemorySpanExporter()


def _install_provider() -> None:
    # The global provider can be set only once per process
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))


_install_provider()


@pytest.fixture(autouse=True)
def spans():
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


def _target(sdk: str = "openai", provider: str = "ollama") -> ProviderTarget:
    return ProviderTarget(
        sdk=sdk,
        api_model="qwen3.6:35b",
        base_url="http://beast:11434/v1",
        api_key=None,
        provider_label=provider,
    )


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


# --- where traces may go ----------------------------------------------------


@pytest.mark.parametrize(
    "url, ok",
    [
        ("http://localhost:6006/v1/traces", True),
        ("http://127.0.0.1:6006/v1/traces", True),
        ("http://[::1]:6006/v1/traces", True),
        ("http://192.168.178.176:6006/v1/traces", True),
        ("http://beast:6006/v1/traces", True),
        ("http://beast.local:6006/v1/traces", True),
        ("http://100.101.102.103:6006/v1/traces", True),  # Tailscale
        ("https://cloud.langfuse.com/api/public/otel", False),
        ("https://8.8.8.8/v1/traces", False),
        ("not a url", False),
    ],
)
def test_is_local_endpoint(url, ok):
    assert tracing.is_local_endpoint(url) is ok


def test_local_run_refuses_cloud_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://example.com/x")
    with pytest.raises(ValueError, match="not on this machine"):
        tracing.configure_tracing(str(tmp_path), local=True)
    assert not (tmp_path / "traces").exists()


def test_tracing_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setenv("HINBOX_TRACING", "off")
    assert tracing.configure_tracing(str(tmp_path), local=False) is None
    assert not (tmp_path / "traces").exists()


def test_jsonl_exporter_writes_one_line_per_span(spans, tmp_path):
    with tracing.tracer().start_as_current_span("hello"):
        pass
    path = tmp_path / "run.jsonl"
    exporter = tracing.JsonlSpanExporter(str(path))
    exporter.export(spans.get_finished_spans())
    exporter.shutdown()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["name"] for r in rows] == ["hello"]


# --- model calls --------------------------------------------------------------


def _fake_openai(response):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), calls


def test_openai_call_records_prompt_reply_tokens_and_reasoning(spans):
    message = SimpleNamespace(
        content="",
        tool_calls=None,
        model_extra={"reasoning": "thinking about Gitmo..."},
    )
    response = SimpleNamespace(
        model="qwen3.6:35b",
        choices=[SimpleNamespace(message=message, finish_reason="length")],
        usage=SimpleNamespace(
            prompt_tokens=900, completion_tokens=16384, total_tokens=17284
        ),
    )
    client, calls = _fake_openai(response)
    tracing.instrument_client(client, _target())
    messages = [
        {"role": "system", "content": "You write profiles."},
        {"role": "user", "content": "Article about Camp Justice"},
    ]
    assert (
        client.chat.completions.create(model="m", messages=messages, max_tokens=5)
        is response
    )
    assert calls[0]["messages"] == messages  # request passed through unchanged

    (span,) = _by_name(spans, "chat qwen3.6:35b")
    a = span.attributes
    assert a["openinference.span.kind"] == "LLM"
    assert a["llm.provider"] == "ollama"
    assert a["llm.input_messages.0.message.role"] == "system"
    assert a["llm.input_messages.1.message.content"] == "Article about Camp Justice"
    assert json.loads(a["llm.invocation_parameters"]) == {"max_tokens": 5}
    assert a["gen_ai.response.finish_reasons"] == ("length",)
    assert a["llm.token_count.completion"] == 16384
    assert a["hinbox.llm.reasoning"] == "thinking about Gitmo..."


def test_failed_call_is_recorded_and_still_raises(spans):
    def create(**kwargs):
        raise RuntimeError("503 overloaded")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    tracing.instrument_client(client, _target())
    with pytest.raises(RuntimeError, match="503"):
        client.chat.completions.create(model="m", messages=[])
    (span,) = _by_name(spans, "chat qwen3.6:35b")
    assert not span.status.is_ok
    assert span.events[0].name == "exception"


def test_anthropic_call_records_system_prompt_and_tool_use(spans):
    response = SimpleNamespace(
        model="claude-haiku-5-5",
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=4),
        content=[
            SimpleNamespace(type="tool_use", name="Person", input={"name": "Spath"})
        ],
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))
    target = _target(sdk="anthropic", provider="anthropic")
    tracing.instrument_client(client, target)
    client.messages.create(
        model="m", system="Be careful.", messages=[{"role": "user", "content": "hi"}]
    )
    (span,) = _by_name(spans, "chat qwen3.6:35b")
    a = span.attributes
    assert a["llm.input_messages.0.message.content"] == "Be careful."
    assert a["llm.input_messages.1.message.role"] == "user"
    base = "llm.output_messages.0.message.tool_calls.0.tool_call.function"
    assert a[f"{base}.name"] == "Person"
    assert json.loads(a[f"{base}.arguments"]) == {"name": "Spath"}
    assert a["llm.token_count.total"] == 14


class Person(BaseModel):
    name: str


def test_instructor_calls_go_through_the_traced_create(spans):
    """Instructor keeps a reference to ``create``; it must be our wrapper."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-6-luna",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "Person",
                                    "arguments": '{"name": "Vance Spath"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        return httpx.Response(200, json=body)

    raw = OpenAI(
        api_key="test",
        base_url="http://test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    target = _target(provider="openai")
    tracing.instrument_client(raw, target)
    client = instructor.from_openai(raw, mode=instructor.Mode.TOOLS)
    person = client.chat.completions.create(
        model="gpt-6-luna",
        response_model=Person,
        messages=[{"role": "user", "content": "Who?"}],
        max_retries=0,
    )
    assert person.name == "Vance Spath"
    (span,) = _by_name(spans, "chat qwen3.6:35b")
    a = span.attributes
    assert "Person" in a["llm.tools.0.tool.json_schema"]
    base = "llm.output_messages.0.message.tool_calls.0.tool_call.function"
    assert json.loads(a[f"{base}.arguments"]) == {"name": "Vance Spath"}


# --- nesting across threads -----------------------------------------------------


def test_article_trace_holds_worker_extraction_and_main_thread_merge(spans):
    article = {"id": "a1", "title": "Camp Justice hearing", "url": "u"}
    span = tracing.start_article_span(1, article)

    def extract():
        with timed_step("extraction"):
            # article_processor runs each type in a pool with copy_context()
            inner = []
            ctx = contextvars.copy_context()
            t = threading.Thread(
                target=ctx.run, args=(lambda: inner.append(_step("extract.people")),)
            )
            t.start()
            t.join()
        return SimpleNamespace(
            was_extracted=True, skip_reason=None, processing_metadata={}
        )

    def _step(name):
        with timed_step(name):
            pass

    result_box = []
    worker = threading.Thread(
        target=lambda: result_box.append(tracing.run_in_span(span, extract))
    )
    worker.start()
    worker.join()
    with tracing.article_scope(span, result_box[0]):
        with timed_step("merge.people"):
            log_decision(DecisionKind.MERGE, "people", "Vance Spath", "similarity=0.91")

    finished = {s.name: s for s in spans.get_finished_spans()}
    root = finished["article #1"]
    assert root.parent is None
    assert root.attributes["hinbox.article.title"] == "Camp Justice hearing"
    assert root.attributes["hinbox.article.outcome"] == "processed"
    assert {s.context.trace_id for s in finished.values()} == {root.context.trace_id}
    assert finished["extraction"].parent.span_id == root.context.span_id
    people = finished["extract.people"]
    assert people.attributes["hinbox.step"] == "extraction/extract.people"
    assert people.parent.span_id == finished["extraction"].context.span_id
    merge = finished["merge.people"]
    assert merge.parent.span_id == root.context.span_id
    (event,) = merge.events
    assert event.attributes["kind"] == "MERGE"
    assert event.attributes["name"] == "Vance Spath"


def test_article_outcome_shows_incomplete_reason(spans):
    span = tracing.start_article_span(2, {"id": "a2", "title": "t"})
    result = SimpleNamespace(
        was_extracted=True,
        skip_reason=None,
        processing_metadata={"reason": "incomplete: error_people"},
    )
    with tracing.article_scope(span, result):
        pass
    (root,) = _by_name(spans, "article #2")
    assert root.attributes["hinbox.article.outcome"] == "incomplete: error_people"


# --- step arguments and results -------------------------------------------------


def test_long_values_are_kept_whole_whatever_the_environment_says(monkeypatch):
    monkeypatch.setenv("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "100")
    monkeypatch.setenv("OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT", "100")
    exporter = InMemorySpanExporter()
    provider = TracerProvider(span_limits=tracing.span_limits())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("t").start_as_current_span("profile.update") as span:
        span.set_attribute("input.value", "x" * 60_000)
    (finished,) = exporter.get_finished_spans()
    assert len(finished.attributes["input.value"]) == 60_000


def test_profile_update_records_its_arguments_and_result(spans, monkeypatch):
    from src.engine import profile_edits
    from src.engine.profiles import VersionedProfile, update_profile

    old_text = "## Background\n" + "Spath was a judge.^[a1] " * 700  # 16,800 chars
    article = "Vance Spath ruled on Tuesday. " * 400
    edited = {
        "text": old_text + "He ruled.^[a2]",
        "sources": ["a1", "a2"],
        "edits": [{"action": "add", "text": "He ruled.^[a2]"}],
        "disagreements": [],
        "edit_report": {"applied": 1, "appended": 0, "skipped": []},
    }
    monkeypatch.setattr(profile_edits, "edit_profile", lambda *a, **k: edited)
    profile = {"text": old_text, "sources": ["a1"], "tags": ["judge"]}
    update_profile(
        "people", "Vance Spath", profile, VersionedProfile(), article, "a2",
        model="ollama/qwen", article_dates={"a2": "2018-05-01"}, kind="judge",
    )  # fmt: skip

    (span,) = _by_name(spans, "profile.update")
    args = json.loads(span.attributes["input.value"])
    assert args["existing_profile"]["text"] == old_text
    assert args["new_article_text"] == article
    assert (args["new_article_id"], args["kind"]) == ("a2", "judge")
    assert args["article_dates"] == {"a2": "2018-05-01"}
    assert args["answering_model"] == "ollama/qwen"
    result = json.loads(span.attributes["output.value"])
    assert result["profile"]["text"] == edited["text"]
    assert result["edit_report"]["applied"] == 1
    assert result["edits"] == edited["edits"]
    assert span.attributes["hinbox.entity_name"] == "Vance Spath"  # labels stay


def test_match_check_records_both_profiles_and_its_verdict(spans, monkeypatch):
    from src.engine import match_checker

    monkeypatch.setattr(
        match_checker,
        "text_generation",
        lambda *a, **k: (
            "[[ ## reasoning ## ]]\nSame judge.\n\n"
            "[[ ## is_match ## ]]\nTrue\n\n[[ ## confidence ## ]]\n0.9"
        ),
    )
    match_checker._check_match(
        "local", "Spath", "Vance Spath", "new text", "old text", "people", "m"
    )
    (span,) = _by_name(spans, "match_check")
    args = json.loads(span.attributes["input.value"])
    assert (args["new_profile_text"], args["existing_profile_text"]) == (
        "new text",
        "old text",
    )
    result = json.loads(span.attributes["output.value"])["result"]
    assert (result["is_match"], result["confidence"]) == (True, 0.9)
