"""Tests that a run notices when the model server can't be reached at all."""

import httpx
import openai
import pytest

from src.utils import server_health
from src.utils.server_health import record, unreachable, watch_connections

REQUEST = httpx.Request("POST", "http://192.168.178.176:8080/v1/chat/completions")


def _no_route() -> openai.APIConnectionError:
    error = openai.APIConnectionError(request=REQUEST)
    error.__cause__ = httpx.ConnectError("[Errno 65] No route to host")
    return error


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setenv("HINBOX_MAX_CONNECTION_FAILURES", "3")
    server_health.reset()
    yield
    server_health.reset()


def test_failures_in_a_row_make_the_server_unreachable():
    for _ in range(2):
        record(_no_route())
    assert unreachable() is None
    record(_no_route())
    assert "3 model calls" in unreachable()
    assert "No route to host" in unreachable()


def test_an_answer_resets_the_count():
    record(_no_route())
    record(_no_route())
    record(None)
    record(_no_route())
    assert unreachable() is None


def test_timeouts_and_http_errors_are_not_connection_failures():
    for _ in range(5):
        record(openai.APITimeoutError(request=REQUEST))
    assert unreachable() is None
    record(_no_route())
    record(_no_route())
    # an HTTP error means the server answered, so the count starts again
    record(
        openai.InternalServerError(
            "boom", response=httpx.Response(500, request=REQUEST), body=None
        )
    )
    record(_no_route())
    assert unreachable() is None


def test_zero_turns_the_stop_off(monkeypatch):
    monkeypatch.setenv("HINBOX_MAX_CONNECTION_FAILURES", "0")
    for _ in range(50):
        record(_no_route())
    assert unreachable() is None


def test_wrapped_create_counts_each_request():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("fail"):
            raise _no_route()
        return "reply"

    wrapped = watch_connections(create)
    for _ in range(3):
        with pytest.raises(openai.APIConnectionError):
            wrapped(fail=True)
    assert unreachable() is not None
    assert wrapped(fail=False) == "reply"
    assert unreachable() is None
    assert len(calls) == 4


def test_sdk_clients_are_watched(monkeypatch):
    from src.utils import llm
    from src.utils.provider_routing import resolve_chat_target

    watched = []
    monkeypatch.setattr(llm, "watch_connections", lambda f: watched.append(f) or f)
    llm._new_raw_client(resolve_chat_target("ollama/qwen"))
    assert len(watched) == 1


def test_the_article_loop_stops_once_the_server_is_gone(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import MagicMock

    from src import process_and_extract as pae

    merged = []

    def merge(result, **kwargs):
        merged.append(result)
        for _ in range(3):  # every call this article made failed to connect
            record(_no_route())

    config = MagicMock()
    config.get_concurrency_config.return_value = {
        "extract_workers": 1,
        "extract_per_article": 1,
        "max_buffered_articles": 1,
    }
    config.get_cache_config.return_value = {}
    config.get_output_dir.return_value = str(tmp_path)
    monkeypatch.setattr(
        pae, "extract_single_article_only", lambda row, *a, **k: MagicMock(row=row)
    )
    monkeypatch.setattr(pae, "merge_and_finalize", merge)
    monkeypatch.setattr(pae, "article_scope", lambda span, result: nullcontext())
    monkeypatch.setattr(pae, "write_partial_report", lambda *a: None)
    rows = [{"id": f"a{i}"} for i in range(5)]
    with pytest.raises(server_health.ModelServerUnreachable):
        pae.process_articles_batch(
            rows,
            MagicMock(),
            {},
            MagicMock(limit=5),
            domain_config=config,
        )
    assert len(merged) == 1


def test_grounding_fails_fast_once_the_server_is_gone():
    """No call is tried; the checks are stored as errors and asked again later."""
    from tests.test_grounding_recheck import _run

    text = "First claim.^[a1] Second claim.^[a2]"
    articles = {"a1": "First claim.", "a2": "Second claim."}
    for _ in range(3):
        record(_no_route())
    first, calls = _run(text, articles)
    assert calls == 0
    assert [s.status for s in first.sources] == ["error", "error"]

    server_health.reset()  # the server is back
    second, calls = _run(text, articles, previous=first.model_dump())
    assert calls == 2
    assert [s.status for s in second.sources] == ["checked", "checked"]
