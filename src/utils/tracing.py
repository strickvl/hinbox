"""OpenTelemetry traces of every pipeline step and model call.

What gets recorded, and where it goes:

- ``configure_tracing()`` runs once at startup. It always writes every span as
  one JSON line to ``{output_dir}/traces/run-<timestamp>.jsonl``, so we keep
  our own copy whatever viewer we use. If ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
  (or ``OTEL_EXPORTER_OTLP_ENDPOINT``) is set, spans are also sent there over
  OTLP/HTTP, e.g. to Phoenix (``just phoenix``) at
  ``http://localhost:6006/v1/traces``. ``HINBOX_TRACING=off`` turns it all off.
- ``timed_step()`` in ``step_timing.py`` opens a span per step, so a trace
  nests as article -> ``merge.people`` -> ``profile.update`` -> model call.
  The thread pools already pass ``contextvars.copy_context()`` to workers, and
  OpenTelemetry keeps the current span in a ``ContextVar``, so spans started
  in worker threads land under the right parent.
- ``instrument_client()`` replaces the ``create`` method on each SDK client
  (before Instructor wraps it), so every request, including Instructor's own
  retries and our fallback paths, becomes one span with the messages sent,
  the reply (text, tool-call arguments, reasoning if the model returned it),
  token counts and the finish reason.

Attribute names follow OpenInference (what Phoenix reads) plus the main
OpenTelemetry ``gen_ai.*`` ones. Until ``configure_tracing()`` runs (tests,
library use) the tracer is OpenTelemetry's no-op one and nothing is recorded.
"""

import contextvars
import functools
import ipaddress
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from opentelemetry import context as otel_context
from opentelemetry import trace

logger = logging.getLogger("hinbox.tracing")

TRACER_NAME = "hinbox"
_provider: Any = None
_trace_path: Optional[str] = None


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def _endpoint() -> Optional[str]:
    return os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )


def is_local_endpoint(url: str) -> bool:
    """True for localhost, private/LAN/Tailscale IPs, ``*.local`` and bare LAN names."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # a bare LAN name such as "beast"
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip in _TAILSCALE


_TAILSCALE = ipaddress.ip_network("100.64.0.0/10")


def configure_tracing(
    output_dir: str, *, local: bool, run_name: str = "run"
) -> Optional[str]:
    """Start recording spans; returns the JSONL path, or None if turned off.

    Raises ``ValueError`` for a ``--local`` run whose OTLP endpoint is not on
    this machine or the LAN, so private articles never leave the network.
    """
    global _provider, _trace_path
    if os.getenv("HINBOX_TRACING", "on").lower() in {"0", "off", "false", "no"}:
        return None
    endpoint = _endpoint()
    if local and endpoint and not is_local_endpoint(endpoint):
        raise ValueError(
            f"--local run, but traces would go to {endpoint}, which is not on "
            "this machine or the LAN. Unset OTEL_EXPORTER_OTLP_*ENDPOINT or "
            "point it at a local Phoenix."
        )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    run_id = f"{run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    _trace_path = os.path.join(traces_dir, f"{run_id}.jsonl")

    class _SessionProcessor(SpanProcessor):
        """Tag every span with the run, so Phoenix groups a run's traces."""

        def on_start(self, span: Any, parent_context: Any = None) -> None:
            span.set_attribute("session.id", run_id)

    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "hinbox", "openinference.project.name": "hinbox"}
        )
    )
    provider.add_span_processor(_SessionProcessor())
    provider.add_span_processor(
        BatchSpanProcessor(JsonlSpanExporter(_trace_path), max_queue_size=65536)
    )
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(), max_queue_size=65536)
        )
    trace.set_tracer_provider(provider)
    _provider = provider
    import atexit

    atexit.register(shutdown_tracing)
    logger.info("Tracing to %s%s", _trace_path, f" and {endpoint}" if endpoint else "")
    return _trace_path


def shutdown_tracing() -> None:
    """Flush and close the exporters (also runs at interpreter exit)."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


class JsonlSpanExporter:
    """Append each finished span to a file as one JSON line."""

    def __init__(self, path: str):
        self._file = open(path, "a", encoding="utf-8")

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        for span in spans:
            self._file.write(span.to_json(indent=None) + "\n")
        self._file.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._file.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._file.flush()
        return True


# ---------------------------------------------------------------------------
# Steps and articles
# ---------------------------------------------------------------------------


@contextmanager
def step_span(name: str, path: str) -> Iterator[None]:
    with tracer().start_as_current_span(
        name, attributes={"openinference.span.kind": "CHAIN", "hinbox.step": path}
    ):
        yield


def start_article_span(row_index: int, article: Dict[str, Any]) -> Any:
    """A root span (own trace) for one article; extraction and merge go under it.

    Extraction runs in a worker thread and merging on the main thread later, so
    the span is started here and entered in both places (``run_in_span``,
    ``article_scope``); it ends after the merge.
    """
    title = str(article.get("title") or "")
    return tracer().start_span(
        f"article #{row_index}",
        context=otel_context.Context(),
        attributes={
            "openinference.span.kind": "CHAIN",
            "hinbox.article.id": str(article.get("id") or ""),
            "hinbox.article.title": title,
            "hinbox.article.url": str(article.get("url") or ""),
            "hinbox.article.published_date": str(article.get("published_date") or ""),
            "input.value": title,
        },
    )


def run_in_span(span: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with ``span`` as the current span (for a worker thread)."""
    with trace.use_span(span, end_on_exit=False):
        return fn(*args, **kwargs)


@contextmanager
def article_scope(span: Any, result: Any) -> Iterator[None]:
    """Enter the article's span for merging, note how it ended, then end it."""
    with trace.use_span(span, end_on_exit=True):
        try:
            yield
        finally:
            meta = getattr(result, "processing_metadata", {}) or {}
            span.set_attribute("hinbox.article.extracted", bool(result.was_extracted))
            outcome = result.skip_reason or meta.get("reason") or "processed"
            span.set_attribute("hinbox.article.outcome", str(outcome))
            span.set_attribute("output.value", str(outcome))


def record_decision(kind: str, entity_type: str, name: str, detail: str) -> None:
    """Add a merge decision (NEW/MERGE/SKIP/...) as an event on the current span."""
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(
            f"{kind} {name}",
            {"kind": kind, "entity_type": entity_type, "name": name, "detail": detail},
        )


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _message_attrs(prefix: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for i, msg in enumerate(messages):
        attrs[f"{prefix}.{i}.message.role"] = str(msg.get("role", ""))
        content = msg.get("content")
        if content is not None:
            attrs[f"{prefix}.{i}.message.content"] = _text(content)
        for j, call in enumerate(msg.get("tool_calls") or []):
            base = f"{prefix}.{i}.message.tool_calls.{j}.tool_call.function"
            attrs[f"{base}.name"] = str(call.get("name", ""))
            attrs[f"{base}.arguments"] = _text(call.get("arguments", ""))
    return attrs


def _request_attrs(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    messages = list(kwargs.get("messages") or [])
    if kwargs.get("system"):  # Anthropic passes the system prompt separately
        messages = [{"role": "system", "content": kwargs["system"]}] + messages
    params = {
        k: v
        for k, v in kwargs.items()
        if k not in {"messages", "system", "tools", "model"}
    }
    attrs: Dict[str, Any] = {
        "openinference.span.kind": "LLM",
        "llm.provider": target.provider_label,
        "llm.model_name": target.api_model,
        "llm.invocation_parameters": _text(params),
        "gen_ai.system": target.provider_label,
        "gen_ai.request.model": target.api_model,
        "input.value": _text({"messages": messages}),
        "input.mime_type": "application/json",
        **_message_attrs("llm.input_messages", messages),
    }
    for i, tool in enumerate(kwargs.get("tools") or []):
        attrs[f"llm.tools.{i}.tool.json_schema"] = _text(tool)
    return attrs


def _openai_reply(response: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(output message, extra attrs) from an OpenAI-style ChatCompletion."""
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    extras: Dict[str, Any] = {"gen_ai.response.model": str(response.model or "")}
    if choice is not None and choice.finish_reason:
        extras["gen_ai.response.finish_reasons"] = [str(choice.finish_reason)]
    usage = getattr(response, "usage", None)
    if usage is not None:
        extras["llm.token_count.prompt"] = usage.prompt_tokens or 0
        extras["llm.token_count.completion"] = usage.completion_tokens or 0
        extras["llm.token_count.total"] = usage.total_tokens or 0
    out: Dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}
    if message is not None:
        out["content"] = message.content
        for call in message.tool_calls or []:
            fn = getattr(call, "function", None)
            out["tool_calls"].append(
                {
                    "name": getattr(fn, "name", ""),
                    "arguments": getattr(fn, "arguments", ""),
                }
            )
        extra = getattr(message, "model_extra", None) or {}
        reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        if reasoning:
            extras["hinbox.llm.reasoning"] = _text(reasoning)
    return out, extras


def _anthropic_reply(response: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(output message, extra attrs) from an Anthropic Message."""
    extras: Dict[str, Any] = {"gen_ai.response.model": str(response.model or "")}
    if response.stop_reason:
        extras["gen_ai.response.finish_reasons"] = [str(response.stop_reason)]
    usage = getattr(response, "usage", None)
    if usage is not None:
        extras["llm.token_count.prompt"] = usage.input_tokens or 0
        extras["llm.token_count.completion"] = usage.output_tokens or 0
        extras["llm.token_count.total"] = (usage.input_tokens or 0) + (
            usage.output_tokens or 0
        )
    texts, calls, thinking = [], [], []
    for block in response.content or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            texts.append(block.text)
        elif kind == "tool_use":
            calls.append({"name": block.name, "arguments": block.input})
        elif kind == "thinking":
            thinking.append(getattr(block, "thinking", ""))
    if thinking:
        extras["hinbox.llm.reasoning"] = "\n".join(thinking)
    out = {"role": "assistant", "content": "".join(texts) or None, "tool_calls": calls}
    return out, extras


def _record_reply(span: Any, response: Any, anthropic: bool) -> None:
    out, extras = (_anthropic_reply if anthropic else _openai_reply)(response)
    span.set_attributes(extras)
    span.set_attributes(_message_attrs("llm.output_messages", [out]))
    span.set_attribute(
        "output.value",
        out["content"] if out["content"] and not out["tool_calls"] else _text(out),
    )


def _traced_create(create: Callable[..., Any], target: Any, anthropic: bool) -> Any:
    @functools.wraps(create)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracer().start_as_current_span(f"chat {target.api_model}") as span:
            if span.is_recording():
                span.set_attributes(_request_attrs(target, kwargs))
            response = create(*args, **kwargs)
            if span.is_recording():
                try:
                    _record_reply(span, response, anthropic)
                except Exception as e:  # never let tracing break a model call
                    logger.debug("Could not record model reply: %s", e)
            return response

    return wrapper


def instrument_client(client: Any, target: Any) -> Any:
    """Trace every chat request made through this SDK client; returns the client.

    Call before ``instructor.from_openai``/``from_anthropic``, which keep a
    reference to ``create`` when they wrap the client.
    """
    if target.sdk == "anthropic":
        client.messages.create = _traced_create(client.messages.create, target, True)
    else:
        completions = client.chat.completions
        completions.create = _traced_create(completions.create, target, False)
    return client


def instrument_embeddings(client: Any, target: Any) -> Any:
    """Trace each embeddings request (model, inputs, token count; not vectors)."""
    create = client.embeddings.create

    @functools.wraps(create)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with tracer().start_as_current_span(f"embeddings {target.api_model}") as span:
            inputs = kwargs.get("input") or []
            if span.is_recording():
                span.set_attributes(
                    {
                        "openinference.span.kind": "EMBEDDING",
                        "embedding.model_name": target.api_model,
                        "hinbox.embedding.count": len(inputs),
                        "input.value": _text(inputs),
                    }
                )
            response = create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if span.is_recording() and usage is not None:
                span.set_attribute("llm.token_count.prompt", usage.prompt_tokens or 0)
            return response

    client.embeddings.create = wrapper
    return client


def in_current_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` to run in the caller's context (for ``run_in_executor``)."""
    ctx = contextvars.copy_context()
    return lambda *a, **kw: ctx.run(fn, *a, **kw)
