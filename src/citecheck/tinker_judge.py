"""The citation check answered by a model sampled on Tinker.

``judge_claims()`` hands its messages here when the judge's name starts with
``tinker``, so a model trained on Tinker can be the pipeline's citation
checker (``HINBOX_GROUNDING_MODEL``), a Kitaru replay's model override, or
one arm of ``just citecheck score --model``, with the same prompt and the
same reply format as every other judge. Two spellings:

- ``tinker://<run>/sampler_weights/<name>``: a checkpoint that training
  saved (``scripts/tinker_citecheck.py train`` lists them);
- ``tinker/<base model>``: the base model untuned, e.g.
  ``tinker/thinkingmachines/Inkling-2-Mini``, as the "before" arm.

Tinker is a cloud service, so this is never a ``--local`` judge; the tuned
model runs locally only once its weights are downloaded and served like any
other local model. The prompt is rendered with the Inkling renderer at
reasoning effort ``HINBOX_TINKER_EFFORT`` (default 0.0, as in training), and
each call is recorded as a model-call span (``llm_call_span()``) with its
token counts, so a Kitaru session shows it like any other model call.

Needs the ``tinker`` and ``tinker-cookbook`` packages and ``TINKER_API_KEY``;
they are not Hinbox dependencies (``experiments/tinker_zenml/`` has them).
"""

import json
import os
import re
import threading
from typing import Any, Dict, List, Tuple

BASE_MODEL = os.getenv("HINBOX_TINKER_BASE_MODEL", "thinkingmachines/Inkling-2-Mini")
EFFORT = float(os.getenv("HINBOX_TINKER_EFFORT", "0.0"))
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_lock = threading.Lock()
_clients: Dict[str, Any] = {}


def is_tinker_model(model: str) -> bool:
    return model.startswith("tinker://") or model.startswith("tinker/")


def parse_reply(text: str, with_passage: bool = False) -> List[Any]:
    """``ClaimVerdict`` objects (``ClaimVerdictWithPassage`` when the prompt
    asked for a copied passage) from a raw reply in the pipeline's JSON
    format (``{"items": [...]}``); an item that doesn't fit is left out, as
    an Instructor reply would leave it out."""
    from src.utils.grounding import ClaimVerdict, ClaimVerdictWithPassage

    cls = ClaimVerdictWithPassage if with_passage else ClaimVerdict

    body = _FENCE.sub("", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    out = []
    for item in items or []:
        try:
            out.append(cls(**item))
        except Exception:  # wrong level or missing field: no verdict
            continue
    return out


def _sampler(model: str) -> Tuple[Any, Any]:
    """(sampling client, renderer) for ``model``, made once per process."""
    with _lock:
        if model not in _clients:
            import tinker
            from tinker_cookbook import model_info, renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            service = tinker.ServiceClient()
            if model.startswith("tinker://"):
                sampler = service.create_sampling_client(model_path=model)
                base = BASE_MODEL
            else:
                base = model.split("/", 1)[1]
                sampler = service.create_sampling_client(base_model=base)
            sampler = sampler.result() if hasattr(sampler, "result") else sampler
            tokenizer_name = base.split(":", 1)[0]
            renderer = renderers.get_renderer(
                model_info.get_recommended_renderer_name(tokenizer_name),
                get_tokenizer(tokenizer_name),
            )
            _clients[model] = (sampler, renderer)
        return _clients[model]


def reply_text(message: Dict[str, Any]) -> str:
    """The answer text of a parsed message (thinking parts dropped)."""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def sampling_settings(max_tokens: int) -> Tuple[float, Dict[str, Any]]:
    """(reasoning effort, Tinker sampling parameters) for one call: temperature
    0, ``max_tokens`` and ``HINBOX_TINKER_EFFORT``, with a Kitaru replay's
    ``model_params`` (``set_model_params()``) on top. ``reasoning_effort``
    must be a number in [0, 1) (the Inkling renderer's effort) or ``"none"``
    (0.0); anything Tinker can't honour fails the call rather than running
    another configuration than the one asked for."""
    from src.utils.llm import model_params

    over = model_params()
    effort = EFFORT
    if "reasoning_effort" in over:
        raw = over.pop("reasoning_effort")
        try:
            effort = 0.0 if raw == "none" else float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Tinker judges take reasoning_effort as a number in [0, 1), not {raw!r}"
            ) from None
        if not 0.0 <= effort < 1.0:
            raise ValueError(f"reasoning_effort {effort} is outside [0, 1)")
    params: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
    for key in ("temperature", "max_tokens", "top_p", "seed"):
        if key in over:
            params[key] = over.pop(key)
    if over:  # a key added to MODEL_PARAM_KEYS that Tinker has no field for
        raise ValueError(f"Tinker judges can't honour model_params {sorted(over)}")
    return effort, params


def tinker_verdicts(
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    with_passage: bool = False,
) -> List[Any]:
    """One Tinker sample for the judge's ``messages``, at temperature 0
    unless a replay's ``model_params`` say otherwise (``sampling_settings()``)."""
    import tinker

    from src.utils.tracing import llm_call_span

    effort, settings = sampling_settings(max_tokens)
    sampler, renderer = _sampler(model)
    prompt = renderer.build_generation_prompt(messages, effort=effort)
    params = tinker.SamplingParams(**settings, stop=renderer.get_stop_sequences())
    with llm_call_span("tinker", model, messages, effort=effort, **settings) as record:
        resp = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)
        resp = resp.result() if hasattr(resp, "result") else resp
        tokens = resp.sequences[0].tokens
        message, _ = renderer.parse_response(tokens)
        text = reply_text(message)
        record(text, prompt.length, len(tokens))
    return parse_reply(text, with_passage)
