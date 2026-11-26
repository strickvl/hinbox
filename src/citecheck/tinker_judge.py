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


def parse_reply(text: str) -> List[Any]:
    """``ClaimVerdict`` objects from a raw reply in the pipeline's JSON
    format (``{"items": [...]}``); an item that doesn't fit is left out, as
    an Instructor reply would leave it out."""
    from src.utils.grounding import ClaimVerdict

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
            out.append(ClaimVerdict(**item))
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


def tinker_verdicts(
    messages: List[Dict[str, str]], model: str, max_tokens: int
) -> List[Any]:
    """One Tinker sample at temperature 0 for the judge's ``messages``."""
    import tinker

    from src.utils.tracing import llm_call_span

    sampler, renderer = _sampler(model)
    prompt = renderer.build_generation_prompt(messages, effort=EFFORT)
    params = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences()
    )
    with llm_call_span(
        "tinker", model, messages, temperature=0.0, max_tokens=max_tokens, effort=EFFORT
    ) as record:
        resp = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)
        resp = resp.result() if hasattr(resp, "result") else resp
        tokens = resp.sequences[0].tokens
        message, _ = renderer.parse_response(tokens)
        text = reply_text(message)
        record(text, prompt.length, len(tokens))
    return parse_reply(text)
