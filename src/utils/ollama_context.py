"""Stop prompts that are too long for an Ollama model's context window.

Ollama doesn't refuse a prompt that is longer than the model's context
(``num_ctx``). It quietly drops the start of the prompt, which is where the
instructions are, and the model answers without them. A 40k-character profile
plus an article is about 25k tokens, so on a model set to 16k the update runs
without its instructions and usually fails with a confusing ``max_tokens``
error on every retry.

``check_prompt_fits()`` estimates the prompt's size before the call and raises
``PromptTooLongError`` when it can't fit, so the failure says what happened.
"""

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import requests

from src.logging_config import get_logger
from src.utils.provider_routing import ProviderTarget

logger = get_logger("utils.ollama_context")

# English prose runs at about 4 characters per token; dividing by 4 slightly
# underestimates, so a prompt is only refused when it clearly doesn't fit.
CHARS_PER_TOKEN = 4


class PromptTooLongError(RuntimeError):
    """The prompt is longer than the local model's context window."""


def estimate_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return chars // CHARS_PER_TOKEN


@lru_cache(maxsize=32)
def _fetch_num_ctx(base_url: str, model: str) -> Optional[int]:
    """Read ``num_ctx`` from Ollama's ``/api/show``; None if it isn't set."""
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    try:
        reply = requests.post(f"{root}/api/show", json={"model": model}, timeout=10)
        reply.raise_for_status()
        params = reply.json().get("parameters") or ""
    except Exception as e:  # the check is a safety net, never a new failure
        logger.debug("Could not read num_ctx for %s: %s", model, e)
        return None
    match = re.search(r"^num_ctx\s+(\d+)", params, re.MULTILINE)
    return int(match.group(1)) if match else None


def context_length(target: ProviderTarget) -> Optional[int]:
    """The model's context window in tokens, or None when unknown.

    ``HINBOX_OLLAMA_NUM_CTX`` overrides it; otherwise it is the ``num_ctx``
    the model was created with (``PARAMETER num_ctx`` in its Modelfile).
    """
    override = os.getenv("HINBOX_OLLAMA_NUM_CTX", "").strip()
    if override:
        return int(override)
    if target.provider_label != "ollama" or not target.base_url:
        return None
    return _fetch_num_ctx(target.base_url, target.api_model)


def check_prompt_fits(target: ProviderTarget, messages: List[Dict[str, Any]]) -> None:
    """Raise ``PromptTooLongError`` if the prompt can't fit the local model."""
    if target.provider_label != "ollama":
        return
    limit = context_length(target)
    if not limit:
        return
    tokens = estimate_prompt_tokens(messages)
    if tokens > limit:
        raise PromptTooLongError(
            f"Prompt is about {tokens} tokens but {target.api_model} has a "
            f"{limit}-token context; Ollama would cut off its start. Use a model "
            "with a larger num_ctx."
        )
