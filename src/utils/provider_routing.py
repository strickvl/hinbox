"""Provider routing: resolve model strings to SDK targets.

Replaces LiteLLM's implicit model-prefix dispatch with an explicit,
testable routing layer.  Every model string like ``"gemini/gemini-2.0-flash"``
is parsed into a ``ProviderTarget`` that tells the caller which SDK to use,
what base URL to hit, and which API key to send.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

from src.constants import OLLAMA_API_URL

# ---------------------------------------------------------------------------
# Well-known endpoint URLs
# ---------------------------------------------------------------------------
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JINA_OPENAI_BASE_URL = "https://api.jina.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ProviderTarget:
    """Resolved provider target for an LLM or embedding call.

    Parameters
    ----------
    sdk : "openai" or "anthropic"
        Which Python SDK to use for the API call.
    provider_label : str
        Human-readable provider name (``"gemini"``, ``"ollama"``, …).
    api_model : str
        Model identifier to send in the API request (prefix stripped).
    base_url : str or None
        Override base URL for the SDK client (``None`` = SDK default).
    api_key : str or None
        Override API key (``None`` = SDK reads its default env var).
    is_local : bool
        Whether this target runs locally (for privacy-mode enforcement).
    """

    sdk: Literal["openai", "anthropic"]
    provider_label: str
    api_model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    is_local: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_model_prefix(model: str) -> tuple:
    """Split ``"prefix/model-name"`` → ``("prefix", "model-name")``.

    Bare strings without a ``/`` are treated as ``("openai", model)``.
    """
    if "/" in model:
        prefix, _, rest = model.partition("/")
        return prefix.lower(), rest
    return "openai", model


def _require_env(var_name: str, provider_label: str) -> str:
    """Read an env var or raise a clear configuration error."""
    value = os.environ.get(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing environment variable {var_name} required for "
            f"{provider_label} provider.  Set it in your shell or .env file."
        )
    return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_chat_target(model: str) -> ProviderTarget:
    """Resolve a model string to a chat-completion provider target.

    Supported prefixes:

    - ``gemini/…``    → OpenAI SDK against Google's OpenAI-compatible endpoint
    - ``openai/…``    → OpenAI SDK (default endpoint)
    - ``anthropic/…`` → Anthropic SDK
    - ``ollama/…``    → OpenAI SDK against local Ollama
    - ``openrouter/…``→ OpenAI SDK against OpenRouter
    - bare string     → treated as ``openai/…``
    """
    prefix, api_model = _split_model_prefix(model)

    if prefix == "gemini":
        return ProviderTarget(
            sdk="openai",
            provider_label="gemini",
            api_model=api_model,
            base_url=GEMINI_OPENAI_BASE_URL,
            api_key=_require_env("GEMINI_API_KEY", "gemini"),
        )

    if prefix == "openai":
        return ProviderTarget(
            sdk="openai",
            provider_label="openai",
            api_model=api_model,
            # api_key=None → OpenAI SDK reads OPENAI_API_KEY automatically
        )

    if prefix == "anthropic":
        return ProviderTarget(
            sdk="anthropic",
            provider_label="anthropic",
            api_model=api_model,
            # api_key=None → Anthropic SDK reads ANTHROPIC_API_KEY automatically
        )

    if prefix == "ollama":
        return ProviderTarget(
            sdk="openai",
            provider_label="ollama",
            api_model=api_model,
            base_url=OLLAMA_API_URL,
            api_key="ollama",  # dummy key for local Ollama
            is_local=True,
        )

    if prefix == "openrouter":
        return ProviderTarget(
            sdk="openai",
            provider_label="openrouter",
            api_model=api_model,
            base_url=OPENROUTER_BASE_URL,
            api_key=_require_env("OPENROUTER_API_KEY", "openrouter"),
        )

    # Unknown prefix → treat as openai-compatible with the full string
    return ProviderTarget(
        sdk="openai",
        provider_label=prefix,
        api_model=api_model,
    )


def resolve_embedding_target(model: str) -> ProviderTarget:
    """Resolve a model string to an embedding provider target.

    Supported prefixes:

    - ``jina_ai/…``  → OpenAI SDK against Jina's OpenAI-compatible endpoint
    - ``openai/…``   → OpenAI SDK (default endpoint)
    - ``gemini/…``   → OpenAI SDK against Google's OpenAI-compatible endpoint
    """
    prefix, api_model = _split_model_prefix(model)

    if prefix == "jina_ai":
        return ProviderTarget(
            sdk="openai",
            provider_label="jina_ai",
            api_model=api_model,
            base_url=JINA_OPENAI_BASE_URL,
            api_key=_require_env("JINA_API_KEY", "jina_ai"),
        )

    if prefix == "openai":
        return ProviderTarget(
            sdk="openai",
            provider_label="openai",
            api_model=api_model,
        )

    if prefix == "gemini":
        return ProviderTarget(
            sdk="openai",
            provider_label="gemini",
            api_model=api_model,
            base_url=GEMINI_OPENAI_BASE_URL,
            api_key=_require_env("GEMINI_API_KEY", "gemini"),
        )

    # Fallback: treat as openai-compatible
    return ProviderTarget(
        sdk="openai",
        provider_label=prefix,
        api_model=api_model,
    )
