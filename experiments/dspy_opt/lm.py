"""Build DSPy language models from Hinbox model strings, without LiteLLM.

Model strings use the same prefixes as Hinbox (``openai/``, ``anthropic/``,
``gemini/``, ``ollama/``), and read the same environment variables:
``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``,
``OLLAMA_API_URL`` and ``HINBOX_OLLAMA_REASONING_EFFORT``.
"""

from typing import Any

import dspy

from src.constants import OLLAMA_API_URL, OLLAMA_REASONING_EFFORT


def make_lm(model: str, **kwargs: Any) -> dspy.LM:
    if model.startswith("ollama/"):
        kwargs.setdefault("api_base", OLLAMA_API_URL)
        # Same as the pipeline's match check. Without it Ollama samples at its
        # default temperature, and repeated runs give different scores.
        kwargs.setdefault("temperature", 0)
        if OLLAMA_REASONING_EFFORT:
            # Same default as the pipeline: thinking off unless configured
            kwargs.setdefault("reasoning_effort", OLLAMA_REASONING_EFFORT)
    # engine="lm15" pins DSPy's own provider clients: an unsupported setting
    # raises an error instead of falling back to LiteLLM (not installed here).
    return dspy.LM(model, engine="lm15", **kwargs)
