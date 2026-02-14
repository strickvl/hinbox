"""Generic entity extraction utilities for cloud and local LLM models.

This module provides generic extraction functions that work with both cloud-based
and local language models. These functions form the core of the entity extraction
pipeline and are used by specific entity extractors for people, organizations,
locations, and events.

Extraction calls are routed through ``cloud_generation``/``local_generation``
in ``src.utils.llm`` so that all retry, rate-limit, and "multiple tool calls"
recovery strategies apply uniformly.

When the extraction sidecar cache is configured (via
``configure_extraction_sidecar_cache``), results for deterministic calls
(temperature=0) are persisted to disk and returned on subsequent runs
without an LLM call.
"""

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from src.constants import (
    CLOUD_MODEL,
    OLLAMA_MODEL,
)
from src.logging_config import get_logger
from src.utils.cache_utils import sha256_text
from src.utils.extraction_cache import (
    ExtractionSidecarCache,
    build_cache_record,
)
from src.utils.llm import (
    cloud_generation,
    create_messages,
    local_generation,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Extraction sidecar cache (configured once at pipeline startup)
# ---------------------------------------------------------------------------

_EXTRACTION_CACHE: Optional[ExtractionSidecarCache] = None


def configure_extraction_sidecar_cache(
    *, base_dir: str, cache_cfg: Dict[str, Any]
) -> None:
    """Configure the persistent extraction sidecar cache.

    Called once from ``process_and_extract.main()`` during pipeline startup.
    If caching is disabled in config, sets the global to ``None`` so that
    extraction functions skip cache lookups entirely.
    """
    global _EXTRACTION_CACHE

    global_enabled = cache_cfg.get("enabled", True)
    ext_cfg = cache_cfg.get("extraction", {})
    ext_enabled = ext_cfg.get("enabled", True)

    if not (global_enabled and ext_enabled):
        _EXTRACTION_CACHE = None
        logger.info("Extraction sidecar cache: disabled")
        return

    _EXTRACTION_CACHE = ExtractionSidecarCache(
        base_dir=base_dir,
        subdir=ext_cfg.get("subdir", "cache/extractions"),
        version=ext_cfg.get("version", 1),
        enabled=True,
    )
    logger.info(
        f"Extraction sidecar cache: enabled (v{ext_cfg.get('version', 1)}, "
        f"root={_EXTRACTION_CACHE._root})"
    )


def reset_extraction_sidecar_cache() -> None:
    """Reset the extraction sidecar cache (useful for tests)."""
    global _EXTRACTION_CACHE
    _EXTRACTION_CACHE = None


# ---------------------------------------------------------------------------
# Cache-aware extraction helper
# ---------------------------------------------------------------------------


def _maybe_cached_extract(
    *,
    text: str,
    system_prompt: str,
    response_model: Any,
    model: str,
    temperature: float,
    entity_type: Optional[str],
    llm_fn: Any,
) -> Any:
    """Try the sidecar cache before calling the LLM.

    On cache miss, calls *llm_fn* (which wraps ``cloud_generation`` or
    ``local_generation``), persists the result, and returns it.

    If the cache is not configured or ``entity_type`` is not provided,
    falls through directly to *llm_fn* with no caching.
    """
    cache = _EXTRACTION_CACHE

    if cache is None or not cache.enabled or entity_type is None:
        return llm_fn()

    key = cache.make_key(
        text=text,
        system_prompt=system_prompt,
        response_model=response_model,
        model=model,
        entity_type=entity_type,
        temperature=temperature,
    )

    # --- Cache hit ---
    record = cache.read(key)
    if record is not None:
        logger.debug(f"Extraction cache hit: {entity_type} key={key[:12]}…")
        return record["output"]

    # --- Cache miss: call LLM ---
    logger.debug(f"Extraction cache miss: {entity_type} key={key[:12]}…")
    result = llm_fn()

    # Persist (only on success — exceptions propagate without caching)
    try:
        content_hash = sha256_text(text)
        prompt_hash = sha256_text(system_prompt)
        from src.utils.extraction_cache import _schema_hash

        schema_hash = _schema_hash(response_model)

        rec = build_cache_record(
            output=result,
            entity_type=entity_type,
            model=model,
            temperature=temperature,
            content_hash=content_hash,
            prompt_hash=prompt_hash,
            schema_hash=schema_hash,
            cache_version=cache._version,
        )
        cache.write(key, rec)
    except Exception:
        logger.debug("Failed to write extraction cache entry", exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------


def extract_entities_cloud(
    text: str,
    system_prompt: str,
    response_model: Union[Type[BaseModel], List[Type[BaseModel]]],
    model: str = CLOUD_MODEL,
    temperature: float = 0,
    *,
    entity_type: Optional[str] = None,
) -> Any:
    """Extract entities from text using cloud-based language models via litellm.

    Uses the configured cloud model (typically Gemini) to extract structured entities
    from text content. Delegates to ``cloud_generation`` which handles retry logic
    for transient failures, rate limiting, and the "multiple tool calls" Instructor
    error automatically.

    When the extraction sidecar cache is active and *entity_type* is provided,
    results are read from / written to the persistent cache.

    Args:
        text: The input text content to extract entities from
        system_prompt: System prompt that defines the extraction task and format
        response_model: Pydantic model or list of models defining the expected response structure
        model: Cloud model name to use for extraction (defaults to configured CLOUD_MODEL)
        temperature: Temperature parameter for generation (0 for deterministic output)
        entity_type: Entity type string (e.g. "people") — required for caching

    Returns:
        Extracted entities structured according to the response_model specification

    Raises:
        Exception: Various exceptions from LLM API calls, network issues, or parsing failures
    """
    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    def _call_llm() -> Any:
        return cloud_generation(
            messages=messages,
            response_model=response_model,
            model=model,
            temperature=temperature,
            metadata={"tags": ["extraction"]},
        )

    return _maybe_cached_extract(
        text=text,
        system_prompt=system_prompt,
        response_model=response_model,
        model=model,
        temperature=temperature,
        entity_type=entity_type,
        llm_fn=_call_llm,
    )


def extract_entities_local(
    text: str,
    system_prompt: str,
    response_model: Any,
    model: str = OLLAMA_MODEL,
    temperature: float = 0,
    *,
    entity_type: Optional[str] = None,
) -> Any:
    """Extract entities from text using local Ollama language models.

    Uses the configured local Ollama model to extract structured entities from text
    content. Delegates to ``local_generation`` which handles retry logic and the
    "multiple tool calls" Instructor error automatically.

    When the extraction sidecar cache is active and *entity_type* is provided,
    results are read from / written to the persistent cache.

    Args:
        text: The input text content to extract entities from
        system_prompt: System prompt that defines the extraction task and format
        response_model: Pydantic model defining the expected response structure
        model: Ollama model name to use for extraction (defaults to configured OLLAMA_MODEL)
        temperature: Temperature parameter for generation (0 for deterministic output)
        entity_type: Entity type string (e.g. "people") — required for caching

    Returns:
        Extracted entities structured according to the response_model specification

    Raises:
        Exception: Various exceptions from Ollama API calls, connection issues, or parsing failures
    """
    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    def _call_llm() -> Any:
        return local_generation(
            messages=messages,
            response_model=response_model,
            model=model,
            temperature=temperature,
            metadata={"tags": ["extraction"]},
        )

    return _maybe_cached_extract(
        text=text,
        system_prompt=system_prompt,
        response_model=response_model,
        model=model,
        temperature=temperature,
        entity_type=entity_type,
        llm_fn=_call_llm,
    )
