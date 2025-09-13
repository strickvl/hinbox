"""Generic entity extraction utilities for cloud and local LLM models.

This module provides generic extraction functions that work with both cloud-based
and local language models. These functions form the core of the entity extraction
pipeline and are used by specific entity extractors for people, organizations,
locations, and events.
"""

import random
import time
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

from src.constants import (
    BASE_DELAY,
    CLOUD_MODEL,
    MAX_RETRIES,
    OLLAMA_MODEL,
)
from src.logging_config import get_logger
from src.utils.llm import (
    DEFAULT_METADATA,
    create_messages,
    get_litellm_client,
    get_ollama_client,
)

logger = get_logger(__name__)


def extract_entities_cloud(
    text: str,
    system_prompt: str,
    response_model: Union[Type[BaseModel], List[Type[BaseModel]]],
    model: str = CLOUD_MODEL,
    temperature: float = 0,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> Any:
    """Extract entities from text using cloud-based language models via litellm.

    Uses the configured cloud model (typically Gemini) to extract structured entities
    from text content. Includes retry logic for transient failures and comprehensive
    error handling.

    Args:
        text: The input text content to extract entities from
        system_prompt: System prompt that defines the extraction task and format
        response_model: Pydantic model or list of models defining the expected response structure
        model: Cloud model name to use for extraction (defaults to configured CLOUD_MODEL)
        temperature: Temperature parameter for generation (0 for deterministic output)
        langfuse_session_id: Optional Langfuse session ID for request tracing
        langfuse_trace_id: Optional Langfuse trace ID for request tracing

    Returns:
        Extracted entities structured according to the response_model specification

    Raises:
        Exception: Various exceptions from LLM API calls, network issues, or parsing failures

    Note:
        Implements exponential backoff retry logic for transient errors like rate limiting
        or server overload. Non-retryable errors are raised immediately.
    """
    client = get_litellm_client()

    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    metadata = dict(DEFAULT_METADATA)
    metadata["tags"] = ["dev", "extraction"]
    if langfuse_trace_id is not None:
        metadata["span_name"] = langfuse_trace_id
    if langfuse_session_id is not None:
        metadata["session_id"] = langfuse_session_id

    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                response_model=response_model,
                temperature=temperature,
                messages=messages,
                metadata=metadata,
            )
        except Exception as e:
            error_str = str(e)

            # Check for retryable errors (503, 529, rate limiting)
            is_retryable = (
                "503" in error_str
                or "529" in error_str
                or "overloaded" in error_str.lower()
                or "rate limit" in error_str.lower()
                or "try again" in error_str.lower()
            )

            if is_retryable and attempt < max_retries:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Retryable error on attempt {attempt + 1}/{max_retries + 1}: {error_str}"
                )
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue

            logger.error(f"Cloud extraction failed: {error_str}")
            raise


def extract_entities_local(
    text: str,
    system_prompt: str,
    response_model: Type[BaseModel],
    model: str = OLLAMA_MODEL,
    temperature: float = 0,
    langfuse_session_id: Optional[str] = None,
    langfuse_trace_id: Optional[str] = None,
) -> Any:
    """Extract entities from text using local Ollama language models.

    Uses the configured local Ollama model to extract structured entities from text
    content. Requires a running Ollama server and does not include retry logic
    as local models typically don't experience transient failures.

    Args:
        text: The input text content to extract entities from
        system_prompt: System prompt that defines the extraction task and format
        response_model: Pydantic model defining the expected response structure
        model: Ollama model name to use for extraction (defaults to configured OLLAMA_MODEL)
        temperature: Temperature parameter for generation (0 for deterministic output)
        langfuse_session_id: Optional Langfuse session ID for request tracing
        langfuse_trace_id: Optional Langfuse trace ID for request tracing

    Returns:
        Extracted entities structured according to the response_model specification

    Raises:
        Exception: Various exceptions from Ollama API calls, connection issues, or parsing failures

    Note:
        Requires Ollama server to be running and accessible at the configured URL.
        Model names are automatically mapped to Ollama-compatible format.
        No retry logic as local models typically don't have transient failures.
    """
    from src.constants import get_ollama_model_name

    client = get_ollama_client()

    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    metadata = dict(DEFAULT_METADATA)
    metadata["tags"] = ["dev", "extraction"]
    if langfuse_trace_id is not None:
        metadata["span_name"] = langfuse_trace_id
    if langfuse_session_id is not None:
        metadata["session_id"] = langfuse_session_id

    results = client.beta.chat.completions.parse(
        model=get_ollama_model_name(model),
        response_format=response_model,
        temperature=temperature,
        messages=messages,
        metadata=metadata,
    )

    return results.choices[0].message.parsed
