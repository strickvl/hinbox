"""Generic entity extraction utilities for cloud and local LLM models.

This module provides generic extraction functions that work with both cloud-based
and local language models. These functions form the core of the entity extraction
pipeline and are used by specific entity extractors for people, organizations,
locations, and events.

Extraction calls are routed through ``cloud_generation``/``local_generation``
in ``src.utils.llm`` so that all retry, rate-limit, and "multiple tool calls"
recovery strategies apply uniformly.
"""

from typing import Any, List, Type, Union

from pydantic import BaseModel

from src.constants import (
    CLOUD_MODEL,
    OLLAMA_MODEL,
)
from src.logging_config import get_logger
from src.utils.llm import (
    cloud_generation,
    create_messages,
    local_generation,
)

logger = get_logger(__name__)


def extract_entities_cloud(
    text: str,
    system_prompt: str,
    response_model: Union[Type[BaseModel], List[Type[BaseModel]]],
    model: str = CLOUD_MODEL,
    temperature: float = 0,
) -> Any:
    """Extract entities from text using cloud-based language models via litellm.

    Uses the configured cloud model (typically Gemini) to extract structured entities
    from text content. Delegates to ``cloud_generation`` which handles retry logic
    for transient failures, rate limiting, and the "multiple tool calls" Instructor
    error automatically.

    Args:
        text: The input text content to extract entities from
        system_prompt: System prompt that defines the extraction task and format
        response_model: Pydantic model or list of models defining the expected response structure
        model: Cloud model name to use for extraction (defaults to configured CLOUD_MODEL)
        temperature: Temperature parameter for generation (0 for deterministic output)

    Returns:
        Extracted entities structured according to the response_model specification

    Raises:
        Exception: Various exceptions from LLM API calls, network issues, or parsing failures
    """
    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    return cloud_generation(
        messages=messages,
        response_model=response_model,
        model=model,
        temperature=temperature,
        metadata={"tags": ["extraction"]},
    )


def extract_entities_local(
    text: str,
    system_prompt: str,
    response_model: Any,
    model: str = OLLAMA_MODEL,
    temperature: float = 0,
) -> Any:
    """Extract entities from text using local Ollama language models.

    Uses the configured local Ollama model to extract structured entities from text
    content. Delegates to ``local_generation`` which handles retry logic and the
    "multiple tool calls" Instructor error automatically.

    Args:
        text: The input text content to extract entities from
        system_prompt: System prompt that defines the extraction task and format
        response_model: Pydantic model defining the expected response structure
        model: Ollama model name to use for extraction (defaults to configured OLLAMA_MODEL)
        temperature: Temperature parameter for generation (0 for deterministic output)

    Returns:
        Extracted entities structured according to the response_model specification

    Raises:
        Exception: Various exceptions from Ollama API calls, connection issues, or parsing failures
    """
    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    return local_generation(
        messages=messages,
        response_model=response_model,
        model=model,
        temperature=temperature,
        metadata={"tags": ["extraction"]},
    )
