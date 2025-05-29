"""Generic entity extraction utilities."""

import random
import time
from typing import Any, List, Type, Union

from pydantic import BaseModel

from src.constants import (
    BASE_DELAY,
    BRAINTRUST_PROJECT_ID,
    BRAINTRUST_PROJECT_NAME,
    CLOUD_MODEL,
    MAX_RETRIES,
    OLLAMA_MODEL,
)
from src.logging_config import get_logger
from src.utils.llm import (
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
) -> Any:
    """
    Generic cloud-based entity extraction.

    Args:
        text: Text to extract entities from
        system_prompt: System prompt defining extraction task
        response_model: Pydantic model or list of models for response
        model: Model to use for extraction
        temperature: Temperature for generation

    Returns:
        Extracted entities according to response_model
    """
    client = get_litellm_client()

    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    # Build metadata with Braintrust configuration
    metadata = {"tags": ["dev", "extraction"]}
    if BRAINTRUST_PROJECT_ID:
        metadata["project_id"] = BRAINTRUST_PROJECT_ID
    elif BRAINTRUST_PROJECT_NAME:
        metadata["project_name"] = BRAINTRUST_PROJECT_NAME

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
) -> Any:
    """
    Generic local Ollama-based entity extraction.

    Args:
        text: Text to extract entities from
        system_prompt: System prompt defining extraction task
        response_model: Pydantic model for response
        model: Model to use for extraction
        temperature: Temperature for generation

    Returns:
        Extracted entities according to response_model
    """
    from src.constants import get_ollama_model_name

    client = get_ollama_client()

    messages = create_messages(
        system_content=system_prompt,
        user_content=text,
    )

    results = client.beta.chat.completions.parse(
        model=get_ollama_model_name(model),
        response_format=response_model,
        temperature=temperature,
        messages=messages,
    )

    return results.choices[0].message.parsed


# Backward compatibility - these functions now load from config files
# The actual prompts are stored in configs/{domain}/prompts/*.md files
