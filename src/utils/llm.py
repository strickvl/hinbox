"""LLM utility functions and client management."""

import json
import random
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Type

import instructor
import litellm
from pydantic import BaseModel

from src.constants import (
    BASE_DELAY,
    BRAINTRUST_PROJECT_ID,
    CLOUD_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_ITERATIONS,
    MAX_RETRIES,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_llm_callbacks,
    get_ollama_model_name,
)
from src.logging_config import get_logger
from src.utils.logging import (
    log_iterative_complete,
    log_iterative_start,
)

# Get logger for this module
logger = get_logger("utils.llm")

# ---------------------------------------------------------------------------
# Global LLM concurrency limiters
# ---------------------------------------------------------------------------
_cloud_llm_semaphore: Optional[threading.Semaphore] = None
_local_llm_semaphore: Optional[threading.Semaphore] = None


def configure_llm_concurrency(
    *,
    cloud_in_flight: Optional[int] = None,
    local_in_flight: Optional[int] = None,
) -> None:
    """Set global concurrency caps for cloud / local LLM calls.

    Call once at pipeline startup. ``None`` or ``<= 0`` disables the limiter
    (unlimited concurrency).
    """
    global _cloud_llm_semaphore, _local_llm_semaphore
    if cloud_in_flight and cloud_in_flight > 0:
        _cloud_llm_semaphore = threading.Semaphore(cloud_in_flight)
        logger.info(f"Cloud LLM concurrency limited to {cloud_in_flight} in-flight")
    else:
        _cloud_llm_semaphore = None
    if local_in_flight and local_in_flight > 0:
        _local_llm_semaphore = threading.Semaphore(local_in_flight)
        logger.info(f"Local LLM concurrency limited to {local_in_flight} in-flight")
    else:
        _local_llm_semaphore = None


@contextmanager
def cloud_llm_slot() -> Iterator[None]:
    """Acquire / release a slot in the global cloud semaphore (no-op if unconfigured)."""
    if _cloud_llm_semaphore is not None:
        _cloud_llm_semaphore.acquire()
        try:
            yield
        finally:
            _cloud_llm_semaphore.release()
    else:
        yield


@contextmanager
def local_llm_slot() -> Iterator[None]:
    """Acquire / release a slot in the global local semaphore (no-op if unconfigured)."""
    if _local_llm_semaphore is not None:
        _local_llm_semaphore.acquire()
        try:
            yield
        finally:
            _local_llm_semaphore.release()
    else:
        yield


# Configure litellm once for the entire module
litellm.enable_json_schema_validation = True
litellm.suppress_debug_info = True
litellm.callbacks = get_llm_callbacks()

# Common metadata for all LLM calls
DEFAULT_METADATA: Dict[str, Any] = {
    "tags": ["dev"],
}
if BRAINTRUST_PROJECT_ID:
    DEFAULT_METADATA["project_id"] = BRAINTRUST_PROJECT_ID


class GenerationMode(str, Enum):
    """Mode for generation function."""

    CLOUD = "cloud"
    LOCAL = "local"


class ReflectionResult(BaseModel):
    """Result of a reflection evaluation."""

    valid: bool
    reasoning: str
    issues: Optional[List[str]] = None


def get_litellm_client():
    """Get a configured litellm client with instructor."""
    return instructor.from_litellm(litellm.completion)


def cloud_generation(
    messages: List[Dict[str, Any]],
    response_model: Any,
    model: str = CLOUD_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a response using a cloud model via litellm.

    Args:
        messages: List of message dictionaries
        response_model: Pydantic model for structured output
        model: Model name to use
        max_tokens: Maximum tokens in response
        temperature: Temperature for generation
        metadata: Optional metadata to merge with defaults (e.g. extra tags)
        **kwargs: Additional arguments passed to the API

    Returns:
        Parsed response according to response_model
    """
    logger.debug(f"Generating response with cloud model: {model}")

    client = get_litellm_client()

    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY

    merged_metadata = dict(DEFAULT_METADATA)
    if metadata:
        # Merge tags additively, overwrite other keys
        extra_tags = metadata.pop("tags", [])
        merged_metadata.update(metadata)
        if extra_tags:
            existing_tags = merged_metadata.get("tags", [])
            merged_metadata["tags"] = list(set(existing_tags + extra_tags))
    metadata = merged_metadata

    for attempt in range(max_retries + 1):
        try:
            with cloud_llm_slot():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_model=response_model,
                    metadata=metadata,
                    **kwargs,
                )
            logger.debug(f"Successfully generated response with {model}")
            return response
        except Exception as e:
            error_str = str(e)

            # Handle Instructor multiple tool calls error
            if "multiple tool calls" in error_str.lower():
                logger.warning(
                    f"Multiple tool calls detected for {model}, attempting recovery strategies"
                )

                # Strategy 1: Try with lower temperature and max_retries=0 to force single response
                try:
                    logger.debug("Trying with temperature=0 and max_retries=0")
                    with cloud_llm_slot():
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=0,
                            response_model=response_model,
                            metadata=metadata,
                            max_retries=0,
                            **kwargs,
                        )
                    logger.debug(
                        "Recovered from multiple tool calls error with strategy 1"
                    )
                    return response
                except Exception as recovery_e1:
                    logger.warning(f"Strategy 1 failed: {recovery_e1}")

                # Strategy 2: Try modifying the messages to be more explicit
                try:
                    logger.debug("Trying with modified system message")
                    modified_messages = messages.copy()
                    if modified_messages and modified_messages[0]["role"] == "system":
                        modified_messages[0]["content"] += (
                            "\n\nIMPORTANT: Provide exactly ONE response. Do not make multiple tool calls."
                        )

                    with cloud_llm_slot():
                        response = client.chat.completions.create(
                            model=model,
                            messages=modified_messages,
                            max_tokens=max_tokens,
                            temperature=0,
                            response_model=response_model,
                            metadata=metadata,
                            max_retries=0,
                            **kwargs,
                        )
                    logger.debug(
                        "Recovered from multiple tool calls error with strategy 2"
                    )
                    return response
                except Exception as recovery_e2:
                    logger.warning(f"Strategy 2 failed: {recovery_e2}")
                    logger.error(
                        "All recovery strategies failed, will raise original error"
                    )

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
                logger.debug(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue

            logger.error(f"Cloud generation failed: {error_str}")
            raise


def local_generation(
    messages: List[Dict[str, Any]],
    response_model: Any,
    model: str = OLLAMA_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a response using a local Ollama model via LiteLLM + instructor.

    Args:
        messages: List of message dictionaries
        response_model: Pydantic model for structured output
        model: Model name to use
        max_tokens: Maximum tokens in response
        temperature: Temperature for generation
        metadata: Optional metadata to merge with defaults (e.g. extra tags)
        **kwargs: Additional arguments passed to the API

    Returns:
        Parsed response according to response_model
    """
    logger.debug(f"Generating response with local model: {model}")

    client = get_litellm_client()

    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY

    merged_metadata = dict(DEFAULT_METADATA)
    if metadata:
        extra_tags = metadata.pop("tags", [])
        merged_metadata.update(metadata)
        if extra_tags:
            existing_tags = merged_metadata.get("tags", [])
            merged_metadata["tags"] = list(set(existing_tags + extra_tags))
    metadata = merged_metadata

    for attempt in range(max_retries + 1):
        try:
            with local_llm_slot():
                response = client.chat.completions.create(
                    model=get_ollama_model_name(model),
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_model=response_model,
                    metadata=metadata,
                    api_base=OLLAMA_API_URL,
                    custom_llm_provider="ollama",
                    **kwargs,
                )
            logger.debug(f"Successfully generated response with {model}")
            return response
        except Exception as e:
            error_str = str(e)

            # Handle Instructor multiple tool calls error
            if "multiple tool calls" in error_str.lower():
                logger.warning(
                    f"Multiple tool calls detected for local model {model}, attempting recovery strategies"
                )

                # Strategy 1: Try with lower temperature and max_retries=0
                try:
                    logger.debug("Trying local with temperature=0 and max_retries=0")
                    with local_llm_slot():
                        response = client.chat.completions.create(
                            model=get_ollama_model_name(model),
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=0,
                            response_model=response_model,
                            metadata=metadata,
                            api_base=OLLAMA_API_URL,
                            custom_llm_provider="ollama",
                            max_retries=0,
                            **kwargs,
                        )
                    logger.debug("Recovered locally with strategy 1")
                    return response
                except Exception as recovery_e1:
                    logger.warning(f"Local strategy 1 failed: {recovery_e1}")

                # Strategy 2: Modify system message
                try:
                    logger.debug("Trying local with modified system message")
                    modified_messages = messages.copy()
                    if modified_messages and modified_messages[0]["role"] == "system":
                        modified_messages[0]["content"] += (
                            "\n\nIMPORTANT: Provide exactly ONE response. Do not make multiple tool calls."
                        )

                    with local_llm_slot():
                        response = client.chat.completions.create(
                            model=get_ollama_model_name(model),
                            messages=modified_messages,
                            max_tokens=max_tokens,
                            temperature=0,
                            response_model=response_model,
                            metadata=metadata,
                            api_base=OLLAMA_API_URL,
                            custom_llm_provider="ollama",
                            max_retries=0,
                            **kwargs,
                        )
                    logger.debug("Recovered locally with strategy 2")
                    return response
                except Exception as recovery_e2:
                    logger.warning(f"Local strategy 2 failed: {recovery_e2}")
                    logger.error(
                        "All local recovery strategies failed, will raise original error"
                    )

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
                    f"Retryable local error on attempt {attempt + 1}/{max_retries + 1}: {error_str}"
                )
                logger.debug(f"Retrying local in {delay:.2f} seconds...")
                time.sleep(delay)
                continue

            logger.error(f"Local generation failed: {error_str}")
            raise


def reflect_and_check(
    text: str,
    reasoning_prompt: str,
    mode: GenerationMode = GenerationMode.CLOUD,
    model: str = None,
) -> ReflectionResult:
    """
    Use an LLM to reflect on generated text and check if it meets requirements.

    Args:
        text: The text to evaluate
        reasoning_prompt: Prompt explaining what to check for
        mode: Whether to use cloud or local model
        model: Optional model override

    Returns:
        ReflectionResult with validity and reasoning
    """
    logger.debug(f"Starting reflection check with {mode.value} model")

    messages = [
        {
            "role": "system",
            "content": "You are a quality assurance expert who evaluates generated text.",
        },
        {
            "role": "user",
            "content": f"{reasoning_prompt}\n\nText to evaluate:\n{text}",
        },
    ]

    try:
        if mode == GenerationMode.CLOUD:
            result = cloud_generation(
                messages=messages,
                response_model=ReflectionResult,
                model=model or CLOUD_MODEL,
            )
        else:
            result = local_generation(
                messages=messages,
                response_model=ReflectionResult,
                model=model or OLLAMA_MODEL,
            )

        logger.debug(f"Reflection check completed - valid: {result.valid}")
        return result
    except Exception as e:
        logger.error(f"Reflection check failed: {e}")
        raise


def extract_reflection_result(reflection: Any) -> ReflectionResult:
    """
    Extract ReflectionResult from various response formats.

    Args:
        reflection: Response from reflect_and_check

    Returns:
        ReflectionResult object
    """
    if isinstance(reflection, ReflectionResult):
        return reflection
    elif hasattr(reflection, "choices") and reflection.choices:
        return reflection.choices[0].message.parsed
    elif isinstance(reflection, dict):
        return ReflectionResult(**reflection)
    else:
        raise ValueError(f"Cannot extract ReflectionResult from {type(reflection)}")


def coerce_to_json_text(value: Any) -> str:
    """Convert a structured LLM response into its full JSON string representation.

    This preserves the entire schema envelope (e.g. tags, confidence, sources)
    rather than extracting a single field like `.text` which would lose metadata.
    """
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if hasattr(value, "choices") and value.choices:
        parsed = value.choices[0].message.parsed
        if isinstance(parsed, BaseModel):
            return parsed.model_dump_json()
        return json.dumps(parsed) if parsed is not None else str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def iterative_improve(
    initial_text: str,
    generation_messages: List[Dict[str, Any]],
    reflection_prompt: str,
    response_model: Type[BaseModel],
    max_iterations: int = MAX_ITERATIONS,
    mode: GenerationMode = GenerationMode.CLOUD,
    model: str = None,
) -> Dict[str, Any]:
    """
    Iteratively improve text using generation and reflection.

    Args:
        initial_text: Starting text to improve
        generation_messages: Messages for generation
        reflection_prompt: Prompt for reflection/evaluation
        response_model: Pydantic model for structured output
        max_iterations: Maximum improvement iterations
        mode: Whether to use cloud or local model
        model: Optional model override

    Returns:
        Dict containing final text and reflection history
    """
    prompt_preview = (
        str(generation_messages)[:100] + "..."
        if len(str(generation_messages)) > 100
        else str(generation_messages)
    )
    start_time = log_iterative_start(
        prompt_preview,
        model or (CLOUD_MODEL if mode == GenerationMode.CLOUD else OLLAMA_MODEL),
        max_iterations,
    )

    current_text = initial_text
    reflection_history = []

    for i in range(max_iterations):
        logger.debug(f"Iteration {i + 1}/{max_iterations}")

        # Reflect on current text
        reflection = reflect_and_check(
            current_text,
            reflection_prompt,
            mode=mode,
            model=model,
        )
        reflection_result = extract_reflection_result(reflection)

        reflection_history.append(
            {
                "iteration": i + 1,
                "valid": reflection_result.valid,
                "reasoning": reflection_result.reasoning,
                "issues": reflection_result.issues,
            }
        )

        if reflection_result.valid:
            logger.debug(f"Text validated after {i + 1} iterations")
            break

        # Generate improved version
        improvement_messages = generation_messages + [
            {
                "role": "user",
                "content": f"VALIDATION FAILED. The current response has these specific issues:\n\n{reflection_result.reasoning}\n\nPlease generate an improved version that addresses ALL the issues above. Pay special attention to:\n- Adding strategic citations with ^[article_id] format (single ID per citation, group facts in paragraphs)\n- Including section headers (### Title)\n- Ensuring minimum 100 character length\n- Valid JSON structure with all required fields\n- At least 2 relevant tags",
            }
        ]

        if mode == GenerationMode.CLOUD:
            improved = cloud_generation(
                messages=improvement_messages,
                response_model=response_model,
                model=model or CLOUD_MODEL,
            )
        else:
            improved = local_generation(
                messages=improvement_messages,
                response_model=response_model,
                model=model or OLLAMA_MODEL,
            )

        # Serialize the full structured response to preserve all fields
        # (tags, confidence, sources) — not just the inner prose `.text`
        current_text = coerce_to_json_text(improved)

    # Log completion
    log_iterative_complete(
        start_time,
        len(reflection_history),
        max_iterations,
        reflection_history[-1]["valid"] if reflection_history else False,
    )

    return {
        "text": current_text,
        "reflection_history": reflection_history,
        "final_valid": reflection_history[-1]["valid"] if reflection_history else False,
    }


def create_messages(
    system_content: str,
    user_content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Create a standard message list for LLM calls.

    Args:
        system_content: System message content
        user_content: User message content
        metadata: Optional metadata to merge with defaults

    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Merge custom metadata with defaults if provided
    if metadata:
        merged_metadata = {**DEFAULT_METADATA, **metadata}
        messages[0]["metadata"] = merged_metadata

    return messages
