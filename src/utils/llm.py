"""LLM utility functions and client management."""

import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import instructor
import litellm
from pydantic import BaseModel

# Langfuse OpenAI wrapper is optional. In some environments (e.g., tests) the
# underlying OpenAI beta resources may be missing; importing the wrapper would
# raise at import-time. We guard this to keep module import safe for tests.
try:
    from langfuse.openai import OpenAI as LangfuseOpenAI  # type: ignore
    _LANGFUSE_AVAILABLE = True
except Exception:
    LangfuseOpenAI = None  # type: ignore[assignment]
    _LANGFUSE_AVAILABLE = False

from src.constants import (
    BASE_DELAY,
    BRAINTRUST_PROJECT_ID,
    CLOUD_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_ITERATIONS,
    MAX_RETRIES,
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_ollama_model_name,
)
from src.logging_config import get_logger
from src.utils.logging import (
    log_iterative_complete,
    log_iterative_start,
)

# Get logger for this module
logger = get_logger("utils.llm")

# Configure litellm once for the entire module
litellm.enable_json_schema_validation = True
litellm.suppress_debug_info = True
litellm.callbacks = ["braintrust"]

# Common metadata for all LLM calls
DEFAULT_METADATA = {
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
    # Use default mode, but we'll handle multiple tool calls in error handling
    return instructor.from_litellm(litellm.completion)


def get_ollama_client():
    """Get a configured Ollama OpenAI-style client.

    Preference is given to the Langfuse OpenAI wrapper when available. In test
    environments where langfuse (or the OpenAI beta resources it patches) is
    not installed, this returns a minimal stub that raises a clear error when
    used. This keeps imports cheap and avoids hard dependencies during test
    collection while preserving explicit failure if local_generation is invoked.
    """
    if _LANGFUSE_AVAILABLE and LangfuseOpenAI is not None:
        return LangfuseOpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    class _MissingClient:
        def __init__(self, base_url: str, api_key: str):
            self.base_url = base_url
            self.api_key = api_key

        class _Beta:
            class _Chat:
                class _Completions:
                    @staticmethod
                    def parse(*args, **kwargs):
                        raise RuntimeError(
                            "Local LLM client unavailable: langfuse.openai is not installed in this environment."
                        )

                completions = _Completions()

            chat = _Chat()

        beta = _Beta()

    return _MissingClient(OLLAMA_API_URL, OLLAMA_API_KEY)


def cloud_generation(
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str = CLOUD_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
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
        **kwargs: Additional arguments passed to the API

    Returns:
        Parsed response according to response_model
    """
    # Not using iteration-based logging for direct generation
    logger.debug(f"Generating response with cloud model: {model}")

    client = get_litellm_client()

    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY

    metadata = dict(DEFAULT_METADATA)
    if langfuse_trace_id is not None:
        metadata["span_name"] = langfuse_trace_id
    if langfuse_session_id is not None:
        metadata["session_id"] = langfuse_session_id

    for attempt in range(max_retries + 1):
        try:
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
                    logger.info("Trying with temperature=0 and max_retries=0")
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
                    logger.info(
                        "✓ Successfully recovered from multiple tool calls error with strategy 1"
                    )
                    return response
                except Exception as recovery_e1:
                    logger.warning(f"Strategy 1 failed: {recovery_e1}")

                # Strategy 2: Try modifying the messages to be more explicit
                try:
                    logger.info("Trying with modified system message")
                    modified_messages = messages.copy()
                    if modified_messages and modified_messages[0]["role"] == "system":
                        modified_messages[0]["content"] += (
                            "\n\nIMPORTANT: Provide exactly ONE response. Do not make multiple tool calls."
                        )

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
                    logger.info(
                        "✓ Successfully recovered from multiple tool calls error with strategy 2"
                    )
                    return response
                except Exception as recovery_e2:
                    logger.warning(f"Strategy 2 failed: {recovery_e2}")
                    logger.error(
                        f"All recovery strategies failed, will raise original error"
                    )
                    # Fall through to raise the original error

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

            logger.error(f"Cloud generation failed: {error_str}")
            raise


def local_generation(
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str = OLLAMA_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
    **kwargs: Any,
) -> Any:
    """
    Generate a response using a local Ollama model.

    Args:
        messages: List of message dictionaries
        response_model: Pydantic model for structured output
        model: Model name to use
        max_tokens: Maximum tokens in response
        temperature: Temperature for generation
        **kwargs: Additional arguments passed to the API

    Returns:
        Parsed response according to response_model
    """
    # Not using iteration-based logging for direct generation
    logger.debug(f"Generating response with local model: {model}")

    client = get_ollama_client()

    metadata = dict(DEFAULT_METADATA)
    if langfuse_trace_id is not None:
        metadata["span_name"] = langfuse_trace_id
    if langfuse_session_id is not None:
        metadata["session_id"] = langfuse_session_id

    try:
        response = client.beta.chat.completions.parse(
            model=get_ollama_model_name(model),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_model,
            metadata=metadata,
            **kwargs,
        )
        result = response.choices[0].message.parsed
        logger.debug(f"Successfully generated response with {model}")
        return result
    except Exception as e:
        logger.error(f"Local generation failed: {str(e)}")
        raise


def reflect_and_check(
    text: str,
    reasoning_prompt: str,
    mode: GenerationMode = GenerationMode.CLOUD,
    model: str = None,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
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
    # Log evaluation without iteration context
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
                langfuse_session_id=langfuse_session_id,
                langfuse_trace_id=langfuse_trace_id,
            )
        else:
            result = local_generation(
                messages=messages,
                response_model=ReflectionResult,
                model=model or OLLAMA_MODEL,
                langfuse_session_id=langfuse_session_id,
                langfuse_trace_id=langfuse_trace_id,
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


def iterative_improve(
    initial_text: str,
    generation_messages: List[Dict[str, Any]],
    reflection_prompt: str,
    response_model: Type[BaseModel],
    max_iterations: int = MAX_ITERATIONS,
    mode: GenerationMode = GenerationMode.CLOUD,
    model: str = None,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
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
    # Log start of iterative process
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
        logger.info(f"Iteration {i + 1}/{max_iterations}")

        # Reflect on current text
        reflection = reflect_and_check(
            current_text,
            reflection_prompt,
            mode=mode,
            model=model,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
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
            logger.info(f"Text validated after {i + 1} iterations")
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
                langfuse_session_id=langfuse_session_id,
                langfuse_trace_id=langfuse_trace_id,
            )
        else:
            improved = local_generation(
                messages=improvement_messages,
                response_model=response_model,
                model=model or OLLAMA_MODEL,
                langfuse_session_id=langfuse_session_id,
                langfuse_trace_id=langfuse_trace_id,
            )

        # Extract text from response
        if hasattr(improved, "text"):
            current_text = improved.text
        elif hasattr(improved, "content"):
            current_text = improved.content
        else:
            current_text = str(improved)

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
