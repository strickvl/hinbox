"""LLM utility functions and client management."""

from enum import Enum
from typing import Any, Dict, List, Optional, Type

import instructor
import litellm
from openai import OpenAI
from pydantic import BaseModel

from src.constants import (
    BRAINTRUST_PROJECT_ID,
    BRAINTRUST_PROJECT_NAME,
    CLOUD_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_ollama_model_name,
)
from src.logging_config import get_logger
from src.utils_logging import (
    log_iterative_complete,
    log_iterative_start,
)

# Get logger for this module
logger = get_logger("utils.llm")

# Configure litellm once for the entire module
litellm.enable_json_schema_validation = True
litellm.suppress_debug_info = True
litellm.callbacks = ["braintrust", "langfuse"]

# Common metadata for all LLM calls
DEFAULT_METADATA = {
    "tags": ["dev"],
}

# Add Braintrust project configuration if available
if BRAINTRUST_PROJECT_ID:
    DEFAULT_METADATA["project_id"] = BRAINTRUST_PROJECT_ID
elif BRAINTRUST_PROJECT_NAME:
    DEFAULT_METADATA["project_name"] = BRAINTRUST_PROJECT_NAME


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


def get_ollama_client():
    """Get a configured Ollama OpenAI client."""
    return OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)


def cloud_generation(
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str = CLOUD_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0,
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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model=response_model,
            metadata=DEFAULT_METADATA,
            **kwargs,
        )
        logger.debug(f"Successfully generated response with {model}")
        return response
    except Exception as e:
        logger.error(f"Cloud generation failed: {str(e)}")
        raise


def local_generation(
    messages: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    model: str = OLLAMA_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0,
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

    try:
        response = client.beta.chat.completions.parse(
            model=get_ollama_model_name(model),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_model,
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


def iterative_improve(
    initial_text: str,
    generation_messages: List[Dict[str, Any]],
    reflection_prompt: str,
    response_model: Type[BaseModel],
    max_iterations: int = 3,
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
            current_text, reflection_prompt, mode=mode, model=model
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
                "content": f"The current text has these issues: {reflection_result.reasoning}\n\nPlease generate an improved version.",
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
