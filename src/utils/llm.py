"""LLM utility functions and client management."""

import json
import random
import re
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

import instructor
from openai import OpenAI
from pydantic import BaseModel

from src.constants import (
    BASE_DELAY,
    CLOUD_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_ITERATIONS,
    MAX_RETRIES,
    OLLAMA_MODEL,
    OLLAMA_REASONING_EFFORT,
    OLLAMA_THINKING_MAX_TOKENS,
)
from src.logging_config import get_logger
from src.utils.logging import (
    log_iterative_complete,
    log_iterative_start,
)
from src.utils.provider_routing import ProviderTarget, resolve_chat_target
from src.utils.step_timing import record_llm_call, timed, timed_step

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
def _llm_slot(semaphore: Optional[threading.Semaphore]) -> Iterator[None]:
    """Hold a slot in ``semaphore`` (if set) and time the call for ``step_timing``."""
    start = time.perf_counter()
    if semaphore is not None:
        semaphore.acquire()
    acquired = time.perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        if semaphore is not None:
            semaphore.release()
        record_llm_call(
            wait_seconds=acquired - start,
            call_seconds=time.perf_counter() - acquired,
            ok=ok,
        )


def cloud_llm_slot() -> Any:
    """Acquire / release a slot in the global cloud semaphore (no-op if unconfigured)."""
    return _llm_slot(_cloud_llm_semaphore)


def local_llm_slot() -> Any:
    """Acquire / release a slot in the global local semaphore (no-op if unconfigured)."""
    return _llm_slot(_local_llm_semaphore)


class GenerationMode(str, Enum):
    """Mode for generation function."""

    CLOUD = "cloud"
    LOCAL = "local"


class ReflectionResult(BaseModel):
    """Result of a reflection evaluation."""

    valid: bool
    reasoning: str
    issues: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Provider-aware Instructor client helpers
# ---------------------------------------------------------------------------


def _resolve_instructor_mode(
    target: ProviderTarget,
    *,
    parallel: bool,
    json_fallback: bool = False,
) -> instructor.Mode:
    """Pick the right Instructor mode for the given provider target."""
    if target.sdk == "anthropic":
        if json_fallback:
            return instructor.Mode.ANTHROPIC_JSON
        return (
            instructor.Mode.ANTHROPIC_PARALLEL_TOOLS
            if parallel
            else instructor.Mode.ANTHROPIC_TOOLS
        )
    # OpenAI-compatible providers
    if json_fallback:
        return instructor.Mode.MD_JSON
    return instructor.Mode.PARALLEL_TOOLS if parallel else instructor.Mode.TOOLS


def _create_raw_client(target: ProviderTarget) -> Any:
    """Create the plain SDK client (Anthropic or OpenAI-compatible) for a target."""
    if target.sdk == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for anthropic/* models. "
                "Install it with: uv pip install 'hinbox[anthropic]'"
            ) from None
        return Anthropic(api_key=target.api_key)

    # OpenAI-compatible (OpenAI, Gemini, Ollama, OpenRouter, etc.).
    # For OpenAI itself, api_key=None lets the SDK read OPENAI_API_KEY.
    return OpenAI(
        base_url=target.base_url,
        api_key=target.api_key
        if target.provider_label == "openai"
        else (target.api_key or "not-set"),
    )


def _create_instructor_client(
    target: ProviderTarget,
    mode: instructor.Mode,
) -> Any:
    """Create an Instructor-wrapped SDK client for the resolved provider."""
    raw = _create_raw_client(target)
    if target.sdk == "anthropic":
        return instructor.from_anthropic(raw, mode=mode)
    return instructor.from_openai(raw, mode=mode)


def _is_retryable(error_str: str) -> bool:
    """Overload and rate-limit errors (503, 529, ...) that are worth retrying."""
    lowered = error_str.lower()
    return (
        "503" in error_str
        or "529" in error_str
        or "overloaded" in lowered
        or "rate limit" in lowered
        or "try again" in lowered
    )


# OpenAI reasoning models reject ``max_tokens`` and a non-default temperature
_OPENAI_REASONING_MODEL = re.compile(r"^(gpt-5|gpt-6|o\d)")
# Of those, these accept reasoning_effort="none", which tool calling in Chat
# Completions needs (gpt-5 itself and the o-series only go down to "minimal"/"low")
_OPENAI_NO_REASONING_OK = re.compile(r"^(gpt-5\.\d|gpt-6)")
# Claude 5-family models reject ``temperature``
_ANTHROPIC_NO_TEMPERATURE = re.compile(r"^claude-[a-z]+-5")


# Output-token ceilings for models that reject a larger request. Profile calls
# ask for up to PROFILE_MAX_TOKENS, which these models can't produce anyway.
_OUTPUT_TOKEN_CAPS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r"^gemini-(1\.5|2\.0)"), 8192),
)


def _sampling_kwargs(
    target: ProviderTarget, max_tokens: int, temperature: float
) -> Dict[str, Any]:
    """Length/temperature/reasoning parameters the target model accepts."""
    model = target.api_model
    for pattern, cap in _OUTPUT_TOKEN_CAPS:
        if pattern.match(model):
            max_tokens = min(max_tokens, cap)
    if target.provider_label == "openai" and _OPENAI_REASONING_MODEL.match(model):
        kwargs: Dict[str, Any] = {"max_completion_tokens": max_tokens}
        if _OPENAI_NO_REASONING_OK.match(model):
            kwargs["reasoning_effort"] = "none"
        return kwargs
    if target.sdk == "anthropic" and _ANTHROPIC_NO_TEMPERATURE.match(model):
        return {"max_tokens": max_tokens}
    kwargs = {"max_tokens": max_tokens, "temperature": temperature}
    if target.provider_label == "ollama":
        if OLLAMA_REASONING_EFFORT:
            kwargs["reasoning_effort"] = OLLAMA_REASONING_EFFORT
        if OLLAMA_REASONING_EFFORT != "none":
            # Thinking may be on, so leave room for it before the answer
            kwargs["max_tokens"] = max(max_tokens, OLLAMA_THINKING_MAX_TOKENS)
    return kwargs


def _prepare_messages_for_target(
    messages: List[Dict[str, Any]],
    target: ProviderTarget,
) -> Dict[str, Any]:
    """Prepare messages and extra create-kwargs for the target provider.

    The Anthropic API requires ``system`` as a separate parameter rather than
    a role in the messages list.  This helper extracts it when needed so that
    callers can keep passing a standard OpenAI-style messages list.
    """
    if target.sdk == "anthropic" and messages and messages[0].get("role") == "system":
        return {
            "system": messages[0]["content"],
            "messages": messages[1:],
        }
    return {"messages": messages}


# ---------------------------------------------------------------------------
# Structured-output helpers
# ---------------------------------------------------------------------------


def _list_item_model(response_model: Any) -> Optional[type[BaseModel]]:
    """Return the item model for a ``List[Model]`` response type."""
    origin = get_origin(response_model)
    if origin is list:
        args = get_args(response_model)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return None


def _count_tool_calls(error: Exception) -> str:
    """Tool calls in the completion behind an Instructor error, if known."""
    try:
        message = error.last_completion.choices[0].message  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return "an unknown number of"
    return str(len(getattr(message, "tool_calls", None) or []))


def _recover_multiple_tool_calls(
    *,
    error: Exception,
    response_model: Any,
) -> Optional[Any]:
    """Recover from Instructor's "multiple tool calls" assertion when possible.

    Instructor fails fast when a provider emits more than one tool call. For
    ``List[Model]`` response shapes, each tool call is often one item. We can
    parse those arguments directly from ``last_completion`` and validate them.
    """
    completion = getattr(error, "last_completion", None)
    if completion is None:
        return None

    choices = getattr(completion, "choices", None)
    if not choices:
        return None

    message = getattr(choices[0], "message", None)
    if message is None:
        return None

    tool_calls = getattr(message, "tool_calls", None) or []
    if len(tool_calls) <= 1:
        return None

    item_model = _list_item_model(response_model)
    if item_model is None:
        return None

    recovered: List[BaseModel] = []
    for call in tool_calls:
        fn = getattr(call, "function", None)
        args_payload = getattr(fn, "arguments", None)
        if args_payload is None:
            return None

        if isinstance(args_payload, str):
            payload = json.loads(args_payload)
        elif isinstance(args_payload, dict):
            payload = args_payload
        else:
            return None

        recovered.append(item_model.model_validate(payload, strict=False))

    logger.warning(
        "Recovered %d items from multiple tool calls without retry", len(recovered)
    )
    return recovered


def _parallel_tools_response_model(response_model: Any) -> Optional[Any]:
    """Return an Iterable[Model] response type for Instructor parallel-tools mode."""
    item_model = _list_item_model(response_model)
    if item_model is None:
        return None
    return Iterable[item_model]


def _normalize_generation_response(response: Any, *, parallel_tools: bool) -> Any:
    """Normalize Instructor return shape to match current callers' expectations."""
    if parallel_tools:
        return list(response)
    return response


# ---------------------------------------------------------------------------
# Unified structured generation (shared by cloud and local)
# ---------------------------------------------------------------------------


def _structured_generation(
    messages: List[Dict[str, Any]],
    response_model: Any,
    model: str,
    max_tokens: int,
    temperature: float,
    slot: Any,
    label: str,
    **kwargs: Any,
) -> Any:
    """Core structured generation with retry and multi-strategy recovery.

    Both ``cloud_generation`` and ``local_generation`` delegate here.
    The ``slot`` parameter is a context-manager factory for concurrency
    control (``cloud_llm_slot`` or ``local_llm_slot``).
    """
    logger.debug(f"Generating response with {label} model: {model}")

    target = resolve_chat_target(model)

    parallel_response_model = _parallel_tools_response_model(response_model)
    use_parallel_tools = parallel_response_model is not None
    mode = _resolve_instructor_mode(target, parallel=use_parallel_tools)
    client = _create_instructor_client(target, mode)
    request_response_model = parallel_response_model or response_model
    logger.debug(
        "%s_generation setup: model=%s mode=%s max_tokens=%s temperature=%s",
        label,
        model,
        mode.value,
        max_tokens,
        temperature,
    )

    max_retries = MAX_RETRIES
    base_delay = BASE_DELAY

    request_kwargs = dict(kwargs)
    if target.sdk != "anthropic":
        request_kwargs.setdefault("parallel_tool_calls", bool(use_parallel_tools))

    msg_kwargs = _prepare_messages_for_target(messages, target)

    for attempt in range(max_retries + 1):
        try:
            with slot():
                response = client.chat.completions.create(
                    model=target.api_model,
                    **_sampling_kwargs(target, max_tokens, temperature),
                    response_model=request_response_model,
                    **msg_kwargs,
                    **request_kwargs,
                )
            logger.debug(f"Successfully generated response with {model}")
            return _normalize_generation_response(
                response, parallel_tools=use_parallel_tools
            )
        except Exception as e:
            error_str = str(e)
            tools_mode_due_to_parallel_failure = False

            # PARALLEL_TOOLS can fail when a provider returns no tool calls.
            if (
                use_parallel_tools
                and "nonetype" in error_str.lower()
                and "not iterable" in error_str.lower()
            ):
                logger.warning(
                    "Parallel-tools %s response had no tool calls; "
                    "retrying with TOOLS mode",
                    label,
                )
                try:
                    fallback_mode = _resolve_instructor_mode(target, parallel=False)
                    fallback_client = _create_instructor_client(target, fallback_mode)
                    fallback_kwargs = dict(request_kwargs)
                    if target.sdk != "anthropic":
                        fallback_kwargs["parallel_tool_calls"] = False
                    fallback_kwargs["max_retries"] = 0
                    with slot():
                        response = fallback_client.chat.completions.create(
                            model=target.api_model,
                            **_sampling_kwargs(target, max_tokens, temperature),
                            response_model=response_model,
                            **msg_kwargs,
                            **fallback_kwargs,
                        )
                    logger.debug(
                        "Recovered %s parallel-tools empty response with TOOLS mode",
                        label,
                    )
                    return response
                except Exception as fallback_e:
                    e = fallback_e
                    error_str = str(fallback_e)
                    tools_mode_due_to_parallel_failure = True

            # Handle Instructor multiple tool calls error
            if "multiple tool calls" in error_str.lower():
                # Instructor raises this whenever there isn't exactly one tool
                # call, including zero (the model answered in plain text).
                logger.warning(
                    "%s model %s returned %s tool calls instead of one, "
                    "attempting recovery strategies",
                    label.title(),
                    model,
                    _count_tool_calls(e),
                )

                # Strategy 0: recover directly from exception's last_completion.
                try:
                    recovered = _recover_multiple_tool_calls(
                        error=e,
                        response_model=response_model,
                    )
                    if recovered is not None:
                        return recovered
                except Exception as recovery_e0:
                    logger.warning(
                        "%s direct recovery failed: %s", label.title(), recovery_e0
                    )

                # Strategy 1: force non-parallel with deterministic settings.
                try:
                    logger.debug(
                        "Trying %s with temperature=0 and max_retries=0", label
                    )
                    retry_client = client
                    retry_response_model = request_response_model
                    retry_parallel_tools = bool(use_parallel_tools)
                    if tools_mode_due_to_parallel_failure:
                        retry_mode = _resolve_instructor_mode(target, parallel=False)
                        retry_client = _create_instructor_client(target, retry_mode)
                        retry_response_model = response_model
                        retry_parallel_tools = False
                        logger.warning(
                            "%s strategy 1 switching to TOOLS mode after "
                            "parallel-tools fallback failure",
                            label.title(),
                        )
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs["max_retries"] = 0
                    if target.sdk != "anthropic":
                        retry_kwargs["parallel_tool_calls"] = retry_parallel_tools
                    with slot():
                        response = retry_client.chat.completions.create(
                            model=target.api_model,
                            **_sampling_kwargs(target, max_tokens, 0),
                            response_model=retry_response_model,
                            **msg_kwargs,
                            **retry_kwargs,
                        )
                    logger.debug("Recovered %s with strategy 1", label)
                    return _normalize_generation_response(
                        response, parallel_tools=retry_parallel_tools
                    )
                except Exception as recovery_e1:
                    logger.warning(
                        "%s strategy 1 failed (%s): %s",
                        label.title(),
                        type(recovery_e1).__name__,
                        recovery_e1,
                    )

                # Strategy 2: fall back to JSON mode (no tool calls).
                try:
                    logger.debug("Trying %s JSON mode fallback", label)
                    json_mode = _resolve_instructor_mode(
                        target, parallel=False, json_fallback=True
                    )
                    json_client = _create_instructor_client(target, json_mode)
                    modified_messages = messages.copy()
                    if modified_messages and modified_messages[0]["role"] == "system":
                        modified_messages[0] = dict(modified_messages[0])
                        modified_messages[0]["content"] += (
                            "\n\nIMPORTANT: Return exactly one JSON response."
                        )

                    json_msg_kwargs = _prepare_messages_for_target(
                        modified_messages, target
                    )
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs["max_retries"] = 0
                    if target.sdk != "anthropic":
                        retry_kwargs["parallel_tool_calls"] = False
                    fallback_max_tokens = max(max_tokens, 16384)
                    with slot():
                        response = json_client.chat.completions.create(
                            model=target.api_model,
                            **_sampling_kwargs(target, fallback_max_tokens, 0),
                            response_model=response_model,
                            **json_msg_kwargs,
                            **retry_kwargs,
                        )
                    logger.debug("Recovered %s with strategy 2", label)
                    return response
                except Exception as recovery_e2:
                    logger.warning(
                        "%s strategy 2 failed: %s", label.title(), recovery_e2
                    )
                    logger.error(
                        "All %s recovery strategies failed, will raise original error",
                        label,
                    )

            if _is_retryable(error_str) and attempt < max_retries:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Retryable %s error on attempt %d/%d: %s",
                    label,
                    attempt + 1,
                    max_retries + 1,
                    error_str,
                )
                logger.debug(f"Retrying {label} in {delay:.2f} seconds...")
                time.sleep(delay)
                continue

            logger.error(f"{label.title()} generation failed: {error_str}")
            raise


# ---------------------------------------------------------------------------
# Public generation API
# ---------------------------------------------------------------------------


def cloud_generation(
    messages: List[Dict[str, Any]],
    response_model: Any,
    model: str = CLOUD_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Generate a structured response using a cloud model.

    Routes to the correct SDK (OpenAI-compatible or Anthropic) based on the
    model prefix.  Supports Gemini, OpenAI, Anthropic, OpenRouter, etc.
    """
    return _structured_generation(
        messages,
        response_model,
        model,
        max_tokens,
        temperature,
        slot=cloud_llm_slot,
        label="cloud",
        **kwargs,
    )


def local_generation(
    messages: List[Dict[str, Any]],
    response_model: Any,
    model: str = OLLAMA_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Generate a structured response using a local Ollama model.

    Uses the OpenAI SDK pointed at the Ollama-compatible endpoint.
    """
    return _structured_generation(
        messages,
        response_model,
        model,
        max_tokens,
        temperature,
        slot=local_llm_slot,
        label="local",
        **kwargs,
    )


def text_generation(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    local: bool,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Generate a plain-text reply (no tool call or JSON schema).

    For prompts that want the model to write free text, such as reasoning,
    before a marked answer that the caller parses itself.
    """
    target = resolve_chat_target(model)
    client = _create_raw_client(target)
    slot = local_llm_slot if local else cloud_llm_slot
    msg_kwargs = _prepare_messages_for_target(messages, target)
    sampling = _sampling_kwargs(target, max_tokens, temperature)
    for attempt in range(MAX_RETRIES + 1):
        try:
            with slot():
                if target.sdk == "anthropic":
                    response = client.messages.create(
                        model=target.api_model, **sampling, **msg_kwargs
                    )
                    return "".join(
                        block.text
                        for block in response.content
                        if getattr(block, "type", "") == "text"
                    )
                response = client.chat.completions.create(
                    model=target.api_model, **sampling, **msg_kwargs
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            if _is_retryable(str(e)) and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "Retryable text generation error on attempt %d/%d: %s",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    e,
                )
                time.sleep(delay)
                continue
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Reflection & iterative improvement
# ---------------------------------------------------------------------------


@timed("reflect")
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
    max_tokens: int = DEFAULT_MAX_TOKENS,
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
        max_tokens: Output token limit for each regeneration

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

        generate = (
            cloud_generation if mode == GenerationMode.CLOUD else local_generation
        )
        default_model = CLOUD_MODEL if mode == GenerationMode.CLOUD else OLLAMA_MODEL
        with timed_step("regenerate"):
            improved = generate(
                messages=improvement_messages,
                response_model=response_model,
                model=model or default_model,
                max_tokens=max_tokens,
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
        metadata: Optional metadata (kept for caller compatibility, not sent to SDK)

    Returns:
        List of message dictionaries
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    return messages
