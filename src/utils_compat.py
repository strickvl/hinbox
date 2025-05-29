"""
Backward compatibility module for utils.

All functionality has been moved to specific submodules in src/utils/.
This file provides imports and compatibility wrappers for backward compatibility.
"""

from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

# Import constants we might need
from src.constants import CLOUD_MODEL, OLLAMA_MODEL

# Import from new locations for backward compatibility
from src.utils.file_ops import sanitize_for_parquet, write_entity_to_file
from src.utils.llm import (
    GenerationMode,
    ReflectionResult,
    cloud_generation,
    extract_reflection_result,
    local_generation,
    reflect_and_check,
)
from src.utils.llm import (
    iterative_improve as _new_iterative_improve,
)
from src.utils.profiles import extract_profile_text


# Compatibility wrapper for the old iterative_improve signature
def iterative_improve(
    prompt: str,
    response_model: Type[BaseModel],
    generation_mode: GenerationMode = GenerationMode.CLOUD,
    model: str = None,
    evaluation_model: str = None,
    max_iterations: int = 3,
    temperature: int = 0,
    evaluation_temperature: int = 0,
    metadata: Dict[str, Any] = None,
    system_prompt: str = None,
) -> Tuple[BaseModel, List[Dict[str, Any]]]:
    """
    Compatibility wrapper for the old iterative_improve signature.

    This adapts the old interface to the new iterative_improve function
    which has a different signature.
    """
    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = """You are an expert assistant tasked with fulfilling the user's request precisely.
Provide a high-quality response that addresses all requirements in the prompt."""

    # Construct messages for the new interface
    generation_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Construct reflection prompt
    reflection_prompt = """Check if this output meets all requirements including:
- Proper use of inline footnotes (like ^[article_id]) if requested
- A sufficiently detailed 'text' section (not just a short phrase)
- Inclusion of confidence scores, tags, or any other mandated fields

If any requirement is missing, mark result=false and provide feedback for improvement."""

    # Determine model if not specified
    if model is None:
        model = CLOUD_MODEL if generation_mode == GenerationMode.CLOUD else OLLAMA_MODEL

    # For the old interface, we need to generate initial content first
    # The new iterative_improve expects to start with some text
    # So we'll do an initial generation, then use iterative_improve to refine it

    from src.utils.llm import cloud_generation, local_generation

    # Generate initial response
    if generation_mode == GenerationMode.CLOUD:
        initial_response = cloud_generation(
            messages=generation_messages,
            response_model=response_model,
            model=model or CLOUD_MODEL,
            temperature=temperature,
        )
    else:
        initial_response = local_generation(
            messages=generation_messages,
            response_model=response_model,
            model=model or OLLAMA_MODEL,
            temperature=temperature,
        )

    # Extract initial text from response
    if hasattr(initial_response, "model_dump_json"):
        initial_text = initial_response.model_dump_json()
    elif hasattr(initial_response, "json"):
        initial_text = initial_response.json()
    else:
        initial_text = str(initial_response)

    # Now use iterative_improve to refine it
    result = _new_iterative_improve(
        initial_text=initial_text,
        generation_messages=generation_messages,
        reflection_prompt=reflection_prompt,
        response_model=response_model,
        max_iterations=max_iterations
        - 1,  # Subtract 1 since we already did initial generation
        mode=generation_mode,
        model=model,
    )

    # Extract the final result from the dictionary
    final_text = result.get("text", "")
    reflection_history = result.get("reflection_history", [])

    # Parse the final result as the response model
    import json

    try:
        # Try to parse the text as JSON and create the model
        if isinstance(final_text, str) and final_text.strip():
            # If it's a JSON string, parse it
            if final_text.strip().startswith("{"):
                data = json.loads(final_text)
                final_result = response_model(**data)
            else:
                # If it's just text, wrap it in the expected structure
                final_result = response_model(text=final_text, tags=[], confidence=0.8)
        else:
            final_result = None
    except:
        # If parsing fails, try to create a minimal valid response
        try:
            final_result = response_model(
                text=final_text or "Failed to generate", tags=[], confidence=0.0
            )
        except:
            final_result = None

    # Convert reflection history to match old format
    # Add the initial generation as the first entry
    history = [
        {
            "iteration": 0,
            "response": initial_response,
            "passed": False,  # Assume initial generation needs improvement
            "reason": "Initial generation",
            "feedback": "Initial response generated, checking for improvements",
        }
    ]

    # Add the rest of the reflection history
    for i, entry in enumerate(reflection_history):
        history.append(
            {
                "iteration": i + 1,
                "response": final_result
                if i == len(reflection_history) - 1
                else initial_response,
                "passed": entry.get("valid", False),
                "reason": entry.get("reasoning", ""),
                "feedback": str(entry.get("issues", "")),  # Ensure it's a string
            }
        )

    # If we have a valid final result from iterative improvement, use it
    # Otherwise, use the initial response
    if not final_result and initial_response:
        final_result = initial_response

    return final_result, history


__all__ = [
    "sanitize_for_parquet",
    "write_entity_to_file",
    "GenerationMode",
    "ReflectionResult",
    "cloud_generation",
    "extract_reflection_result",
    "iterative_improve",
    "local_generation",
    "reflect_and_check",
    "extract_profile_text",
]
