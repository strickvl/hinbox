"""Generic entity extraction utilities."""

from typing import Any, List, Type, Union

from pydantic import BaseModel

from src.constants import (
    BRAINTRUST_PROJECT_ID,
    BRAINTRUST_PROJECT_NAME,
    CLOUD_MODEL,
    OLLAMA_MODEL,
)
from src.utils.llm import (
    create_messages,
    get_litellm_client,
    get_ollama_client,
)


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

    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        temperature=temperature,
        messages=messages,
        metadata=metadata,
    )


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
