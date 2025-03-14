"""
Utility functions for the hinbox project.
"""

import os
from enum import Enum
from typing import Any, Dict

import instructor
import litellm
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
from pydantic import BaseModel

from src.constants import (
    CLOUD_MODEL,
    EVENTS_OUTPUT_PATH,
    LOCATIONS_OUTPUT_PATH,
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    ORGANIZATIONS_OUTPUT_PATH,
    PEOPLE_OUTPUT_PATH,
    get_ollama_model_name,
)

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


def write_entity_to_file(
    entity_type: str, entity_key: Any, entity_data: Dict[str, Any]
):
    """
    Write a single entity to its respective Parquet file. This function now uses
    an append/update approach for simplicity:
      - reads existing data from the file
      - updates or appends the entity
      - writes it all back

    Args:
        entity_type: "people", "events", "locations", or "organizations"
        entity_key: Key to identify entity (name for people, tuple for others)
        entity_data: Entity data to write
    """
    if entity_type == "people":
        output_path = PEOPLE_OUTPUT_PATH
    elif entity_type == "events":
        output_path = EVENTS_OUTPUT_PATH
    elif entity_type == "locations":
        output_path = LOCATIONS_OUTPUT_PATH
    elif entity_type == "organizations":
        output_path = ORGANIZATIONS_OUTPUT_PATH
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

    entities = []
    entity_found = False

    if os.path.exists(output_path):
        table = pq.read_table(output_path)
        entities = table.to_pylist()

        for i, entity in enumerate(entities):
            if entity_type == "people" and entity.get("name") == entity_key:
                entities[i] = entity_data  # Update existing entity
                entity_found = True
                break
            elif (
                entity_type == "events"
                and (entity.get("title"), entity.get("start_date", "")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break
            elif (
                entity_type in ["locations", "organizations"]
                and (entity.get("name"), entity.get("type", "")) == entity_key
            ):
                entities[i] = entity_data
                entity_found = True
                break

    if not entity_found:
        entities.append(entity_data)  # Append new entity

    # Sort entities appropriately
    if entity_type == "people":
        entities.sort(key=lambda x: x["name"])
    elif entity_type == "events":
        entities.sort(key=lambda x: (x.get("start_date", ""), x.get("title", "")))
    elif entity_type in ["locations", "organizations"]:
        entities.sort(key=lambda x: (x.get("name", ""), x.get("type", "")))

    # Convert to PyArrow table and write to Parquet
    table = pa.Table.from_pylist(entities)
    pq.write_table(table, output_path)


class ReflectionResult(BaseModel):
    result: bool
    reason: str
    feedback_for_improvement: str


class GenerationMode(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"


def cloud_generation(
    system_prompt: str,
    user_prompt: str,
    response_model: BaseModel,
    model: str = CLOUD_MODEL,
    temperature: int = 0,
    metadata: Dict[str, Any] = None,
):
    """Generate a response using the cloud model.

    Args:
        system_prompt: The system prompt to use for the generation
        user_prompt: The user prompt to use for the generation
        response_model: The response model to use for the generation
        model: The model to use for the generation
        temperature: The temperature to use for the generation
        metadata: The metadata to use for the generation

    Returns:
        The generated response
    """
    client = instructor.from_litellm(litellm.completion)

    if metadata is None:
        metadata = {
            "project_name": "hinbox",
            "tags": ["dev"],
        }

    try:
        response = client.chat.completions.create(
            model=model,
            response_model=response_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            metadata=metadata,
        )
    except Exception as e:
        raise e
    return response


def local_generation(
    system_prompt: str,
    user_prompt: str,
    response_model: BaseModel,
    model: str = OLLAMA_MODEL,
    temperature: int = 0,
    metadata: Dict[str, Any] = None,
) -> str:
    """Generate a response using the local model.

    Args:
        system_prompt: The system prompt to use for the generation
        user_prompt: The user prompt to use for the generation
        response_model: The response model to use for the generation
        model: The model to use for the generation
        temperature: The temperature to use for the generation
        metadata: The metadata to use for the generation

    Returns:
        The generated response
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    if metadata is None:
        metadata = {
            "project_name": "hinbox",
            "tags": ["dev"],
        }

    try:
        response = client.beta.chat.completions.parse(
            model=get_ollama_model_name(model),
            response_format=response_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            metadata=metadata,
        )
    except Exception as e:
        raise e
    return response


def reflect_and_check(
    input_prompt: str,
    output_json_str: str,
    generation_mode: GenerationMode = GenerationMode.CLOUD,
) -> ReflectionResult:
    """Check if the output JSON string is a valid response given the input
    prompt.

    Args:
        input_prompt: The prompt to reflect on
        output_json_str: The output JSON string to check

    Returns:
        ReflectionResult: The result of the reflection
    """
    if generation_mode == GenerationMode.CLOUD:
        output_json_str = cloud_generation(input_prompt)
    elif generation_mode == GenerationMode.LOCAL:
        output_json_str = local_generation(input_prompt)

    return ReflectionResult(result=False, reason="", feedback_for_improvement="")


if __name__ == "__main__":
    print("Checking reflection and checking...")
