"""
Utility functions for the hinbox project.
"""

import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

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
from src.logging_config import get_logger
from src.utils_logging import (
    log_evaluation_result,
    log_evaluation_start,
    log_generation_attempt,
    log_generation_result,
    log_iterative_complete,
    log_iterative_start,
)

# Get logger for this module
logger = get_logger("utils")

litellm.enable_json_schema_validation = True
litellm.suppress_debug_info = True
litellm.callbacks = ["braintrust"]


def extract_profile_text(
    profile_response: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Extract profile text from either a simple dict or nested API response.

    Args:
        profile_response: Response from create_profile() which could be either:
            - A simple dict with 'text' key
            - A nested API response with parsed content

    Returns:
        Dict with 'text' and other fields, or None if text cannot be extracted
    """
    if not profile_response:
        return None

    # Handle simple dict case
    if isinstance(profile_response, dict):
        if "text" in profile_response:
            return profile_response
        if "choices" in profile_response and len(profile_response["choices"]) > 0:
            message = profile_response["choices"][0].get("message", {})
            if "parsed" in message:
                return message["parsed"]

    return None


def sanitize_for_parquet(entity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively remove or transform fields that are incompatible with Arrow from
    an entity dictionary. Specifically, we remove/serialize any LLM response objects
    in reflection_history.
    """
    # Create a shallow copy so we don't mutate the original
    sanitized = dict(entity)

    # If there's a reflection_history field, strip or convert the "response" object
    if "reflection_history" in sanitized and isinstance(
        sanitized["reflection_history"], list
    ):
        for iteration in sanitized["reflection_history"]:
            # Convert iteration to a dict if needed
            if isinstance(iteration, dict):
                # If there's a 'response' field that's not arrow-compatible, replace it or string-ify it
                if "response" in iteration:
                    # We'll store a string representation or just clear it out
                    iteration["response"] = str(iteration["response"])

    # Return the sanitized dict
    return sanitized


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

    # Sanitize incoming entity data to avoid storing objects that PyArrow can't handle
    entity_data = sanitize_for_parquet(entity_data)

    entities = []
    entity_found = False

    if os.path.exists(output_path):
        table = pq.read_table(output_path)
        entities = table.to_pylist()

        # Sanitize existing entities from file
        for i, existing_entity in enumerate(entities):
            existing_entity = sanitize_for_parquet(existing_entity)
            entities[i] = existing_entity

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
    """Result of an LLM's reflection on its own output."""

    result: bool  # Whether the output meets the requirements
    reason: str  # Reason for the decision
    feedback_for_improvement: str  # Feedback for improvement if result is False


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
    model: str = None,
    temperature: int = 0,
    metadata: Dict[str, Any] = None,
) -> ReflectionResult:
    """Check if the output JSON string is a valid response given the input
    prompt.

    Args:
        input_prompt: The original prompt given to the model
        output_json_str: The output JSON string to evaluate
        generation_mode: Whether to use cloud or local model for evaluation
        model: Model to use for evaluation (uses default if None)
        temperature: Temperature for the evaluation model
        metadata: Metadata for the evaluation request

    Returns:
        ReflectionResult: The result of the reflection with feedback
    """
    # Create evaluation system prompt
    system_prompt = """
    You are an expert evaluator of AI-generated content. Your task is to evaluate 
    if the provided output properly fulfills the requirements specified in the original prompt.
    
    You should carefully analyze:
    1. Whether all requirements in the prompt were addressed
    2. The quality and correctness of the output
    3. If there are any errors, omissions, or misunderstandings
    4. Whether the text includes proper inline citations (e.g. ^[article_id]) if the prompt requires footnotes
    5. Whether the profile text is sufficiently detailed (e.g., at least a few sentences long)
    6. Verify that the output includes a 'text' field of at least 50 characters. If not, mark the result as false.
    
    If the output is missing footnotes when required, is too short, or ignores key instructions, you must fail it.
    
    Provide honest, critical feedback for improvement when necessary.
    """

    # Construct evaluation user prompt with original prompt and output
    evaluation_prompt = f"""
    Original Prompt:
    ```
    {input_prompt}
    ```
    
    Output to Evaluate:
    ```
    {output_json_str}
    ```
    
    Please evaluate if this output meets all requirements, including:
    - Proper use of inline footnotes (like ^[article_id]) if requested
    - A sufficiently detailed 'text' section (not just a short phrase)
    - Inclusion of confidence scores, tags, or any other mandated fields
    
    If any requirement is missing, mark result=false and provide feedback for improvement.
    """

    # Use appropriate generation function based on mode
    if model is None:
        model = CLOUD_MODEL if generation_mode == GenerationMode.CLOUD else OLLAMA_MODEL

    if generation_mode == GenerationMode.CLOUD:
        result = cloud_generation(
            system_prompt=system_prompt,
            user_prompt=evaluation_prompt,
            response_model=ReflectionResult,
            model=model,
            temperature=temperature,
            metadata=metadata,
        )
    else:
        result = local_generation(
            system_prompt=system_prompt,
            user_prompt=evaluation_prompt,
            response_model=ReflectionResult,
            model=model,
            temperature=temperature,
            metadata=metadata,
        )

    # Now parse the reflection result:
    reflection_result = extract_reflection_result(result)

    # Additional strict validation: check for a 'text' field in the user's raw output
    try:
        data = json.loads(output_json_str)
        # Only enforce if it's a dict with a potential 'text' field
        if isinstance(data, dict):
            text_val = data.get("text", "")
            if len(text_val) < 50:
                reflection_result.result = False
                reflection_result.reason = (
                    f"Profile text is missing or too short (len={len(text_val)})"
                )
                reflection_result.feedback_for_improvement = (
                    "Please include a 'text' field with at least 50 characters."
                )
        else:
            reflection_result.result = False
            reflection_result.reason = "Output is not a JSON object."
            reflection_result.feedback_for_improvement = (
                "Return valid JSON with a top-level 'text' field of sufficient length."
            )
    except Exception as e:
        reflection_result.result = False
        reflection_result.reason = f"Error parsing JSON: {str(e)}"
        reflection_result.feedback_for_improvement = (
            "Provide valid JSON with a 'text' field of adequate length."
        )

    return reflection_result


def extract_reflection_result(response_obj) -> ReflectionResult:
    """
    Extract the ReflectionResult from different response object formats.

    This is a helper function to handle the different formats that might be returned
    by different models or APIs.

    Args:
        response_obj: The response object from the LLM

    Returns:
        ReflectionResult containing the extracted values
    """
    # If it's already a ReflectionResult, return it directly
    if isinstance(response_obj, ReflectionResult):
        return response_obj

    # If it's a ParsedChatCompletion with the value property
    if hasattr(response_obj, "value") and isinstance(
        response_obj.value, ReflectionResult
    ):
        return response_obj.value

    # For local models using OpenAI client, usually the response is a dict-like object
    # with choices containing the message and content
    try:
        import json

        # Create a default ReflectionResult
        result = ReflectionResult(
            result=False,
            reason="Could not determine if requirements are met",
            feedback_for_improvement="Try to match the requirements more closely",
        )

        # Try different ways to extract the reflection result
        if hasattr(response_obj, "choices") and len(response_obj.choices) > 0:
            message = response_obj.choices[0].message

            # Check if we have a parsed property (Ollama's response format)
            if hasattr(message, "parsed") and message.parsed is not None:
                # Extract the reflection result from the parsed property
                if isinstance(message.parsed, dict):
                    result.result = message.parsed.get("result", False)
                    result.reason = message.parsed.get("reason", "")
                    result.feedback_for_improvement = message.parsed.get(
                        "feedback_for_improvement", ""
                    )
                elif hasattr(message.parsed, "result"):
                    result.result = message.parsed.result
                    result.reason = message.parsed.reason
                    result.feedback_for_improvement = (
                        message.parsed.feedback_for_improvement
                    )
                return result

            # Check if we have content property
            if hasattr(message, "content") and message.content:
                content = message.content

                # Try to parse it as JSON if it's a string
                if isinstance(content, str):
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            result.result = data.get("result", False)
                            result.reason = data.get("reason", "")
                            result.feedback_for_improvement = data.get(
                                "feedback_for_improvement", ""
                            )
                            return result
                    except json.JSONDecodeError:
                        # If it's not valid JSON, but contains key phrases, try to extract them
                        if "PASS" in content.upper():
                            result.result = True
                            result.reason = "Requirements appear to be met"
                            result.feedback_for_improvement = ""
                        elif (
                            "FAIL" in content.upper()
                            or "NEEDS_IMPROVEMENT" in content.upper()
                        ):
                            result.result = False
                            # Try to extract feedback from the content
                            import re

                            feedback_match = re.search(
                                r"(?:feedback|improvement):\s*(.*?)(?:\n\n|\Z)",
                                content,
                                re.IGNORECASE | re.DOTALL,
                            )
                            if feedback_match:
                                result.feedback_for_improvement = feedback_match.group(
                                    1
                                ).strip()
                            reason_match = re.search(
                                r"(?:reason|evaluation):\s*(.*?)(?:\n\n|\Z)",
                                content,
                                re.IGNORECASE | re.DOTALL,
                            )
                            if reason_match:
                                result.reason = reason_match.group(1).strip()
                        return result

        # If we haven't returned yet, try other methods
        if hasattr(response_obj, "model_dump"):
            data = response_obj.model_dump()
            if "result" in data:
                result.result = data.get("result", False)
                result.reason = data.get("reason", "")
                result.feedback_for_improvement = data.get(
                    "feedback_for_improvement", ""
                )

        return result
    except Exception as e:
        # If all else fails, return a default ReflectionResult
        return ReflectionResult(
            result=False,
            reason=f"Error extracting reflection result: {str(e)}",
            feedback_for_improvement="Could not properly evaluate the response.",
        )


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
    Iteratively improve an LLM response until it meets the requirements or
    reaches the maximum number of iterations.

    Args:
        prompt: The user prompt to generate a response for
        response_model: The Pydantic model to validate the response against
        generation_mode: Whether to use cloud or local models
        model: The model to use for generation (uses default if None)
        evaluation_model: The model to use for evaluation (uses generation model if None)
        max_iterations: Maximum number of improvement iterations
        temperature: Temperature for the generation model
        evaluation_temperature: Temperature for the evaluation model
        metadata: Metadata for the requests

    Returns:
        Tuple containing:
        - The final response as a Pydantic model
        - List of improvement iterations with original responses and feedback
    """
    if model is None:
        model = CLOUD_MODEL if generation_mode == GenerationMode.CLOUD else OLLAMA_MODEL

    if evaluation_model is None:
        evaluation_model = model

    if metadata is None:
        metadata = {
            "project_name": "hinbox",
            "tags": ["iterative_improvement"],
        }

    # Start timing the overall process
    process_start_time = log_iterative_start(prompt, model, max_iterations)
    history = []

    # Default system prompts for the generator
    default_system_prompt = """
    You are an expert assistant tasked with fulfilling the user's request precisely.
    Provide a high-quality response that addresses all requirements in the prompt.
    """

    # For subsequent iterations, we enhance the system prompt with feedback
    default_feedback_system_prompt = """
    You are an expert assistant tasked with fulfilling the user's request precisely.
    Your previous response needed improvement. Please address the feedback and provide
    a better response that fully satisfies all requirements.
    """

    # Use the custom system prompt if provided, otherwise fall back
    if system_prompt is None:
        system_prompt = default_system_prompt

    feedback_system_prompt = default_feedback_system_prompt

    # Initial generation
    generation_fn = (
        cloud_generation
        if generation_mode == GenerationMode.CLOUD
        else local_generation
    )
    current_system_prompt = system_prompt

    for iteration in range(max_iterations):
        # Start timing this generation attempt
        generation_start_time = log_generation_attempt(iteration, model)

        # Generate response
        try:
            response = generation_fn(
                system_prompt=current_system_prompt,
                user_prompt=prompt,
                response_model=response_model,
                model=model,
                temperature=temperature,
                metadata={**metadata, "iteration": iteration},
            )
            logger.debug(
                f"Raw LLM response object (iteration {iteration + 1}): {response}"
            )
            log_generation_result(generation_start_time, iteration, True)
        except Exception as e:
            # Log generation failure
            log_generation_result(generation_start_time, iteration, False, str(e))

            history.append(
                {
                    "iteration": iteration,
                    "response": None,
                    "error": str(e),
                    "passed": False,
                    "feedback": f"Error during generation: {str(e)}",
                }
            )
            continue

        # Convert to string representation for evaluation
        if hasattr(response, "model_dump_json"):
            response_str = response.model_dump_json()
        elif hasattr(response, "json"):
            response_str = response.json()
        else:
            # Fallback for other response types
            response_str = str(response)

        logger.debug(
            f"LLM response (string) for iteration {iteration + 1}: {response_str}"
        )

        # STAGE 3: Optional post-validation before calling reflect_and_check
        # STAGE 3: Optional post-validation before calling reflect_and_check
        # Attempt to parse response_str as JSON. If invalid or too short 'text', we forcibly fail this iteration.
        import json

        try:
            data = json.loads(response_str)  # parse the top-level JSON
            text_val = ""

            # If it has top-level "text", read that, else look deeper in choices->message->parsed->text
            if isinstance(data, dict):
                # 1) check if top-level 'text' exists
                if "text" in data and isinstance(data["text"], str):
                    text_val = data["text"]
                else:
                    # 2) if not, check in data["choices"][0]["message"]["parsed"]["text"]
                    choices = data.get("choices", [])
                    if len(choices) > 0:
                        first_choice = choices[0]
                        message = first_choice.get("message", {})
                        parsed = message.get("parsed", {})
                        if isinstance(parsed, dict):
                            # finally get the text
                            if "text" in parsed and isinstance(parsed["text"], str):
                                text_val = parsed["text"]

            logger.debug(f"Local post-validation: text length = {len(text_val)}")

            if len(text_val) < 50:
                logger.debug(
                    "Forcing iteration fail because no valid text returned or text too short."
                )
                history.append(
                    {
                        "iteration": iteration,
                        "response": response,
                        "passed": False,
                        "reason": "Locally forced fail (text < 50)",
                        "feedback": "Add more detail to the 'text' field; at least 50 characters.",
                    }
                )
                continue

        except Exception as local_e:
            logger.debug(f"JSON parse error in local post-validation: {local_e}")
            logger.debug("Forcing iteration fail because invalid JSON returned.")
            history.append(
                {
                    "iteration": iteration,
                    "response": response,
                    "passed": False,
                    "reason": f"Locally forced fail (invalid JSON: {str(local_e)})",
                    "feedback": "Return valid JSON with a 'text' field of >= 50 chars.",
                }
            )
            continue

        # Start timing the evaluation
        evaluation_start_time = log_evaluation_start(iteration, evaluation_model)

        # Evaluate the response
        evaluation = reflect_and_check(
            input_prompt=prompt,
            output_json_str=response_str,
            generation_mode=generation_mode,
            model=evaluation_model,
            temperature=evaluation_temperature,
            metadata={**metadata, "iteration": iteration, "evaluation": True},
        )

        # Extract the reflection result
        reflection = extract_reflection_result(evaluation)

        # Log evaluation result
        log_evaluation_result(
            evaluation_start_time,
            iteration,
            reflection.result,
            reflection.reason,
            reflection.feedback_for_improvement,
        )

        # Record the iteration
        history.append(
            {
                "iteration": iteration,
                "response": response,
                "passed": reflection.result,
                "reason": reflection.reason,
                "feedback": reflection.feedback_for_improvement,
            }
        )

        # If the response passes evaluation, we're done
        if reflection.result:
            break

        # Otherwise, update the system prompt with feedback for the next iteration
        current_system_prompt = f"""
        {feedback_system_prompt}
        
        Previous response:
        ```
        {response_str}
        ```
        
        Feedback for improvement:
        {reflection.feedback_for_improvement}
        
        Please provide an improved response.
        """

    # Log completion of the iterative process
    iterations_completed = len(history)
    success = any(entry.get("passed", False) for entry in history)
    log_iterative_complete(
        process_start_time, iterations_completed, max_iterations, success
    )

    # Return the best response (the last one that passed, or the final attempt)
    for entry in reversed(history):
        if entry.get("passed", False) and entry.get("response") is not None:
            return entry["response"], history

    # If no successful generations, return the last attempt
    return history[-1]["response"] if history and history[-1].get(
        "response"
    ) is not None else None, history


if __name__ == "__main__":
    print("Testing iterative improvement pattern...")

    # Example model for testing
    class WeatherForecast(BaseModel):
        location: str
        temperature: float
        conditions: str
        forecast_days: List[Dict[str, str]]

    # Example prompt
    test_prompt = """
    Provide a detailed 3-day weather forecast for London, including daily conditions
    and temperature in Celsius. Include a brief description for each day.
    """

    # Test the iterative improvement
    try:
        final_result, improvement_history = iterative_improve(
            prompt=test_prompt,
            response_model=WeatherForecast,
            generation_mode=GenerationMode.LOCAL,  # Using local models for testing
            max_iterations=3,
            temperature=0.2,
        )

        # Print the improvement history
        print(f"Total iterations: {len(improvement_history)}")
        for i, entry in enumerate(improvement_history):
            print(f"\n--- Iteration {i + 1} ---")
            print(f"Passed: {entry.get('passed', False)}")
            if not entry.get("passed", False):
                print(f"Feedback: {entry.get('feedback', 'No feedback')}")

        # Print final result
        print("\n=== Final Result ===")
        if final_result:
            # Try to extract structured data from the final result
            if hasattr(final_result, "model_dump_json"):
                print(final_result.model_dump_json(indent=2))
            elif hasattr(final_result, "choices") and len(final_result.choices) > 0:
                message = final_result.choices[0].message
                if hasattr(message, "parsed") and message.parsed is not None:
                    # Get the parsed property which contains our structured data
                    parsed_data = message.parsed
                    # Convert it to JSON for display
                    import json

                    if hasattr(parsed_data, "model_dump"):
                        print(json.dumps(parsed_data.model_dump(), indent=2))
                    else:
                        print(json.dumps(parsed_data, indent=2))
                else:
                    print(message.content)
            else:
                print(final_result)
        else:
            print("Failed to generate a satisfactory response.")

        # Print a more human-readable version of the weather forecast
        print("\n=== Weather Forecast Summary ===")
        if final_result:
            try:
                # Extract the parsed data
                data = None
                # Check for Ollama's response format first
                if hasattr(final_result, "choices") and len(final_result.choices) > 0:
                    message = final_result.choices[0].message
                    if hasattr(message, "parsed"):
                        data = message.parsed
                        # For Ollama's response, 'parsed' is the actual data object
                        # and doesn't need to be unwrapped further
                        if hasattr(data, "location"):
                            location = data.location
                            temperature = data.temperature
                            conditions = data.conditions
                            forecast_days = data.forecast_days
                            print(f"Location: {location}")
                            print(f"Current Temperature: {temperature}°C")
                            print(f"Current Conditions: {conditions}")
                            print("\nDaily Forecast:")

                            for day in forecast_days:
                                print(f"\n{day.get('day', 'Unknown day')}")
                                # Check different possible temperature field names
                                if "temperature" in day:
                                    print(
                                        f"Temperature: {day.get('temperature', 'N/A')}"
                                    )
                                elif "high_temp" in day and "low_temp" in day:
                                    print(
                                        f"High: {day.get('high_temp', 'N/A')}, Low: {day.get('low_temp', 'N/A')}"
                                    )
                                elif (
                                    "temperature_high" in day
                                    and "temperature_low" in day
                                ):
                                    print(
                                        f"High: {day.get('temperature_high', 'N/A')}°C, Low: {day.get('temperature_low', 'N/A')}°C"
                                    )
                                print(
                                    f"Description: {day.get('description', 'No description available')}"
                                )
                        else:
                            # Use the fallback for other data formats
                            if isinstance(data, dict):
                                print(f"Location: {data.get('location', 'Unknown')}")
                                print(
                                    f"Current Temperature: {data.get('temperature', 'N/A')}°C"
                                )
                                print(
                                    f"Current Conditions: {data.get('conditions', 'Unknown')}"
                                )
                                print("\nDaily Forecast:")

                                for idx, day in enumerate(
                                    data.get("forecast_days", [])
                                ):
                                    print(f"\n{day.get('day', f'Day {idx + 1}')}")
                                    # Check different possible temperature field names
                                    if "temperature" in day:
                                        print(
                                            f"Temperature: {day.get('temperature', 'N/A')}"
                                        )
                                    elif "high_temp" in day and "low_temp" in day:
                                        print(
                                            f"High: {day.get('high_temp', 'N/A')}, Low: {day.get('low_temp', 'N/A')}"
                                        )
                                    elif (
                                        "temperature_high" in day
                                        and "temperature_low" in day
                                    ):
                                        print(
                                            f"High: {day.get('temperature_high', 'N/A')}°C, Low: {day.get('temperature_low', 'N/A')}°C"
                                        )
                                    print(
                                        f"Description: {day.get('description', 'No description available')}"
                                    )
                            else:
                                print("Data is not in the expected dictionary format")
                # Check for Pydantic model format next
                elif hasattr(final_result, "model_dump"):
                    data = final_result.model_dump()
                    if isinstance(data, dict):
                        print(f"Location: {data.get('location', 'Unknown')}")
                        print(
                            f"Current Temperature: {data.get('temperature', 'N/A')}°C"
                        )
                        print(
                            f"Current Conditions: {data.get('conditions', 'Unknown')}"
                        )
                        print("\nDaily Forecast:")

                        for idx, day in enumerate(data.get("forecast_days", [])):
                            print(f"\n{day.get('day', f'Day {idx + 1}')}")
                            # Check different possible temperature field names
                            if "temperature" in day:
                                print(f"Temperature: {day.get('temperature', 'N/A')}")
                            elif "high_temp" in day and "low_temp" in day:
                                print(
                                    f"High: {day.get('high_temp', 'N/A')}, Low: {day.get('low_temp', 'N/A')}"
                                )
                            elif "temperature_high" in day and "temperature_low" in day:
                                print(
                                    f"High: {day.get('temperature_high', 'N/A')}°C, Low: {day.get('temperature_low', 'N/A')}°C"
                                )
                            print(
                                f"Description: {day.get('description', 'No description available')}"
                            )
                    else:
                        print("Data is not in the expected dictionary format")
                else:
                    print("No structured weather data found in response")
            except Exception as e:
                print(f"Error formatting weather data: {str(e)}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
