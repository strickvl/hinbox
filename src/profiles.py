import logging
from typing import Dict, List

import litellm
from pydantic import BaseModel, Field
from rich.console import Console

from src.constants import (
    BRAINTRUST_PROJECT_ID,
    BRAINTRUST_PROJECT_NAME,
    CLOUD_MODEL,
    OLLAMA_MODEL,
)
from src.utils.llm import (
    GenerationMode,
    cloud_generation,
    iterative_improve,
    local_generation,
)
from src.utils.profiles import extract_profile_text

# Enable JSON schema validation for structured responses
litellm.enable_json_schema_validation = True
litellm.suppress_debug_info = True
litellm.callbacks = ["braintrust", "langfuse"]

logger = logging.getLogger(__name__)
console = Console()


def _build_metadata(operation: str) -> Dict[str, any]:
    """Build metadata dict with Braintrust configuration."""
    metadata = {"tags": [operation]}
    if BRAINTRUST_PROJECT_ID:
        metadata["project_id"] = BRAINTRUST_PROJECT_ID
    elif BRAINTRUST_PROJECT_NAME:
        metadata["project_name"] = BRAINTRUST_PROJECT_NAME
    return metadata


class EntityProfile(BaseModel):
    """Structured profile model for an entity."""

    text: str = Field(..., description="Comprehensive profile text about the entity")
    tags: List[str] = Field(
        default_factory=list,
        description="List of relevant tags/keywords for this entity",
    )
    confidence: float = Field(
        ...,
        description="Confidence score (0-1) in the accuracy of this profile",
        ge=0,
        le=1,
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of article IDs used as sources for this profile",
    )


def generate_profile_with_reflection(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    max_iterations: int = 3,
) -> (Dict, list):
    """
    Generate a profile with reflection and improvement.
    Uses iterative_improve from utils.py.
    Returns a tuple: (final_profile_dict, improvement_history).
    """

    # Enhanced system prompt emphasizing requirements
    custom_system_prompt = f"""You are an expert at creating profiles for entities mentioned in news articles.

Your task is to create a comprehensive profile for a {entity_type} named "{entity_name}" based solely on the provided article text.

The profile MUST:
1. Be organized with clear section headers (e.g., ### Background, ### Role)
2. Include citations for every fact using format: fact^[article_id]
3. For multiple sources use format: fact^[id1, id2]
4. Include relevant tags/keywords
5. Provide a confidence score (0-1)
6. Only include factual information from the source

Example format:
```
John Smith is a military officer^[abc123] who oversees operations at Guantánamo Bay^[abc123, def456].

### Background
* Previously served in Afghanistan^[abc123]
* Extensive experience in detention operations^[abc123, def456]
```
"""

    # Build messages for the new iterative_improve signature
    generation_messages = [
        {"role": "system", "content": custom_system_prompt},
        {
            "role": "user",
            "content": f"Article text:\n\n{article_text}\n\nCreate a profile for {entity_type} '{entity_name}'. Article ID: {article_id}",
        },
    ]

    reflection_prompt = """Check if this profile meets all requirements including:
    - Proper use of inline footnotes (like ^[article_id]) 
    - A sufficiently detailed 'text' section (not just a short phrase)
    - Inclusion of confidence scores, tags, and source references
    
    If any requirement is missing, mark result=false and provide feedback for improvement."""

    # Generate initial profile text for iterative improvement
    mode = GenerationMode.CLOUD if model_type == "gemini" else GenerationMode.LOCAL
    if mode == GenerationMode.CLOUD:
        initial_response = cloud_generation(
            messages=generation_messages,
            response_model=EntityProfile,
            model=CLOUD_MODEL,
            temperature=0,
        )
    else:
        initial_response = local_generation(
            messages=generation_messages,
            response_model=EntityProfile,
            model=OLLAMA_MODEL,
            temperature=0,
        )

    # Convert to text for iterative improvement
    initial_text = initial_response.model_dump_json() if initial_response else "{}"

    # Use iterative_improve to refine the profile
    result = iterative_improve(
        initial_text=initial_text,
        generation_messages=generation_messages,
        reflection_prompt=reflection_prompt,
        response_model=EntityProfile,
        max_iterations=max_iterations,
        mode=mode,
    )

    # Extract final result
    final_text = result.get("text", "{}")
    history = result.get("reflection_history", [])

    # Parse the final result
    try:
        import json

        if final_text.strip():
            if final_text.strip().startswith("{"):
                data = json.loads(final_text)
                final_result = EntityProfile(**data)
            else:
                final_result = EntityProfile(
                    text=final_text, tags=[], confidence=0.8, sources=[article_id]
                )
        else:
            final_result = None
    except:
        final_result = None

    if final_result is None:
        # Fallback to basic profile if all iterations fail
        fallback_profile = {
            "text": f"Profile generation failed for {entity_name}^[{article_id}]",
            "tags": [],
            "confidence": 0.0,
            "sources": [article_id],
        }
        return fallback_profile, history

    result_dict = final_result.model_dump()
    result_dict["sources"] = [article_id]

    return result_dict, history


def generate_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
) -> (Dict, list):
    """
    Generate a profile for an entity based on article text using structured extraction.
    Now uses reflection pattern for validation and improvement.
    Returns (profile_dict, improvement_history).
    """
    profile_dict, improvement_history = generate_profile_with_reflection(
        entity_type=entity_type,
        entity_name=entity_name,
        article_text=article_text,
        article_id=article_id,
        model_type=model_type,
    )
    return profile_dict, improvement_history


def update_profile(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
) -> (Dict, list):
    """
    Update an existing profile with new information from an article using iterative reflection.

    We treat the existing profile text plus the new article text as input for improvement.
    Returns (updated_profile, improvement_history).
    """
    console.print(f"\n[cyan]Updating profile for {entity_type} '{entity_name}'[/cyan]")
    console.print(f"[cyan]Using model: {model_type}[/cyan]")
    console.print(f"[cyan]New article ID: {new_article_id}[/cyan]")
    console.print(
        f"[cyan]Existing profile length: {len(existing_profile.get('text', ''))} characters[/cyan]"
    )
    console.print(
        f"[cyan]New article text length: {len(new_article_text)} characters[/cyan]"
    )

    try:
        # We'll build a combined prompt that includes the existing profile text and the new article text,
        # instructing the LLM to integrate them into a new updated profile with the same formatting rules.
        system_prompt = f"""You are an expert at updating profiles for a {entity_type} named "{entity_name}" with new information from news articles.

The final updated profile must:
1. Incorporate new relevant factual information from the new article.
2. Retain existing valid information from the old profile.
3. Resolve contradictions by preferring newer details.
4. Include inline citations in the format: fact^[id]
5. Provide a confidence score (0-1), relevant tags, and a well-structured, sectioned layout.

Existing Profile (already validated):
----------------
{existing_profile.get("text", "")}

Existing Tags: {existing_profile.get("tags", [])}
Existing Confidence: {existing_profile.get("confidence", 0.0)}
Existing Sources: {existing_profile.get("sources", [])}

New Article (ID: {new_article_id}):
----------------
{new_article_text}
"""

        console.print("[cyan]Starting iterative improvement process...[/cyan]")

        # For the iterative process, we treat the entire text above as the 'prompt' so the LLM merges them.
        # Import already at module level

        generation_mode = (
            GenerationMode.CLOUD if model_type == "gemini" else GenerationMode.LOCAL
        )

        # We'll use the same EntityProfile structure for the updated profile.
        final_result, history = iterative_improve(
            prompt=system_prompt,
            response_model=EntityProfile,
            generation_mode=generation_mode,
            max_iterations=3,
            metadata=_build_metadata("profile_update"),
        )

        if final_result is None:
            console.print(
                f"[red]Failed to generate updated profile for {entity_name}[/red]"
            )
            console.print("[yellow]Falling back to existing profile[/yellow]")
            return existing_profile, history

        # Log reflection pass/fail results for each iteration
        console.print(
            f"\n[bold magenta]Reflection Iterations for {entity_name}:[/bold magenta]"
        )
        for idx, entry in enumerate(history):
            passed_str = "✓ PASSED" if entry.get("passed", False) else "✗ FAILED"
            console.print(f"[magenta]Iteration {idx + 1} - {passed_str}[/magenta]")
            console.print(f"  Reason: {entry.get('reason', '')}")
            console.print(f"  Feedback: {entry.get('feedback', '')}\n")

        # Convert final_result to a dict
        updated_profile_dict = final_result.model_dump()

        # Ensure there's a non-empty "text" field to avoid KeyError later
        if "text" not in updated_profile_dict or not isinstance(
            updated_profile_dict["text"], str
        ):
            console.print(
                f"[red]Profile updated but no valid 'text' found. Using fallback text for {entity_name}.[/red]"
            )
            updated_profile_dict["text"] = (
                f"Profile update for {entity_name} is incomplete^[{new_article_id}]"
            )

        # Merge old and new sources
        old_sources = existing_profile.get("sources", [])
        all_sources = list(set(old_sources + [new_article_id]))
        updated_profile_dict["sources"] = all_sources

        console.print(f"[green]Successfully updated profile for {entity_name}[/green]")
        console.print(
            f"[cyan]New profile length: {len(updated_profile_dict.get('text', ''))} characters[/cyan]"
        )
        console.print(
            f"[cyan]New confidence score: {updated_profile_dict.get('confidence', 0.0)}[/cyan]"
        )
        console.print(
            f"[cyan]Number of tags: {len(updated_profile_dict.get('tags', []))}[/cyan]"
        )
        console.print(f"[cyan]Improvement iterations: {len(history)}[/cyan]")

        # Optional debug check: confirm final text length
        final_text_len = len(updated_profile_dict.get("text", ""))
        console.print(f"[cyan]Final text length after update: {final_text_len}[/cyan]")
        if final_text_len < 50:
            console.print(
                "[yellow]Warning: Final text is under 50 characters, may be incomplete.[/yellow]"
            )

        return updated_profile_dict, history

        return updated_profile_dict, history

    except Exception as e:
        console.print(f"[red]Error updating profile for {entity_name}:[/red]")
        console.print(f"[red]Error details: {str(e)}[/red]")
        import traceback

        console.print(f"[red]Traceback:\n{traceback.format_exc()}[/red]")
        raise


def create_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
) -> (Dict, list):
    """
    Create an initial profile for an entity based on article text using structured extraction.
    Returns (profile_dict, improvement_history).
    """
    console.print(f"\n[cyan]Creating profile for {entity_type} '{entity_name}'[/cyan]")
    console.print(f"[cyan]Using model: {model_type}[/cyan]")
    console.print(f"[cyan]Article ID: {article_id}[/cyan]")
    console.print(f"[cyan]Article text length: {len(article_text)} characters[/cyan]")

    try:
        profile_dict, improvement_history = generate_profile(
            entity_type, entity_name, article_text, article_id, model_type
        )

        # Extract the actual text from nested/parsed fields before logging
        profile_dict = extract_profile_text(profile_dict)

        console.print(
            f"[green]Successfully generated profile for {entity_name}[/green]"
        )
        console.print(
            f"[cyan]Profile length: {len(profile_dict.get('text', ''))} characters[/cyan]"
        )
        console.print(
            f"[cyan]Confidence score: {profile_dict.get('confidence', 0.0)}[/cyan]"
        )
        console.print(
            f"[cyan]Number of tags: {len(profile_dict.get('tags', []))}[/cyan]"
        )
        console.print(
            f"[cyan]Improvement iterations: {len(improvement_history)}[/cyan]"
        )

        return profile_dict, improvement_history
    except Exception as e:
        console.print(f"[red]Error creating profile for {entity_name}:[/red]")
        console.print(f"[red]Error details: {str(e)}[/red]")
        import traceback

        console.print(f"[red]Traceback:\n{traceback.format_exc()}[/red]")
        raise
