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
    custom_system_prompt = f"""You are an expert at creating detailed profiles for entities mentioned in news articles.

Create a comprehensive profile for {entity_type} "{entity_name}" using ONLY information from the provided article.

CRITICAL REQUIREMENTS:
1. Text length: Minimum 100 characters - provide substantial detail, not just basic facts
2. Citations: Use smart citation strategy to reduce repetition:
   - Group related facts in paragraphs, cite once per paragraph: paragraph content^[{article_id}]
   - For multiple sources use: fact^[{article_id},other_id]
   - Don't cite every sentence - cite logical groupings of information
3. Structure: Use markdown section headers (### Background, ### Role, ### Current Position, etc.)
4. Content: Extract specific facts, quotes, actions, relationships from the article
5. Tags: Include at least 2 relevant descriptive tags/keywords
6. Confidence: Score 0.0-1.0 based on information quality and completeness
7. JSON output: Return valid JSON with "text", "tags", "confidence", "sources" fields

EXAMPLE FORMAT:
{{
  "text": "John Smith is a military officer who currently oversees detention operations at Guantánamo Bay. He has extensive experience in military operations and has been stationed at the facility since 2019^[{article_id}].\\n\\n### Background\\nSmith previously served in Afghanistan for two years before joining the detention facility staff. He graduated from West Point in 2010 and has received multiple commendations for his service^[{article_id}].\\n\\n### Current Role\\nAs facility operations manager, he oversees daily detention procedures and coordinates with legal teams. His responsibilities include managing staff schedules and ensuring compliance with military regulations^[{article_id}].",
  "tags": ["military", "detention-operations", "guantanamo"],
  "confidence": 0.9,
  "sources": ["{article_id}"]
}}

Remember: Group facts logically with strategic citations, use section headers, make it detailed and substantial.
"""

    # Build messages for the new iterative_improve signature
    generation_messages = [
        {"role": "system", "content": custom_system_prompt},
        {
            "role": "user",
            "content": f"Article text:\n\n{article_text}\n\nCreate a profile for {entity_type} '{entity_name}'. Article ID: {article_id}",
        },
    ]

    reflection_prompt = f"""You are evaluating a profile for {entity_type} "{entity_name}". Check if it meets ALL requirements:

REQUIRED CRITERIA:
1. Text length: Minimum 100 characters (current profile should be substantial, not just a name)
2. Citations: Must have citations but use smart strategy:
   - At least one citation per paragraph/section
   - Format: ^[{article_id}] or ^[{article_id},other_id] for multiple sources
   - Don't need citation on every sentence, group logically
3. Structure: Must have section headers (### Background, ### Role, etc.)
4. Content: Must contain specific facts from the article, not generic information
5. JSON format: Must be valid JSON with "text", "tags", "confidence", "sources" fields
6. Tags: Must include at least 2 relevant tags
7. Confidence: Must be between 0.0 and 1.0

COMMON FAILURES:
- No citations anywhere in text
- Too short/generic text (under 100 chars)
- No section headers
- Invalid JSON structure
- Empty or missing tags array

Mark valid=true ONLY if ALL criteria are met. If any fail, mark valid=false and specify exactly which criteria failed."""

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

        # Build messages for iterative improvement
        generation_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please create an updated profile by merging the existing profile with new information from the article.",
            },
        ]

        # Reflection prompt for updates
        reflection_prompt = f"""Check if this updated profile for {entity_type} "{entity_name}" meets all requirements:

REQUIRED CRITERIA:
1. Text length: Minimum 100 characters
2. Citations: Strategic citations with ^[article_id] format for facts
3. Structure: Section headers (### Background, ### Role, etc.)
4. Content: Integrates both old and new information appropriately
5. JSON format: Valid JSON with "text", "tags", "confidence", "sources" fields
6. Tags: At least 2 relevant tags
7. Confidence: Between 0.0 and 1.0

Mark valid=true ONLY if ALL criteria are met."""

        # We'll use the same EntityProfile structure for the updated profile.
        result = iterative_improve(
            initial_text="{}",  # Start with empty JSON
            generation_messages=generation_messages,
            reflection_prompt=reflection_prompt,
            response_model=EntityProfile,
            max_iterations=3,
            mode=generation_mode,
        )

        final_result = None
        history = result.get("reflection_history", [])

        # Parse the final result
        try:
            import json

            final_text = result.get("text", "{}")
            if final_text.strip():
                if final_text.strip().startswith("{"):
                    data = json.loads(final_text)
                    final_result = EntityProfile(**data)
                else:
                    # If it's not JSON, wrap it
                    final_result = EntityProfile(
                        text=final_text,
                        tags=existing_profile.get("tags", []),
                        confidence=0.8,
                        sources=existing_profile.get("sources", []) + [new_article_id],
                    )
        except Exception as e:
            console.print(f"[red]Error parsing iterative improvement result: {e}[/red]")
            final_result = None

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
            passed_str = "✓ PASSED" if entry.get("valid", False) else "✗ FAILED"
            console.print(f"[magenta]Iteration {idx + 1} - {passed_str}[/magenta]")
            console.print(f"  Reasoning: {entry.get('reasoning', '')}")
            if entry.get("issues"):
                console.print(f"  Issues: {entry.get('issues', '')}\n")
            else:
                console.print()

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
