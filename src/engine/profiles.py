import copy
import json
import logging
import traceback
from datetime import UTC, datetime
from typing import Dict, List, Optional, Tuple

import litellm
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

from src.config_loader import get_domain_config
from src.constants import (
    CLOUD_MODEL,
    DEFAULT_TEMPERATURE,
    ENABLE_PROFILE_VERSIONING,
    MAX_ITERATIONS,
    OLLAMA_MODEL,
)
from src.exceptions import (
    ProfileGenerationError,
    ProfileUpdateError,
)
from src.utils.error_handler import (
    handle_profile_error,
    retry_on_error,
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
litellm.callbacks = ["braintrust"]

logger = logging.getLogger(__name__)
console = Console()


class ProfileVersion(BaseModel):
    """A single version of an entity profile."""

    version_number: int
    profile_data: Dict  # Complete profile snapshot
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trigger_article_id: Optional[str] = None

    @field_validator("profile_data")
    @classmethod
    def deep_copy_profile_data(cls, v):
        """Ensure profile_data is deep copied to prevent mutation."""
        return copy.deepcopy(v)


class VersionedProfile(BaseModel):
    """Container for versioned profile history."""

    current_version: int = 1
    versions: List[ProfileVersion] = Field(default_factory=list)

    def add_version(
        self, profile_data: Dict, trigger_article_id: Optional[str] = None
    ) -> ProfileVersion:
        """Add a new version to the history."""
        new_version = ProfileVersion(
            version_number=len(self.versions) + 1,
            profile_data=copy.deepcopy(profile_data),
            trigger_article_id=trigger_article_id,
        )
        self.versions.append(new_version)
        self.current_version = new_version.version_number
        return new_version

    def get_version(self, version_number: int) -> Optional[ProfileVersion]:
        """Get a specific version by number."""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None

    def get_latest(self) -> Optional[ProfileVersion]:
        """Get the latest version."""
        return self.versions[-1] if self.versions else None


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


@retry_on_error(max_retries=2, initial_delay=1.0)
def generate_profile_with_reflection(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    max_iterations: int = MAX_ITERATIONS,
    domain: str = "guantanamo",
) -> (Dict, list):
    """
    Generate a profile with reflection and improvement.
    Uses iterative_improve from utils.py.
    Returns a tuple: (final_profile_dict, improvement_history).
    """

    # Load prompt template from config
    config = get_domain_config(domain)
    prompt_template = config.load_profile_prompt("generation")

    # Format the prompt with the specific values
    custom_system_prompt = prompt_template.format(
        entity_type=entity_type, entity_name=entity_name, article_id=article_id
    )

    # Build messages for the new iterative_improve signature
    generation_messages = [
        {"role": "system", "content": custom_system_prompt},
        {
            "role": "user",
            "content": f"Article text:\n\n{article_text}\n\nCreate a profile for {entity_type} '{entity_name}'. Article ID: {article_id}",
        },
    ]

    # Load reflection prompt from config
    reflection_template = config.load_profile_prompt("reflection")
    reflection_prompt = reflection_template.format(
        entity_type=entity_type, entity_name=entity_name, article_id=article_id
    )

    # Generate initial profile text for iterative improvement
    mode = GenerationMode.CLOUD if model_type == "gemini" else GenerationMode.LOCAL
    if mode == GenerationMode.CLOUD:
        initial_response = cloud_generation(
            messages=generation_messages,
            response_model=EntityProfile,
            model=CLOUD_MODEL,
            temperature=DEFAULT_TEMPERATURE,
        )
    else:
        initial_response = local_generation(
            messages=generation_messages,
            response_model=EntityProfile,
            model=OLLAMA_MODEL,
            temperature=DEFAULT_TEMPERATURE,
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
    except Exception as e:
        logger.error(f"Error parsing profile generation result for {entity_name}: {e}")
        final_result = None

    if final_result is None:
        # Use standardized error handling for profile generation failures
        error = ProfileGenerationError(
            f"Profile generation failed after {len(history)} iterations",
            entity_name,
            entity_type,
            article_id,
            {"iterations": len(history), "history": history},
        )
        fallback_profile = handle_profile_error(
            entity_name, entity_type, article_id, error, "generation"
        )
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
    domain: str = "guantanamo",
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
        domain=domain,
    )
    return profile_dict, improvement_history


def _update_profile_internal(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
) -> Tuple[Dict, list]:
    """
    Internal function to update an existing profile with new information from an article.

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
        # Load update prompt template from config
        config = get_domain_config(domain)
        update_template = config.load_profile_prompt("update")

        # Format the base prompt
        base_prompt = update_template.format(
            entity_type=entity_type, entity_name=entity_name
        )

        # Build the complete system prompt with existing profile and new article
        system_prompt = f"""{base_prompt}

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

        # Load reflection prompt for updates
        reflection_template = config.load_profile_prompt("reflection")
        reflection_prompt = reflection_template.format(
            entity_type=entity_type, entity_name=entity_name, article_id=new_article_id
        )

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
            error = ProfileUpdateError(
                f"Failed to update profile for {entity_name}",
                entity_name,
                entity_type,
                new_article_id,
                {"existing_profile": existing_profile, "history": history},
            )
            console.print(
                f"[red]Failed to generate updated profile for {entity_name}[/red]"
            )
            console.print("[yellow]Falling back to existing profile[/yellow]")
            # Log the error but return existing profile as fallback
            logger.error(f"Profile update failed: {error}")
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

        console.print(f"[red]Traceback:\n{traceback.format_exc()}[/red]")
        raise


def update_profile(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    versioned_profile: VersionedProfile,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
) -> Tuple[Dict, VersionedProfile, list]:
    """
    Update profile and add version if versioning is enabled.

    Returns (updated_profile, versioned_profile, improvement_history).
    """
    # Call existing update logic
    updated_profile, history = _update_profile_internal(
        entity_type,
        entity_name,
        existing_profile,
        new_article_text,
        new_article_id,
        model_type,
        domain,
    )

    # Add new version if versioning enabled
    if ENABLE_PROFILE_VERSIONING:
        versioned_profile.add_version(
            profile_data=updated_profile, trigger_article_id=new_article_id
        )

    return updated_profile, versioned_profile, history


def create_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
) -> Tuple[Dict, VersionedProfile, list]:
    """
    Create an initial profile for an entity based on article text using structured extraction.
    Returns (profile_dict, versioned_profile, improvement_history).
    """
    console.print(f"\n[cyan]Creating profile for {entity_type} '{entity_name}'[/cyan]")
    console.print(f"[cyan]Using model: {model_type}[/cyan]")
    console.print(f"[cyan]Article ID: {article_id}[/cyan]")
    console.print(f"[cyan]Article text length: {len(article_text)} characters[/cyan]")

    try:
        profile_dict, improvement_history = generate_profile(
            entity_type,
            entity_name,
            article_text,
            article_id,
            model_type,
            domain,
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

        # Create versioned profile
        versioned_profile = VersionedProfile()
        if ENABLE_PROFILE_VERSIONING:
            versioned_profile.add_version(
                profile_data=profile_dict, trigger_article_id=article_id
            )

        return profile_dict, versioned_profile, improvement_history
    except Exception as e:
        console.print(f"[red]Error creating profile for {entity_name}:[/red]")
        console.print(f"[red]Error details: {str(e)}[/red]")
        import traceback

        console.print(f"[red]Traceback:\n{traceback.format_exc()}[/red]")
        raise
