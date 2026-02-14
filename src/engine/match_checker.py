"""Entity match checking functionality."""

from pydantic import BaseModel, Field

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.logging_config import log
from src.utils.llm import cloud_generation, local_generation


class MatchCheckResult(BaseModel):
    is_match: bool
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the model is in the match decision (0=uncertain, 1=certain)",
    )
    reason: str


def local_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    model: str = OLLAMA_MODEL,
) -> MatchCheckResult:
    """Check if new article evidence refers to the same entity as an existing profile.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: Evidence text (or profile) from the new article
        existing_profile_text: The existing profile text we're comparing against
        model: The LLM model to use for comparison
    """
    system_content = """You are an expert analyst specializing in entity resolution for news articles about Guantánamo Bay.

Your task is to determine if two profiles refer to the same real-world entity (person, organization, location, or event).

Consider the following when making your determination:
1. Name variations: Different spellings, nicknames, titles, or partial names
2. Contextual information: Role, affiliations, actions, and biographical details
3. Temporal consistency: Whether the information in both profiles could apply to the same entity at different times
4. Sub-location rule: If one location is a smaller subset (e.g., a camp within Guantánamo Bay), it is NOT the same as the larger location.

You MUST provide:
- is_match: true or false
- confidence: a float from 0.0 to 1.0 indicating how certain you are (0.9+ = very confident, 0.5-0.7 = uncertain, below 0.5 = guessing)
- reason: a detailed explanation citing specific evidence

If one is a sub-location or facility inside a bigger one, do NOT merge them."""

    user_content = f"""I need to determine if these two entities refer to the same real-world entity:

## EVIDENCE FROM NEW ARTICLE:
Name: {new_name}
{new_profile_text}

## EXISTING PROFILE IN DATABASE:
Name: {existing_name}
Profile: {existing_profile_text}

Do these refer to the same entity? Provide your analysis with a confidence score."""

    try:
        return local_generation(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_model=MatchCheckResult,
            model=model,
            temperature=0,
        )
    except Exception as e:
        log("Error with Ollama API", level="error", exception=e)
        # Return a default result indicating failure
        return MatchCheckResult(
            is_match=False, confidence=0.0, reason=f"API error: {str(e)}"
        )


def cloud_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    model: str = CLOUD_MODEL,
) -> MatchCheckResult:
    """Check if new article evidence refers to the same entity as an existing profile.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: Evidence text (or profile) from the new article
        existing_profile_text: The existing profile text we're comparing against
        model: The LLM model to use for comparison
    """
    system_content = """You are an expert analyst specializing in entity resolution for news articles about Guantánamo Bay.

Your task is to determine if two profiles refer to the same real-world entity (person, organization, location, or event).

Consider the following when making your determination:
1. Name variations: Different spellings, nicknames, titles, or partial names
2. Contextual information: Role, affiliations, actions, and biographical details
3. Temporal consistency: Whether the information in both profiles could apply to the same entity at different times
4. Sub-location rule: If one location is a smaller subset (e.g., a camp within Guantánamo Bay), it is NOT the same as the larger location.

You MUST provide:
- is_match: true or false
- confidence: a float from 0.0 to 1.0 indicating how certain you are (0.9+ = very confident, 0.5-0.7 = uncertain, below 0.5 = guessing)
- reason: a detailed explanation citing specific evidence

If one is a sub-location or facility inside a bigger one, do NOT merge them."""

    user_content = f"""I need to determine if these two entities refer to the same real-world entity:

## EVIDENCE FROM NEW ARTICLE:
Name: {new_name}
{new_profile_text}

## EXISTING PROFILE IN DATABASE:
Name: {existing_name}
Profile: {existing_profile_text}

Do these refer to the same entity? Provide your analysis with a confidence score."""

    try:
        return cloud_generation(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_model=MatchCheckResult,
            model=model,
            temperature=0,
        )
    except Exception as e:
        log("Error with Gemini API", level="error", exception=e)
        return MatchCheckResult(
            is_match=False, confidence=0.0, reason=f"API error: {str(e)}"
        )
