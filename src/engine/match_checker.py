"""Entity match checking functionality."""

from pydantic import BaseModel

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.logging_config import log
from src.utils.llm import cloud_generation, local_generation


class MatchCheckResult(BaseModel):
    is_match: bool
    reason: str


def local_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    model: str = OLLAMA_MODEL,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
) -> MatchCheckResult:
    """
    Check if a newly extracted profile refers to the same entity as an existing profile.

    This function uses an LLM to determine if two profiles refer to the same person,
    even when names might have variations or additional context is needed to establish identity.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: The profile text generated from the new article
        existing_profile_text: The existing profile text we're comparing against
        model: The LLM model to use for comparison
        langfuse_session_id: The Langfuse session ID
        langfuse_trace_id: The Langfuse trace ID
    """
    system_content = """You are an expert analyst specializing in entity
                               resolution for news articles about Guantánamo Bay.

                    Your task is to determine if two profiles refer to the same real-world entity (person, organization, location, or event).

                    Consider the following when making your determination:
                    1. Name variations: Different spellings, nicknames, titles, or partial names
                    2. Contextual information: Role, affiliations, actions, and biographical details
                    3. Temporal consistency: Whether the information in both profiles could apply to the same entity at different times
                    4. Sub-location rule: If one location is a smaller subset (e.g., a camp within Guantánamo Bay), it is NOT the same as the larger location.

                    Provide a detailed explanation for your decision, citing specific evidence from both profiles. If one is a sub-location or facility inside a bigger one, do NOT merge them.
                    """

    user_content = f"""I need to determine if these two profiles refer to the same entity:

## PROFILE FROM NEW ARTICLE:
Name: {new_name}
Profile: {new_profile_text}

## EXISTING PROFILE IN DATABASE:
Name: {existing_name}
Profile: {existing_profile_text}

Are these profiles referring to the same entity? Provide your analysis."""

    try:
        return local_generation(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_model=MatchCheckResult,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        log(f"Error with Ollama API", level="error", exception=e)
        # Return a default result indicating failure
        return MatchCheckResult(is_match=False, reason=f"API error: {str(e)}")


def cloud_model_check_match(
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    model: str = CLOUD_MODEL,
    langfuse_session_id: str = None,
    langfuse_trace_id: str = None,
) -> MatchCheckResult:
    """
    Check if a newly extracted profile refers to the same entity as an existing profile,
    using a cloud-based LLM. This function specifically ensures sub-locations
    are not merged with their larger locations.

    Args:
        new_name: The name extracted from the new article
        existing_name: The name from an existing profile in our database
        new_profile_text: The profile text generated from the new article
        existing_profile_text: The existing profile text we're comparing against
        model: The LLM model to use for comparison
        langfuse_session_id: The Langfuse session ID
        langfuse_trace_id: The Langfuse trace ID

    Returns:
        MatchCheckResult with is_match flag and detailed reasoning
    """
    system_content = """You are an expert analyst specializing in entity resolution for news articles about Guantánamo Bay.

Your task is to determine if two profiles refer to the same real-world entity (person, organization, location, or event).

Consider the following when making your determination:
1. Name variations: Different spellings, nicknames, titles, or partial names
2. Contextual information: Role, affiliations, actions, and biographical details
3. Temporal consistency: Whether the information in both profiles could apply to the same entity at different times
4. Sub-location rule: If one location is a smaller subset (e.g., a camp within Guantánamo Bay), it is NOT the same as the larger location.

Provide a detailed explanation for your decision, citing specific evidence from both profiles. If one is a sub-location or facility inside a bigger one, do NOT merge them.
"""

    user_content = f"""I need to determine if these two profiles refer to the same entity:

## PROFILE FROM NEW ARTICLE:
Name: {new_name}
Profile: {new_profile_text}

## EXISTING PROFILE IN DATABASE:
Name: {existing_name}
Profile: {existing_profile_text}

Are these profiles referring to the same entity? Provide your analysis."""

    try:
        return cloud_generation(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_model=MatchCheckResult,
            model=model,
            temperature=0,
            langfuse_session_id=langfuse_session_id,
            langfuse_trace_id=langfuse_trace_id,
        )
    except Exception as e:
        log(f"Error with Gemini API", level="error", exception=e)
        return MatchCheckResult(is_match=False, reason=f"API error: {str(e)}")
