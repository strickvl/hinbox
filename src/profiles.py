import logging
import re
from typing import Dict, List

import instructor
import litellm
from openai import OpenAI
from pydantic import BaseModel, Field

from src.constants import (
    CLOUD_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_ollama_model_name,
)
from src.utils import GenerationMode, ReflectionResult, iterative_improve

# Enable JSON schema validation for structured responses
litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]

logger = logging.getLogger(__name__)


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


class ProfileValidation(BaseModel):
    """Model for validating generated profiles"""

    is_valid: bool = Field(
        ..., description="Whether the profile meets all requirements"
    )
    reason: str = Field(..., description="Explanation of validation result")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for improvement if invalid"
    )


def validate_profile(profile: Dict) -> ProfileValidation:
    """
    Validate a generated profile against defined criteria.

    Checks:
    1. Required sections present
    2. Citation format correct
    3. Reasonable confidence score
    """
    issues = []
    suggestions = []

    # Check text exists and isn't too short
    text = profile.get("text", "")
    if not text or len(text.strip()) < 50:  # Basic sanity check
        issues.append("Profile text is missing or too short")
        suggestions.append("Expand the profile with more details from the source")

    # Check for required sections (flexible - looks for any section headers)
    if not any(line.startswith("#") for line in text.split("\n")):
        issues.append("No sections found in profile")
        suggestions.append("Add relevant sections like 'Background', 'Role', etc.")

    # Check citation format
    citation_pattern = r"\^\[([^\]]+)\]"
    citations = re.findall(citation_pattern, text)
    if not citations:
        issues.append("No citations found")
        suggestions.append("Add citations in the format: fact^[article_id]")
    else:
        # Check citation format
        invalid_citations = [
            c
            for c in citations
            if "," in c and not c.replace(" ", "").replace(",", "").isalnum()
        ]
        if invalid_citations:
            issues.append("Invalid citation format found")
            suggestions.append(
                "Use only article IDs in citations, e.g. fact^[abc123] or fact^[abc123, def456]"
            )

    # Check confidence score
    confidence = profile.get("confidence", 0)
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        issues.append("Invalid confidence score")
        suggestions.append("Provide confidence score between 0 and 1")

    # Check tags exist
    tags = profile.get("tags", [])
    if not tags:
        issues.append("No tags provided")
        suggestions.append("Add relevant tags for the profile")

    # Check sources exist
    sources = profile.get("sources", [])
    if not sources:
        issues.append("No sources listed")
        suggestions.append("Include source article IDs")

    is_valid = len(issues) == 0
    reason = "Profile meets all requirements" if is_valid else "; ".join(issues)

    return ProfileValidation(is_valid=is_valid, reason=reason, suggestions=suggestions)


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

    def _validate_response(response: EntityProfile) -> ReflectionResult:
        """Validate the generated profile and return reflection result."""
        validation = validate_profile(response.model_dump())
        return ReflectionResult(
            result=validation.is_valid,
            reason=validation.reason,
            feedback_for_improvement="\n".join(validation.suggestions),
        )

    # Enhanced system prompt emphasizing requirements
    system_prompt = f"""You are an expert at creating profiles for entities mentioned in news articles.

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

    # Use iterative_improve to generate and refine the profile
    final_result, history = iterative_improve(
        prompt=f"Article text:\n\n{article_text}\n\nCreate a profile for {entity_type} '{entity_name}'. Article ID: {article_id}",
        response_model=EntityProfile,
        generation_mode=GenerationMode.CLOUD
        if model_type == "gemini"
        else GenerationMode.LOCAL,
        max_iterations=max_iterations,
        metadata={"project_name": "hinbox", "tags": ["profile_generation"]},
    )

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


def generate_with_gemini(
    entity_type: str, entity_name: str, article_text: str, article_id: str
) -> Dict:
    """
    Generate a profile for an entity using the Gemini model with structured extraction.

    Args:
        entity_type: The type of entity
        entity_name: The name of the entity
        article_text: The text of the article
        article_id: The article ID to add as a source

    Returns:
        Dict containing the structured profile.
    """
    client = instructor.from_litellm(litellm.completion)
    result = client.chat.completions.create(
        model=CLOUD_MODEL,
        response_model=EntityProfile,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at creating profiles for entities mentioned in news articles.

Your task is to create a comprehensive profile for a {entity_type} named "{entity_name}" based solely on the provided article text.

The profile should:
1. Be written in clear, concise language and organized in logical paragraphs.
2. Include only factual information present in the article.
3. Provide relevant dates, locations, and connections to other entities.
4. For people, include their role, affiliation, actions, and biographical details.
5. For organizations, include their purpose, activities, leadership, and significance.
6. For locations, include geographical context and relevant events.
7. For events, include when and where they occurred, who was involved, and their significance.

Additionally, the profile must adhere to the following Markdown formatting rules:
- Use bullet points for lists.
- Use *italics* for emphasis.
- Include inline footnotes for every fact using the format: `fact text^[{article_id}]`.
- When multiple sources confirm a fact, combine them into one inline footnote (e.g., `fact text^[abc123, def456]`).
- Always include the article ID in the footnote without adding extra text.
- For uncertainties, explicitly indicate them with terms like "reportedly" or "according to", and if needed, include a "## Conflicting Information" section.

Example profile:
```
John Smith is a military officer^[abc123] who oversees operations at Guantánamo Bay^[abc123]. He was appointed in January 2024^[def456].

### Background
* Previously served in Afghanistan^[abc123]
* Extensive experience in detention operations^[abc123, def456]
```

Also include:
- Relevant tags/keywords for the entity.
- A confidence score (0-1) reflecting the completeness and accuracy of the profile.
- The article ID as a source.

If the article offers limited information, acknowledge this and keep the profile brief.
""",
            },
            {
                "role": "user",
                "content": f"Here is the article text:\n\n{article_text}\n\nPlease create a profile for the {entity_type} named '{entity_name}'. This article has the following ID: {article_id}",
            },
        ],
        metadata={
            "project_name": "hinbox",
            "tags": ["dev"],
        },
    )
    result_dict = result.model_dump()
    result_dict["sources"] = [article_id]
    return result_dict


def generate_with_ollama(
    entity_type: str, entity_name: str, article_text: str, article_id: str
) -> Dict:
    """
    Generate a profile for an entity using the Ollama model with structured extraction.

    Args:
        entity_type: The type of entity
        entity_name: The name of the entity
        article_text: The article text to extract from
        article_id: The article ID to add as a source

    Returns:
        Dict containing the structured profile.
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)
    result = client.beta.chat.completions.parse(
        model=get_ollama_model_name(OLLAMA_MODEL),  # Strip ollama/ prefix
        response_format=EntityProfile,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at creating profiles for entities mentioned in news articles.

Your task is to create a comprehensive profile for a {entity_type} named "{entity_name}" based solely on the provided article text.

The profile should:
1. Be written in clear, concise language and organized in logical paragraphs.
2. Include only factual information present in the article.
3. Provide relevant dates, locations, and connections to other entities.
4. For people, include their role, affiliation, actions, and biographical details.
5. For organizations, include their purpose, activities, leadership, and significance.
6. For locations, include geographical context and relevant events.
7. For events, include when and where they occurred, who was involved, and their significance.

Additionally, the profile must adhere to the following Markdown formatting rules:
- Use bullet points for lists.
- Use *italics* for emphasis.
- Include inline footnotes for every fact using the format: `fact text^[{article_id}]`.
- When multiple sources confirm a fact, combine them into one inline footnote (e.g., `fact text^[abc123, def456]`).
- Always include the article ID in the footnote without adding extra text.
- For uncertainties, explicitly indicate them with terms like "reportedly" or "according to", and if needed, include a "## Conflicting Information" section.

Example profile:
```
John Smith is a military officer^[abc123] who oversees operations at Guantánamo Bay^[abc123]. He was appointed in January 2024^[def456].

### Background
* Previously served in Afghanistan^[abc123]
* Extensive experience in detention operations^[abc123, def456]
```

Also include:
- Relevant tags/keywords for the entity.
- A confidence score (0-1) reflecting the completeness and accuracy of the profile.
- The article ID as a source.

If the article offers limited information, acknowledge this and keep the profile brief.
""",
            },
            {
                "role": "user",
                "content": f"Here is the article text:\n\n{article_text}\n\nPlease create a profile for the {entity_type} named '{entity_name}'. This article has the following ID: {article_id}",
            },
        ],
    )
    result_dict = result.choices[0].message.parsed.model_dump()
    result_dict["sources"] = [article_id]
    return result_dict


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

    # For the iterative process, we treat the entire text above as the 'prompt' so the LLM merges them.
    from src.utils import GenerationMode, iterative_improve

    generation_mode = (
        GenerationMode.CLOUD if model_type == "gemini" else GenerationMode.LOCAL
    )

    # We'll use the same EntityProfile structure for the updated profile.
    final_result, history = iterative_improve(
        prompt=system_prompt,
        response_model=EntityProfile,
        generation_mode=generation_mode,
        max_iterations=3,
        metadata={"project_name": "hinbox", "tags": ["profile_update"]},
    )

    if final_result is None:
        # If it fails, just return existing_profile as fallback
        return existing_profile, history

    updated_profile_dict = final_result.model_dump()
    # Merge old and new sources
    old_sources = existing_profile.get("sources", [])
    all_sources = list(set(old_sources + [new_article_id]))
    updated_profile_dict["sources"] = all_sources

    return updated_profile_dict, history


def update_with_gemini(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
) -> Dict:
    """
    Update an existing profile using the Gemini model with structured extraction.

    Args:
        entity_type: The type of entity
        entity_name: The name of the entity
        existing_profile: The current structured profile (as a dict)
        new_article_text: The text of the new article
        new_article_id: The new article ID to add to sources

    Returns:
        Dict containing the updated structured profile.
    """
    client = instructor.from_litellm(litellm.completion)
    result = client.chat.completions.create(
        model=CLOUD_MODEL,
        response_model=EntityProfile,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at updating profiles for entities mentioned in news articles.

Your task is to update the existing profile for a {entity_type} named "{entity_name}" with new information from the provided article text.

The updated profile should:
1. Incorporate any new relevant factual information.
2. Retain existing valid information.
3. Resolve contradictions by preferring newer details.
4. Be clear, concise, and organized in logical paragraphs.
5. Include updated tags/keywords and adjust the confidence score accordingly.
6. Add the new article ID as an additional source.

Additionally, ensure the updated profile follows these Markdown formatting rules:
- Use bullet points for lists.
- Use *italics* for emphasis.
- Include inline footnotes for every fact using the format: `fact text^[{new_article_id}]`.
- Combine footnotes for facts confirmed by multiple sources (e.g., `fact text^[abc123, def456]`).
- Always include the article ID in the footnote.
- Mark uncertainties with "reportedly" or "according to", and if needed, add a "## Conflicting Information" section.

Existing profile details and sources should be maintained appropriately.
""",
            },
            {
                "role": "user",
                "content": f"""Existing Profile:
{existing_profile["text"]}

Existing Tags: {", ".join(existing_profile["tags"])}
Existing Confidence: {existing_profile["confidence"]}
Existing Sources: {", ".join(existing_profile["sources"])}

New Article Text:
{new_article_text}

Please update the profile for the {entity_type} named '{entity_name}' with any
new information. This article has the following ID: {new_article_id}
""",
            },
        ],
        metadata={
            "project_name": "hinbox",
            "tags": ["dev"],
        },
    )
    result_dict = result.model_dump()
    result_dict["sources"] = list(
        set(existing_profile.get("sources", []) + [new_article_id])
    )
    return result_dict


def update_with_ollama(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict,
    new_article_text: str,
    new_article_id: str,
) -> Dict:
    """
    Update an existing profile using the Ollama model with structured extraction.

    Args:
        entity_type: The type of entity
        entity_name: The name of the entity
        existing_profile: The current structured profile (as a dict)
        new_article_text: The text of the new article
        new_article_id: The new article ID to add to sources

    Returns:
        Dict containing the updated structured profile.
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)
    result = client.beta.chat.completions.parse(
        model=get_ollama_model_name(OLLAMA_MODEL),  # Strip ollama/ prefix
        response_format=EntityProfile,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at updating profiles for entities mentioned in news articles.

Your task is to update the existing profile for a {entity_type} named "{entity_name}" with new information from the provided article text.

The updated profile should:
1. Incorporate any new relevant factual information.
2. Retain existing valid information.
3. Resolve contradictions by preferring newer details.
4. Be clear, concise, and organized in logical paragraphs.
5. Include updated tags/keywords and adjust the confidence score accordingly.
6. Add the new article ID as an additional source.

Additionally, ensure the updated profile adheres to these Markdown formatting rules:
- Use bullet points for lists.
- Use *italics* for emphasis.
- Include inline footnotes for every fact using the format: `fact text^[{new_article_id}]`.
- Combine footnotes when facts are confirmed by multiple sources (e.g., `fact text^[abc123, def456]`).
- Always include the article ID in the footnote.
- Mark uncertainties with "reportedly" or "according to", and if needed, add a "## Conflicting Information" section.

Maintain existing profile details and sources as appropriate.
""",
            },
            {
                "role": "user",
                "content": f"""Existing Profile:
{existing_profile["text"]}

Existing Tags: {", ".join(existing_profile["tags"])}
Existing Confidence: {existing_profile["confidence"]}
Existing Sources: {", ".join(existing_profile["sources"])}

New Article Text:
{new_article_text}

Please update the profile for the {entity_type} named '{entity_name}' with any
new information. This article has the following ID: {new_article_id}
""",
            },
        ],
    )
    result_dict = result.choices[0].message.parsed.model_dump()
    result_dict["sources"] = list(
        set(existing_profile.get("sources", []) + [new_article_id])
    )
    return result_dict


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
    profile_dict, improvement_history = generate_profile(
        entity_type, entity_name, article_text, article_id, model_type
    )
    return profile_dict, improvement_history
