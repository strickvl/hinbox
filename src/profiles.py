import logging
from typing import Dict, List

import instructor
import litellm
from openai import OpenAI
from pydantic import BaseModel, Field

from src.constants import (
    GEMINI_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_ollama_model_name,
)

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


def generate_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
) -> Dict:
    """
    Generate a profile for an entity based on article text using structured extraction.

    Args:
        entity_type: Type of entity (e.g., person, organization, location, event)
        entity_name: The name of the entity
        article_text: Text of the article to extract information from
        article_id: The article ID to use as source
        model_type: Model to use ("gemini" or "ollama")

    Returns:
        Dict representation of the structured profile.
    """
    if model_type == "gemini":
        logger.info("Generating profile using Gemini model")
        return generate_with_gemini(entity_type, entity_name, article_text, article_id)
    else:
        logger.info("Generating profile using Ollama model")
        return generate_with_ollama(entity_type, entity_name, article_text, article_id)


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
        model=GEMINI_MODEL,
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
) -> Dict:
    """
    Update an existing profile with new information from an article using structured extraction.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        existing_profile: The current structured profile (as a dict)
        new_article_text: The text of the new article to extract additional information from
        new_article_id: The new article ID to add to the sources
        model_type: The model to use ("gemini" or "ollama")

    Returns:
        Dict containing the updated structured profile.
    """
    if model_type == "gemini":
        logger.info("Updating profile using Gemini model")
        return update_with_gemini(
            entity_type, entity_name, existing_profile, new_article_text, new_article_id
        )
    else:
        logger.info("Updating profile using Ollama model")
        return update_with_ollama(
            entity_type, entity_name, existing_profile, new_article_text, new_article_id
        )


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
        model=GEMINI_MODEL,
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
) -> Dict:
    """
    Create an initial profile for an entity based on article text using structured extraction.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        article_text: The article text to extract information from
        article_id: The article ID to use as source
        model_type: The model to use ("gemini" or "ollama")

    Returns:
        Dict containing the structured profile.
    """
    return generate_profile(
        entity_type, entity_name, article_text, article_id, model_type
    )
