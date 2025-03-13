import logging
import random
import time
from datetime import datetime
from functools import wraps
from typing import Dict, List, Literal

import instructor
import litellm
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console

from src.v2.constants import GEMINI_MODEL, OLLAMA_API_KEY, OLLAMA_API_URL, OLLAMA_MODEL

console = Console()

# Ensure we have JSON schema validation enabled
litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]

EntityType = Literal["person", "location", "event", "organization"]
ModelType = Literal["gemini", "ollama"]

logger = logging.getLogger(__name__)


def exponential_backoff(max_retries: int = 5, initial_delay: float = 1.0):
    """Decorator for exponential backoff retry logic."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if retry == max_retries - 1:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise

                    # Add jitter to avoid thundering herd
                    jitter = random.uniform(0, 0.1) * delay
                    sleep_time = delay + jitter

                    logger.warning(
                        f"Attempt {retry + 1} failed: {str(e)}. "
                        f"Retrying in {sleep_time:.2f} seconds..."
                    )

                    time.sleep(sleep_time)
                    delay *= 2  # Exponential backoff
            return None

        return wrapper

    return decorator


@exponential_backoff()
def generate_with_gemini(prompt: str, model: str = GEMINI_MODEL) -> str:
    """Generate text using Gemini with retry logic."""
    client = instructor.from_litellm(litellm.completion)
    results = client.chat.completions.create(
        model=model,
        response_model=List[str],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating detailed, factual profiles in Markdown format.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        metadata={
            "project_name": "hinbox",
            "tags": ["dev"],
        },
    )
    return results[0] if results else ""


@exponential_backoff()
def generate_with_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Generate text using Ollama with retry logic."""
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating detailed, factual profiles in Markdown format.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response.choices[0].message.content


def get_creation_prompt(
    entity_type: EntityType, name: str, text: str, article_id: str
) -> str:
    """Get prompt for initial profile creation with enhanced instructions."""
    return f"""Given this article text:
{text}

Create a profile about {name} ({entity_type}) in Markdown format. The profile should:

1. Start with the profile text (no need for a header with the entity name)
2. Include only factual information from the article
3. Be concise but thorough
4. Use proper Markdown formatting:
   - Use bullet points for lists
   - Use *italics* for emphasis
   - Use inline footnotes for every fact: `text^[{article_id}]
   - When multiple sources confirm a fact, combine them: text^[abc123, def456]
   - always use the article id in the footnote and don't add extra text

5. If there are uncertainties:
   - Note them explicitly with "reportedly" or "according to"
   - Create a "## Conflicting Information" section if needed

Example profile:
```
John Smith is a military officer^[abc123] who oversees operations at Guantánamo Bay^[abc123]. He was appointed to this position in January 2024^[def456].

### Background
* Previously served in Afghanistan^[Source: article abc123]
* Has extensive experience in detention operations^[abc123, def456]
```

Return ONLY the profile text in Markdown format."""


def get_update_prompt(
    entity_type: EntityType, name: str, current: str, text: str, article_id: str
) -> str:
    """Get prompt for profile updating with enhanced instructions."""
    return f"""Current profile:
{current}

New article text:
{text}

Update the profile for {name} ({entity_type}). The updated profile should:

1. Keep the existing Markdown format and sections (unless conflicting
   information needs its own section)
2. Add new information from the article
3. Use inline footnotes for new facts: `text^[{article_id}]`
4. When an article confirms existing information, add it to the existing footnote:
   - Before: `text^[abc123]`
   - After: `text^[abc123, {article_id}]`
5. Create a "## Conflicting Information" section if needed

Return ONLY the updated profile text in Markdown format."""


def generate_profile(prompt: str, model_type: ModelType) -> str:
    """Generate profile text using specified LLM."""
    logger.info(f"Generating profile using {model_type} model")
    if model_type == "gemini":
        logger.info("Using Gemini for profile generation")
        return generate_with_gemini(prompt)
    logger.info("Using Ollama for profile generation")
    return generate_with_ollama(prompt)


def create_profile(
    entity_type: EntityType,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: ModelType = "gemini",
) -> Dict[str, str]:
    """Create initial profile for an entity with metadata."""
    prompt = get_creation_prompt(entity_type, entity_name, article_text, article_id)
    profile_text = generate_profile(prompt, model_type)

    return {
        "text": profile_text,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "last_article_id": article_id,
    }


def update_profile(
    entity_type: EntityType,
    entity_name: str,
    current_profile: Dict[str, str],
    article_text: str,
    article_id: str,
    model_type: ModelType = "gemini",
) -> Dict[str, str]:
    """Update existing profile with new information."""
    prompt = get_update_prompt(
        entity_type, entity_name, current_profile["text"], article_text, article_id
    )
    updated_text = generate_profile(prompt, model_type)

    return {
        "text": updated_text,
        "created_at": current_profile["created_at"],
        "updated_at": datetime.now().isoformat(),
        "last_article_id": article_id,
    }


def regenerate_profile(
    entity_type: EntityType,
    entity_name: str,
    articles: List[Dict[str, str]],
    model_type: ModelType = "gemini",
) -> Dict[str, str]:
    """Regenerate profile from all available articles. Not used by default."""
    # Start with the oldest article
    sorted_articles = sorted(articles, key=lambda x: x["published_date"])

    # Create initial profile from first article
    profile = create_profile(
        entity_type,
        entity_name,
        sorted_articles[0]["content"],
        sorted_articles[0]["id"],
        model_type,
    )

    # Update with each subsequent article
    for article in sorted_articles[1:]:
        profile = update_profile(
            entity_type,
            entity_name,
            profile,
            article["content"],
            article["id"],
            model_type,
        )

    return profile


class EntityProfile(BaseModel):
    """Model for entity profiles."""

    text: str = Field(description="Comprehensive profile text about the entity")
    tags: List[str] = Field(
        description="List of relevant tags/keywords for this entity",
        default_factory=list,
    )
    confidence: float = Field(
        description="Confidence score (0-1) in the accuracy of this profile",
        ge=0,
        le=1,
    )
    sources: List[str] = Field(
        description="List of article IDs used as sources for this profile",
        default_factory=list,
    )


def generate_profile(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
) -> Dict:
    """
    Generate a profile for an entity based on article text.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        article_text: The text of the article
        article_id: The ID of the article
        model_type: The type of model to use (gemini or ollama)

    Returns:
        Dict containing the profile
    """
    if model_type == "gemini":
        return generate_with_gemini(entity_type, entity_name, article_text, article_id)
    else:
        return generate_with_ollama(entity_type, entity_name, article_text, article_id)


def generate_with_gemini(
    entity_type: str, entity_name: str, article_text: str, article_id: str
) -> Dict:
    """
    Generate a profile for an entity using Gemini.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        article_text: The text of the article
        article_id: The ID of the article

    Returns:
        Dict containing the profile
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
                
Your task is to create a comprehensive profile for a {entity_type} named "{entity_name}" based on the article text provided.

The profile should:
1. Be written in clear, concise language
2. Include all relevant information from the article about the {entity_type}
3. Be organized in a logical structure with paragraphs
4. Focus only on factual information present in the article
5. Not include speculation or information not supported by the article
6. Include relevant dates, locations, and connections to other entities when available
7. For people, include their role, affiliation, actions, and any biographical details
8. For organizations, include their purpose, activities, leadership, and significance
9. For locations, include geographical context, significance, and relevant events
10. For events, include when and where they occurred, who was involved, and their significance

Also provide:
- Relevant tags/keywords for this entity
- A confidence score (0-1) indicating how confident you are in the accuracy and completeness of this profile based on the available information
- The article ID as the source

If there is very limited information about the entity in the article, acknowledge this in your profile and keep it brief while still providing what information is available.""",
            },
            {
                "role": "user",
                "content": f"Here is the article text:\n\n{article_text}\n\nPlease create a profile for the {entity_type} named '{entity_name}'.",
            },
        ],
        metadata={
            "project_name": "hinbox",  # for braintrust
            "tags": ["dev"],
        },
    )

    # Add the article ID to sources
    result_dict = result.model_dump()
    result_dict["sources"] = [article_id]
    return result_dict


def generate_with_ollama(
    entity_type: str, entity_name: str, article_text: str, article_id: str
) -> Dict:
    """
    Generate a profile for an entity using Ollama.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        article_text: The text of the article
        article_id: The ID of the article

    Returns:
        Dict containing the profile
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    result = client.beta.chat.completions.parse(
        model=OLLAMA_MODEL,
        response_format=EntityProfile,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at creating profiles for entities mentioned in news articles.
                
Your task is to create a comprehensive profile for a {entity_type} named "{entity_name}" based on the article text provided.

The profile should:
1. Be written in clear, concise language
2. Include all relevant information from the article about the {entity_type}
3. Be organized in a logical structure with paragraphs
4. Focus only on factual information present in the article
5. Not include speculation or information not supported by the article
6. Include relevant dates, locations, and connections to other entities when available
7. For people, include their role, affiliation, actions, and any biographical details
8. For organizations, include their purpose, activities, leadership, and significance
9. For locations, include geographical context, significance, and relevant events
10. For events, include when and where they occurred, who was involved, and their significance

Also provide:
- Relevant tags/keywords for this entity
- A confidence score (0-1) indicating how confident you are in the accuracy and completeness of this profile based on the available information
- The article ID as the source

If there is very limited information about the entity in the article, acknowledge this in your profile and keep it brief while still providing what information is available.""",
            },
            {
                "role": "user",
                "content": f"Here is the article text:\n\n{article_text}\n\nPlease create a profile for the {entity_type} named '{entity_name}'.",
            },
        ],
    )

    # Convert to dict and add the article ID to sources
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
    Update an existing profile with new information from an article.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        existing_profile: The existing profile to update
        new_article_text: The text of the new article
        new_article_id: The ID of the new article
        model_type: The type of model to use (gemini or ollama)

    Returns:
        Dict containing the updated profile
    """
    if model_type == "gemini":
        return update_with_gemini(
            entity_type, entity_name, existing_profile, new_article_text, new_article_id
        )
    else:
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
    Update an existing profile with new information using Gemini.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        existing_profile: The existing profile to update
        new_article_text: The text of the new article
        new_article_id: The ID of the new article

    Returns:
        Dict containing the updated profile
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
                
Your task is to update an existing profile for a {entity_type} named "{entity_name}" with new information from an article.

The updated profile should:
1. Incorporate all relevant NEW information from the article
2. Maintain the existing information that is still relevant
3. Resolve any contradictions between old and new information (prefer newer information)
4. Be written in clear, concise language
5. Be organized in a logical structure with paragraphs
6. Focus only on factual information present in the articles
7. Not include speculation or information not supported by the articles
8. Include relevant dates, locations, and connections to other entities
9. For people, include their role, affiliation, actions, and any biographical details
10. For organizations, include their purpose, activities, leadership, and significance
11. For locations, include geographical context, significance, and relevant events
12. For events, include when and where they occurred, who was involved, and their significance

Also update:
- Relevant tags/keywords for this entity (add new ones if appropriate)
- The confidence score (0-1) based on the combined information
- Add the new article ID to the sources list

If the new article doesn't add significant information, you can keep the profile mostly the same but acknowledge that the entity was mentioned in the new article.""",
            },
            {
                "role": "user",
                "content": f"""Here is the existing profile:

{existing_profile["text"]}

Existing tags: {", ".join(existing_profile["tags"])}
Existing confidence score: {existing_profile["confidence"]}
Existing sources: {", ".join(existing_profile["sources"])}

Here is the new article text:

{new_article_text}

Please update the profile for the {entity_type} named '{entity_name}' with any new information from this article.""",
            },
        ],
        metadata={
            "project_name": "hinbox",  # for braintrust
            "tags": ["dev"],
        },
    )

    # Add the new article ID to sources
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
    Update an existing profile with new information using Ollama.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        existing_profile: The existing profile to update
        new_article_text: The text of the new article
        new_article_id: The ID of the new article

    Returns:
        Dict containing the updated profile
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    result = client.beta.chat.completions.parse(
        model=OLLAMA_MODEL,
        response_format=EntityProfile,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert at updating profiles for entities mentioned in news articles.
                
Your task is to update an existing profile for a {entity_type} named "{entity_name}" with new information from an article.

The updated profile should:
1. Incorporate all relevant NEW information from the article
2. Maintain the existing information that is still relevant
3. Resolve any contradictions between old and new information (prefer newer information)
4. Be written in clear, concise language
5. Be organized in a logical structure with paragraphs
6. Focus only on factual information present in the articles
7. Not include speculation or information not supported by the articles
8. Include relevant dates, locations, and connections to other entities
9. For people, include their role, affiliation, actions, and any biographical details
10. For organizations, include their purpose, activities, leadership, and significance
11. For locations, include geographical context, significance, and relevant events
12. For events, include when and where they occurred, who was involved, and their significance

Also update:
- Relevant tags/keywords for this entity (add new ones if appropriate)
- The confidence score (0-1) based on the combined information
- Add the new article ID to the sources list

If the new article doesn't add significant information, you can keep the profile mostly the same but acknowledge that the entity was mentioned in the new article.""",
            },
            {
                "role": "user",
                "content": f"""Here is the existing profile:

{existing_profile["text"]}

Existing tags: {", ".join(existing_profile["tags"])}
Existing confidence score: {existing_profile["confidence"]}
Existing sources: {", ".join(existing_profile["sources"])}

Here is the new article text:

{new_article_text}

Please update the profile for the {entity_type} named '{entity_name}' with any new information from this article.""",
            },
        ],
    )

    # Convert to dict and add the new article ID to sources
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
    Create a profile for an entity based on article text.

    This is a wrapper around generate_profile for backward compatibility.

    Args:
        entity_type: The type of entity (person, organization, location, event)
        entity_name: The name of the entity
        article_text: The text of the article
        article_id: The ID of the article
        model_type: The type of model to use (gemini or ollama)

    Returns:
        Dict containing the profile
    """
    return generate_profile(
        entity_type, entity_name, article_text, article_id, model_type
    )
