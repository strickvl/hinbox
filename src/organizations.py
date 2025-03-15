from typing import Any, Dict, List

import instructor
import litellm
from openai import OpenAI
from pydantic import BaseModel

from src.constants import (
    CLOUD_MODEL,
    OLLAMA_API_KEY,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    get_ollama_model_name,
)
from src.models import Organization

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class ArticleOrganizations(BaseModel):
    organizations: List[Organization]


def gemini_extract_organizations(
    text: str, model: str = CLOUD_MODEL
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Gemini."""
    client = instructor.from_litellm(litellm.completion)

    results = client.chat.completions.create(
        model=model,
        response_model=List[Organization],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting organizations from news articles about Guantánamo Bay.

When identifying organizations, categorize them using the following organization types:
- military: Military organizations (e.g., JTF-GTMO, US Army, Navy)
- intelligence: Intelligence agencies (e.g., CIA, FBI)
- legal: Legal organizations and law firms (e.g., ACLU, Center for Constitutional Rights)
- humanitarian: Organizations focused on humanitarian aid and human rights (e.g., Red Cross, Physicians for Human Rights)
- advocacy: Advocacy groups and activist organizations (e.g., Amnesty International, Human Rights Watch)
- media: Media organizations (e.g., Miami Herald, NY Times)
- government: Government entities and departments (e.g., US DoD, DoJ)
- intergovernmental: International governmental bodies (e.g., UN, European Union)
- other: Any other organization type not covered above

Make extra sure when creating an organization that it's actually an organization and
not a person.

Only use standard ASCII characters for the names that you extract.

Extract all organizations mentioned in the text and categorize them appropriately.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        metadata={
            "project_name": "hinbox",  # for braintrust
            "tags": ["dev"],
        },
    )
    return results


def ollama_extract_organizations(
    text: str, model: str = OLLAMA_MODEL
) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Ollama."""
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    results = client.beta.chat.completions.parse(
        model=get_ollama_model_name(model),  # Strip ollama/ prefix for API call
        response_format=ArticleOrganizations,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting organizations from news articles about Guantánamo Bay.

When identifying organizations, categorize them using the following organization types:
- military: Military organizations (e.g., JTF-GTMO, US Army, Navy)
- intelligence: Intelligence agencies (e.g., CIA, FBI)
- legal: Legal organizations and law firms (e.g., ACLU, Center for Constitutional Rights)
- humanitarian: Organizations focused on humanitarian aid and human rights (e.g., Red Cross, Physicians for Human Rights)
- advocacy: Advocacy groups and activist organizations (e.g., Amnesty International, Human Rights Watch)
- media: Media organizations (e.g., Miami Herald, NY Times)
- government: Government entities and departments (e.g., US DoD, DoJ)
- intergovernmental: International governmental bodies (e.g., UN, European Union)
- other: Any other organization type not covered above

Make extra sure when creating an organization that it's actually an organization and
not a person.

Only use standard ASCII characters for the names that you extract.

Extract all organizations mentioned in the text and categorize them appropriately.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return results.choices[0].message.parsed.organizations
