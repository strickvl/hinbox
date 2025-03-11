from typing import Any, Dict, List

import instructor
import litellm
import spacy
from openai import OpenAI
from pydantic import BaseModel

from src.v2.models import Organization, OrganizationType

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class ArticleOrganizations(BaseModel):
    organizations: List[Organization]


def spacy_extract_organizations(text: str) -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using spaCy."""
    nlp = spacy.load("en_core_web_lg")

    # Process the text
    doc = nlp(text)

    # Extract organization entities
    organizations = []
    seen_organizations = set()  # Track organizations we've already added

    for ent in doc.ents:
        if ent.label_ == "ORG":
            # Only add if we haven't seen this organization name before
            if ent.text not in seen_organizations:
                organizations.append(
                    Organization(name=ent.text, type=OrganizationType.OTHER)
                )
                seen_organizations.add(ent.text)

    return organizations


def gemini_extract_organizations(
    text: str, model: str = "gemini/gemini-2.0-flash"
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


def ollama_extract_organizations(text: str, model: str = "qwq") -> List[Dict[str, Any]]:
    """Extract organization entities from the provided text using Ollama."""
    client = OpenAI(base_url="http://192.168.178.175:11434/v1", api_key="ollama")

    results = client.beta.chat.completions.parse(
        model=model,
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

Extract all organizations mentioned in the text and categorize them appropriately.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return results.choices[0].message.parsed.organizations
