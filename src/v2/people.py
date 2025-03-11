from typing import Any, Dict, List

import instructor
import litellm
import spacy
from openai import OpenAI
from pydantic import BaseModel

from src.v2.models import Person, PersonType

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class ArticlePeople(BaseModel):
    people: List[Person]


def spacy_extract_people(text: str) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using spaCy."""
    nlp = spacy.load("en_core_web_lg")

    # Process the text
    doc = nlp(text)

    # Extract person entities
    people = []
    seen_people = set()  # Track people we've already added

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Only add if we haven't seen this person name before
            if ent.text not in seen_people:
                people.append(Person(name=ent.text, type=PersonType.OTHER))
                seen_people.add(ent.text)

    return people


def gemini_extract_people(
    text: str, model: str = "gemini/gemini-2.0-flash"
) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Gemini."""
    client = instructor.from_litellm(litellm.completion)

    results = client.chat.completions.create(
        model=model,
        response_model=List[Person],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting people from news articles.

When identifying people, categorize them using the following person types:
- detainee: A person who is or was detained at Guantánamo Bay or another detention facility
- military: Military personnel including soldiers, officers, and other armed forces members
- government: Government officials, politicians, and civil servants
- lawyer: Attorneys, legal representatives, and other legal professionals
- journalist: Reporters, writers, and other media professionals
- other: Any other type of person not covered by the above categories

Extract all people mentioned in the text and categorize them appropriately.""",
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


def ollama_extract_people(text: str, model: str = "qwq") -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Ollama."""
    client = OpenAI(base_url="http://192.168.178.175:11434/v1", api_key="ollama")

    results = client.beta.chat.completions.parse(
        model=model,
        response_format=ArticlePeople,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting people from news articles.

When identifying people, categorize them using the following person types:
- detainee: A person who is or was detained at Guantánamo Bay or another detention facility
- military: Military personnel including soldiers, officers, and other armed forces members
- government: Government officials, politicians, and civil servants
- lawyer: Attorneys, legal representatives, and other legal professionals
- journalist: Reporters, writers, and other media professionals
- other: Any other type of person not covered by the above categories

Extract all people mentioned in the text and categorize them appropriately.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return results.choices[0].message.parsed.people
