"""Extract people from article text."""

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
from src.models import Person

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class ArticlePeople(BaseModel):
    people: List[Person]


def gemini_extract_people(text: str, model: str = CLOUD_MODEL) -> List[Dict[str, Any]]:
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

Only use standard ASCII characters for the names that you extract.
Extract all people mentioned in the text and categorize them appropriately.

12. If you footnote or add refernces, use the following format:

Normal text would go here^[source_id, source_id, ...].


You MUST return each person as an object with 'name' and 'type' properties.
For example:
[
  {"name": "John Doe", "type": "journalist"},
  {"name": "Jane Smith", "type": "lawyer"}
]

Do NOT return strings like "John Doe (journalist)". Always use the proper object
format.

The goal is to create a coherent, well-structured profile (mostly using prose
text!) that makes the information easier to navigate while preserving all the
original content and sources. Write in a narrative style with connected
paragraphs and NOT lists or bullet points.""",
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


def ollama_extract_people(text: str, model: str = OLLAMA_MODEL) -> List[Dict[str, Any]]:
    """Extract person entities from the provided text using Ollama."""
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    results = client.beta.chat.completions.parse(
        model=get_ollama_model_name(model),  # Strip ollama/ prefix for API call
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

Only use standard ASCII characters for the names that you extract.

Extract all people mentioned in the text and categorize them appropriately.

12. If you footnote or add refernces, use the following format:

Normal text would go here^[source_id, source_id, ...].

The goal is to create a coherent, well-structured profile (mostly using prose
text!) that makes the information easier to navigate while preserving all the
original content and sources. Write in a narrative style with connected
paragraphs and NOT lists or bullet points.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return results.choices[0].message.parsed.people


if __name__ == "__main__":
    text = "John Doe is a journalist at the New York Times. He is friends with Jane Smith, who is a lawyer at the same newspaper."

    print(gemini_extract_people(text))
    # print(ollama_extract_people(text))
