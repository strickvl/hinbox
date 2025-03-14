import instructor
import litellm
from openai import OpenAI
from pydantic import BaseModel, Field

from src.constants import GEMINI_MODEL, OLLAMA_API_KEY, OLLAMA_API_URL, OLLAMA_MODEL

# Ensure we have JSON schema validation enabled
litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class ArticleRelevance(BaseModel):
    """Model for determining if an article is relevant to Guantanamo detention/torture."""

    is_relevant: bool = Field(
        description="Whether the article is relevant to Guantanamo detention, prison, torture, naval base, or related issues"
    )
    reason: str = Field(
        description="Brief explanation of why the article is or is not relevant"
    )


def gemini_check_relevance(text: str, model: str = GEMINI_MODEL) -> ArticleRelevance:
    """
    Check if an article is relevant to Guantanamo detention/prison using Gemini.

    Args:
        text: The article text to check
        model: The Gemini model to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    client = instructor.from_litellm(litellm.completion)

    result = client.chat.completions.create(
        model=model,
        response_model=ArticleRelevance,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at determining whether news articles are relevant to the Guantánamo Bay detention facility, naval base, prisoner treatment, torture allegations, military tribunals, and related topics.

Determine if the article is about Guantánamo Bay or closely related issues. Articles should be considered relevant if they discuss:
- The Guantánamo Bay detention facility/prison
- The Guantánamo naval base and military operations
- Detainees or prisoners at Guantánamo
- Legal proceedings related to Guantánamo detainees
- Allegations of torture, mistreatment, or abuse at Guantánamo
- Military commissions or tribunals for Guantánamo detainees
- Hunger strikes, protests, or other actions by detainees
- Policy decisions about the detention facility
- Transfers or releases of detainees

Articles should be considered NOT relevant if they primarily discuss:
- Music festivals, tourism, or other non-military/detention activities at Guantánamo
- Passing mentions of Guantánamo that aren't central to the article

Return a boolean indicating relevance and a brief explanation.""",
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
    return result


def ollama_check_relevance(text: str, model: str = OLLAMA_MODEL) -> ArticleRelevance:
    """
    Check if an article is relevant to Guantanamo detention/prison using Ollama.

    Args:
        text: The article text to check
        model: The Ollama model to use

    Returns:
        ArticleRelevance object with is_relevant flag and reason
    """
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    result = client.beta.chat.completions.parse(
        model=model,
        response_format=ArticleRelevance,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at determining whether news articles are relevant to the Guantánamo Bay detention facility, naval base, prisoner treatment, torture allegations, military tribunals, and related topics.

Determine if the article is about Guantánamo Bay or closely related issues. Articles should be considered relevant if they discuss:
- The Guantánamo Bay detention facility/prison
- The Guantánamo naval base and military operations
- Detainees or prisoners at Guantánamo
- Legal proceedings related to Guantánamo detainees
- Allegations of torture, mistreatment, or abuse at Guantánamo
- Military commissions or tribunals for Guantánamo detainees
- Hunger strikes, protests, or other actions by detainees
- Policy decisions about the detention facility
- Transfers or releases of detainees

Articles should be considered NOT relevant if they primarily discuss:
- Music festivals, tourism, or other non-military/detention activities at Guantánamo
- Passing mentions of Guantánamo that aren't central to the article

Return a boolean indicating relevance and a brief explanation.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return result.choices[0].message.parsed
