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
from src.models import Event

litellm.enable_json_schema_validation = True
litellm.suppress_debug_info = True
litellm.callbacks = ["braintrust"]


class ArticleEvents(BaseModel):
    events: List[Event]


def gemini_extract_events(text: str, model: str = CLOUD_MODEL) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Gemini."""
    client = instructor.from_litellm(litellm.completion)

    # Use a simpler approach with List[Event] instead of ArticleEvents
    # This might avoid the issue with default values in nested models
    try:
        results = client.chat.completions.create(
            model=model,
            response_model=List[Event],
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert at extracting events from news articles about Guantánamo Bay.

For each event, identify the following:
1. Title: A concise title for the event
2. Description: A brief description of what happened
3. Event type: Categorize using the provided event types
4. Start date: When the event started (in ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS if time is available)
5. End date: When the event ended (if applicable)
6. Is fuzzy date: Set to true if the date is approximate or described in relative terms (e.g., "last week", "early 2002")
7. Tags: Relevant tags from the provided list that apply to this event

Event types:
- transfer: Detainee transfers between facilities
- release: Detainee releases
- hearing: Court hearings or legal proceedings
- hunger_strike: Hunger strikes by detainees
- protest: Protests by detainees or outside groups
- policy_announcement: New policies regarding Guantánamo
- visit: Official visits to the facility
- torture_incident: Reported torture incidents
- medical_incident: Medical issues or incidents
- military_operation: Military operations at the facility
- legal_decision: Legal decisions affecting detainees/operations
- media_coverage: Significant media coverage events
- other: Other types of events

Event tags:
- torture: Torture allegations or incidents
- force_feeding: Force-feeding of hunger strikers
- hunger_strike: Hunger strike events
- transfer: Detainee transfer events
- interrogation: Interrogation sessions
- protest: Protest actions
- policy_change: Changes in detention policies
- legal_challenge: Legal challenges to detention
- habeas_corpus: Habeas corpus proceedings
- medical_care: Medical care issues
- isolation: Isolation of detainees
- suicide_attempt: Suicide attempts
- abuse: Detainee abuse incidents
- official_statement: Official statements about Guantánamo
- other: Other tags

For dates that are unclear or approximate, set is_fuzzy_date to true and provide your best estimate of the date.
If a specific time is mentioned, include it in the datetime format.

Extract all significant events mentioned in the text.""",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            metadata={
                "project_name": "hinbox",
                "tags": ["dev"],
            },
        )
        return results
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        # Return an empty list as fallback
        return []


def ollama_extract_events(text: str, model: str = OLLAMA_MODEL) -> List[Dict[str, Any]]:
    """Extract events from the provided text using Ollama."""
    client = OpenAI(base_url=OLLAMA_API_URL, api_key=OLLAMA_API_KEY)

    results = client.beta.chat.completions.parse(
        model=get_ollama_model_name(model),  # Strip ollama/ prefix for API call
        response_format=ArticleEvents,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an expert at extracting events from news articles about Guantánamo Bay.

For each event, identify the following:
1. Title: A concise title for the event
2. Description: A brief description of what happened
3. Event type: Categorize using the provided event types
4. Start date: When the event started (in ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS if time is available)
5. End date: When the event ended (if applicable)
6. Is fuzzy date: Set to true if the date is approximate or described in relative terms (e.g., "last week", "early 2002")
7. Tags: Relevant tags from the provided list that apply to this event

Event types:
- transfer: Detainee transfers between facilities
- release: Detainee releases
- hearing: Court hearings or legal proceedings
- hunger_strike: Hunger strikes by detainees
- protest: Protests by detainees or outside groups
- policy_announcement: New policies regarding Guantánamo
- visit: Official visits to the facility
- torture_incident: Reported torture incidents
- medical_incident: Medical issues or incidents
- military_operation: Military operations at the facility
- legal_decision: Legal decisions affecting detainees/operations
- media_coverage: Significant media coverage events
- other: Other types of events

Event tags:
- torture: Torture allegations or incidents
- force_feeding: Force-feeding of hunger strikers
- hunger_strike: Hunger strike events
- transfer: Detainee transfer events
- interrogation: Interrogation sessions
- protest: Protest actions
- policy_change: Changes in detention policies
- legal_challenge: Legal challenges to detention
- habeas_corpus: Habeas corpus proceedings
- medical_care: Medical care issues
- isolation: Isolation of detainees
- suicide_attempt: Suicide attempts
- abuse: Detainee abuse incidents
- official_statement: Official statements about Guantánamo
- other: Other tags

For dates that are unclear or approximate, set is_fuzzy_date to true and provide your best estimate of the date.
If a specific time is mentioned, include it in the datetime format.

Extract all significant events mentioned in the text.""",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )
    return results.choices[0].message.parsed.events
