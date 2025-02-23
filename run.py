import json
from datetime import datetime
from enum import Enum
from typing import List, Optional

import instructor
import litellm
from pydantic import BaseModel
from rich import print

from src.prompts import get_events_prompt

litellm.enable_json_schema_validation = True
litellm.callbacks = ["braintrust"]


class EventType(str, Enum):
    MEETING = "meeting"
    HUNGER_STRIKE = "hunger_strike"
    COURT_HEARING = "court_hearing"
    PROTEST = "protest"
    DETAINEE_TRANSFER = "detainee_transfer"  # inter-facility movements
    DETAINEE_RELEASE = (
        "detainee_release"  # Repatriation or transfer elsewhere off the island
    )
    INSPECTION_OR_VISIT = (
        "inspection_or_visit"  # Visits by officials, NGOs, Red Cross, etc.
    )
    PRESS_CONFERENCE = "press_conference"
    POLICY_ANNOUNCEMENT = (
        "policy_announcement"  # Executive order, new legislation, etc.
    )
    DEATH_IN_CUSTODY = "death_in_custody"  # Includes suicides if known cause
    MILITARY_OPERATION = (
        "military_operation"  # Large-scale or notable operational changes
    )
    INVESTIGATION = "investigation"  # Internal or external official investigations
    MEDICAL_EMERGENCY = "medical_emergency"  # Health crises beyond hunger strikes
    LEGAL_VERDICT = "legal_verdict"  # Court decisions
    INTERROGATION = "interrogation"  # Specific questioning sessions
    FACILITY_CHANGE = "facility_change"  # Facility openings/closures/modifications
    OTHER = "other"


class Event(BaseModel):
    title: str
    description: str
    event_type: EventType
    start: datetime
    end: Optional[datetime]
    fuzzy_or_unclear_dates: bool


class ArticleEvents(BaseModel):
    events: List[Event]


with open("data/raw_sources/miami_herald_articles.jsonl", "r") as file:
    first_line = file.readline()
    first_article = json.loads(first_line)
    published_date = datetime.fromtimestamp(first_article["published_date"])
    article_text = f"# {first_article['title']}\n\n## Article Publication Date: {published_date.strftime('%B %d, %Y')}\n\n## Article Content:\n\n{first_article['content']}"

publication_date = published_date.strftime("%Y-%m-%d")

EVENTS_PROMPT = get_events_prompt(
    article_text=article_text,
    event_categories=EventType._member_names_,
    publication_date=publication_date,
)

client = instructor.from_litellm(litellm.completion)

# model = "ollama/everythinglm"
# model = "ollama/llama3-gradient"
# model = "ollama/llama3.3"
# model = "ollama_chat/llama3.3"
# model = "ollama/mistral-small"
model = "gemini/gemini-2.0-flash"
# model = "gemini/gemini-2.0-pro-exp-02-05"

results = client.chat.completions.create(
    model=model,
    response_model=ArticleEvents,
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": "You are an expert at extracting events from news articles.",
        },
        {
            "role": "user",
            "content": EVENTS_PROMPT,
        },
    ],
    metadata={
        "project_name": "hinbox",  # for braintrust
        "tags": ["dev"],
    },
)

print(results)
