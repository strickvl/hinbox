import json
import os
from datetime import datetime
from enum import Enum
from typing import List, Optional

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from rich import print

logfire.configure(token=os.environ["LOGFIRE_WRITE_TOKEN"])
logfire.instrument_httpx(capture_all=True)


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


class ArticleEvents(BaseModel):
    events: List[Event]


model = GeminiModel("gemini-2.0-flash", api_key=os.environ["GEMINI_API_KEY"])

agent = Agent(
    model,
    system_prompt="You are a data analyst with years of experience working on information extraction.",
    result_type=ArticleEvents,
)

# open the miami_herald_articles.jsonl file as jsonl and read the first article
with open("data/raw_sources/miami_herald_articles.jsonl", "r") as file:
    first_line = file.readline()
    first_article = json.loads(first_line)
    published_date = datetime.fromtimestamp(first_article["published_date"])
    article_text = f"# {first_article['title']}\n\n## Article Publication Date: {published_date.strftime('%B %d, %Y')}\n\n## Article Content:\n\n{first_article['content']}"

print(article_text)

result = agent.run_sync(
    f"Extract all events mentioned in the following article. Make sure to extract a title, a description, and the event type as well as the start date and an optional end date if applicable: {article_text}"
)
print(result.data)
print(result.usage())
