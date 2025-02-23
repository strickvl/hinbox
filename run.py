import json
from datetime import datetime
from enum import Enum
from typing import List, Optional

import instructor
from litellm import completion
from pydantic import BaseModel
from rich import print


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


# open the miami_herald_articles.jsonl file as jsonl and read the first article
with open("data/raw_sources/miami_herald_articles.jsonl", "r") as file:
    first_line = file.readline()
    first_article = json.loads(first_line)
    published_date = datetime.fromtimestamp(first_article["published_date"])
    article_text = f"# {first_article['title']}\n\n## Article Publication Date: {published_date.strftime('%B %d, %Y')}\n\n## Article Content:\n\n{first_article['content']}"

PROMPT = f"""Analyze this news article and extract all significant events. Follow these steps:

1. Identify Event Candidates:
- Look for actions, incidents, or official activities mentioned
- Include both direct events and implied consequences
- Consider recurring events as separate instances if dates differ

2. For each event:
a) Title: Create a 5-8 word summary starting with verb (e.g. "Hunger Strike Initiated Over Visitation Rights")
b) Description: 1-2 sentences with key details (who, what, where, why)
c) Type: Strictly use these categories:
{EventType._member_names_}

d) Dates:
- Start: Use explicit date if mentioned, otherwise article publication date ({published_date.strftime("%Y-%m-%d")})
- End: Only include if explicitly stated

3. Output Format Example:
{{
  "events": [
    {{
      "title": "Protest Organized Outside Detention Center",
      "description": "Approximately 50 activists gathered outside the XYZ Detention Center demanding improved conditions, holding signs and chanting slogans for 3 hours.",
      "event_type": "protest",
      "start": "2024-03-15",
      "end": null
    }}
  ]
}}

Article Content:
{article_text}

Maintain strict JSON schema compliance.

JSON Output:"""

client = instructor.from_litellm(completion)

model = "ollama/mistral-small"
# model = "gemini/gemini-2.0-flash"

results = client.chat.completions.create(
    model=model,
    response_model=ArticleEvents,
    messages=[
        {
            "role": "user",
            "content": PROMPT,
        },
    ],
)

print(results)
