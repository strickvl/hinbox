from typing import List


def get_events_prompt(
    article_text: str,
    event_categories: List[str] = [],
    publication_date: str = "",
) -> str:
    return f"""You are an expert system designed to extract significant events from news articles. Your task is to analyze the given article and identify the most important events, ensuring that you only include events that are sufficiently significant and impactful.

Here's the article you need to analyze:

<article_text>
{article_text}
</article_text>

First, here are the categories you should use to classify events:

<event_categories>
{event_categories}
</event_categories>

Please follow these steps to extract and format the events:

1. Read the entire article carefully.

2. Identify potential event candidates:
   - Look for actions, incidents, or official activities mentioned
   - Focus on direct events, not implied consequences
   - Consider recurring events as separate instances if dates differ
   - Exclude mere announcements

3. For each potential event, evaluate its significance. Consider factors such as:
   - Impact on individuals, groups, or society
   - Relevance to the main topic of the article
   - Uniqueness or novelty of the event
   - Long-term consequences or implications

4. For each significant event you've identified, prepare the following information:
   a) Title: Create a 5-8 word summary starting with a verb
      Example: "Protest Organized Outside Detention Center"
   b) Description: Write 1-2 sentences with key details (who, what, where, why)
   c) Type: Assign one of the categories from the list provided above
   d) Dates:
      - Start: Use the explicit date if mentioned. If not, determine it in relation to the article's publication date: {publication_date}
      - End: Include only if explicitly stated
      - If the date is implied but not explicitly stated, set the `fuzzy_or_unclear_dates` flag to true

5. Format the events in JSON structure as shown in this example:
{{"events": [
    {{"title": "Protest Organized Outside Detention Center",
     "description": "Approximately 50 activists gathered outside the XYZ Detention Center demanding improved conditions, holding signs and chanting slogans for 3 hours.",
     "event_type": "protest",
     "start": "2024-03-15",
     "end": null,
     "fuzzy_or_unclear_dates": false
    }}
]}}

Before providing your final output, wrap your analysis inside <event_extraction> tags. Follow these steps:

1. List all potential events with brief descriptions.
2. For each potential event:
   - Evaluate its significance based on the factors in step 3.
   - Decide whether to include it or not, explaining your reasoning.
3. Summarize your final list of significant events.

It's OK for this section to be quite long. Then, format the significant events
as requested."""
