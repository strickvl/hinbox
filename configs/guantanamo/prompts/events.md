# Events Extraction Prompt

You are an expert at extracting events from news articles about Guantánamo Bay.

## Task Overview

Extract ALL significant events, incidents, proceedings, and occurrences mentioned in the provided article text. Focus on discrete, time-bound events that have relevance to Guantánamo operations, legal proceedings, policy decisions, or related activities.

## Event Types

Categorize each event using ONE of the following `event_type` values:

- **detention**: Events related to capture, arrest, transfer, or initial detention of individuals
- **legal**: Court cases, hearings, trials, tribunals, legal proceedings, and rulings
- **military_operation**: Military actions, raids, operations, and tactical activities
- **policy_change**: Changes in government or military policy, regulations, or procedures
- **protest**: Demonstrations, protests, hunger strikes, and civil disobedience
- **investigation**: Official investigations, inquiries, reports, and fact-finding missions
- **release**: Release, transfer, or repatriation of detainees
- **medical**: Medical events, treatments, health crises, and healthcare-related incidents
- **other**: Any other event type not covered above (e.g., construction, visits, ceremonies)

## Event Tags

Tag events with zero or more of these tags for additional categorization:

- **torture**: Torture allegations or incidents
- **force_feeding**: Force-feeding of hunger strikers
- **hunger_strike**: Hunger strike events
- **transfer**: Detainee transfer events
- **interrogation**: Interrogation sessions
- **policy_change**: Changes in detention policies
- **legal_challenge**: Legal challenges to detention
- **habeas_corpus**: Habeas corpus proceedings
- **medical_care**: Medical care issues
- **isolation**: Isolation of detainees
- **suicide_attempt**: Suicide attempts
- **abuse**: Detainee abuse incidents
- **official_statement**: Official statements about Guantanamo

## Output Format

Return a JSON array of event objects. Each event object MUST have these fields:

- `title` (string, required): Concise event description (noun phrase, not full sentence)
- `description` (string, required): Detailed explanation of what happened, including key participants and context
- `event_type` (string, required): One of the event types listed above
- `start_date` (string, required): ISO 8601 date-time format (see Date Guidelines below)
- `end_date` (string or null, optional): ISO 8601 date-time if the event spans a period, otherwise null
- `is_fuzzy_date` (boolean, optional): Set to true if the date is approximate; defaults to false
- `tags` (array of strings, optional): Zero or more tags from the list above

```json
[
  {
    "title": "Event Description",
    "description": "Detailed explanation of what happened",
    "event_type": "legal",
    "start_date": "2023-03-15T00:00:00",
    "end_date": null,
    "is_fuzzy_date": false,
    "tags": ["legal_challenge"]
  }
]
```

## Examples

### Example 1: Legal and Detention Events

**Input excerpt**: "On March 15, 2023, the military commission heard testimony in the case of Khalid Sheikh Mohammed. The defendant was originally captured in Pakistan in 2003 and transferred to Guantanamo in 2006."

**Output**:
```json
[
  {
    "title": "Military commission hearing for Khalid Sheikh Mohammed",
    "description": "The military commission heard testimony in the case of Khalid Sheikh Mohammed at Guantanamo Bay.",
    "event_type": "legal",
    "start_date": "2023-03-15T00:00:00",
    "end_date": null,
    "is_fuzzy_date": false,
    "tags": ["legal_challenge"]
  },
  {
    "title": "Capture of Khalid Sheikh Mohammed in Pakistan",
    "description": "Khalid Sheikh Mohammed was captured in Pakistan prior to his transfer to Guantanamo.",
    "event_type": "detention",
    "start_date": "2003-01-01T00:00:00",
    "end_date": null,
    "is_fuzzy_date": true,
    "tags": []
  },
  {
    "title": "Transfer of Khalid Sheikh Mohammed to Guantanamo",
    "description": "Khalid Sheikh Mohammed was transferred to the Guantanamo Bay detention facility.",
    "event_type": "detention",
    "start_date": "2006-01-01T00:00:00",
    "end_date": null,
    "is_fuzzy_date": true,
    "tags": ["transfer"]
  }
]
```

### Example 2: Policy and Investigation Events

**Input excerpt**: "The FBI launched an investigation into interrogation practices in January 2004. Later that year, the Pentagon announced new detention policies following the Abu Ghraib scandal."

**Output**:
```json
[
  {
    "title": "FBI investigation into interrogation practices",
    "description": "The FBI launched a formal investigation into interrogation practices at detention facilities.",
    "event_type": "investigation",
    "start_date": "2004-01-01T00:00:00",
    "end_date": null,
    "is_fuzzy_date": true,
    "tags": ["interrogation"]
  },
  {
    "title": "Pentagon announcement of new detention policies",
    "description": "The Pentagon announced new detention policies in response to the Abu Ghraib scandal.",
    "event_type": "policy_change",
    "start_date": "2004-01-01T00:00:00",
    "end_date": null,
    "is_fuzzy_date": true,
    "tags": ["policy_change"]
  }
]
```

### Example 3: Medical and Protest Events

**Input excerpt**: "In 2005, over 130 detainees participated in a hunger strike protesting their conditions. Dr. Smith examined several participants for medical complications on June 10, 2005."

**Output**:
```json
[
  {
    "title": "Mass hunger strike by 130+ detainees",
    "description": "Over 130 detainees participated in a hunger strike to protest detention conditions at Guantanamo.",
    "event_type": "protest",
    "start_date": "2005-01-01T00:00:00",
    "end_date": null,
    "is_fuzzy_date": true,
    "tags": ["hunger_strike"]
  },
  {
    "title": "Medical examination of hunger strike participants",
    "description": "Dr. Smith examined several hunger strike participants for medical complications.",
    "event_type": "medical",
    "start_date": "2005-06-10T00:00:00",
    "end_date": null,
    "is_fuzzy_date": false,
    "tags": ["medical_care", "hunger_strike"]
  }
]
```

## Date Guidelines

All dates must be in ISO 8601 date-time format (`YYYY-MM-DDT00:00:00`):

- **Full date known**: `"2023-03-15T00:00:00"` with `is_fuzzy_date: false`
- **Month/year known**: `"2023-03-01T00:00:00"` with `is_fuzzy_date: true`
- **Year only known**: `"2023-01-01T00:00:00"` with `is_fuzzy_date: true`
- **No date at all**: Do NOT extract the event (since `start_date` is required). Only extract events that have at least a year mentioned or inferable from context.

## Important Guidelines

1. **Discrete Events**: Focus on specific, time-bound occurrences rather than ongoing states or conditions
2. **Concise Titles**: Use noun phrases that clearly identify the event without being a full sentence
3. **Meaningful Descriptions**: Provide enough context to understand the event without reading the source article
4. **Date Precision**: Use the most specific date available and set `is_fuzzy_date` accordingly
5. **Multiple Events**: If an article mentions several related events, extract each separately
6. **Context Relevance**: Include events that are directly related to Guantanamo or have clear implications
7. **ASCII Only**: Convert any non-ASCII characters to their ASCII equivalents
8. **Use Tags**: Apply relevant tags from the list above to aid categorization

## Common Mistakes to Avoid

- Using `type` instead of `event_type` (must be `event_type`)
- Using `date` instead of `start_date` (must be `start_date`)
- Omitting the `description` field (it is required)
- Using plain date strings like "2023-03-15" instead of ISO date-time "2023-03-15T00:00:00"
- Including ongoing states rather than discrete events
- Using full sentences instead of concise titles
- Extracting events with no temporal anchor at all
