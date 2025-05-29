# Events Extraction Prompt

You are an expert at extracting events from news articles about Guantánamo Bay.

## Task Overview

Extract ALL significant events, incidents, proceedings, and occurrences mentioned in the provided article text. Focus on discrete, time-bound events that have relevance to Guantánamo operations, legal proceedings, policy decisions, or related activities.

## Event Types

Categorize each event using ONE of the following types:

- **detention**: Events related to capture, arrest, transfer, or initial detention of individuals
- **legal**: Court cases, hearings, trials, tribunals, legal proceedings, and rulings
- **military_operation**: Military actions, raids, operations, and tactical activities
- **policy_change**: Changes in government or military policy, regulations, or procedures
- **protest**: Demonstrations, protests, hunger strikes, and civil disobedience
- **investigation**: Official investigations, inquiries, reports, and fact-finding missions
- **release**: Release, transfer, or repatriation of detainees
- **medical**: Medical events, treatments, health crises, and healthcare-related incidents
- **other**: Any other event type not covered above (e.g., construction, visits, ceremonies)

## Output Format

Return a JSON array of objects. Each event MUST have:
- `title`: Concise event description (noun phrase, not full sentence)
- `type`: One event type from the list above
- `date`: Date in YYYY-MM-DD format when available, or null if not specified

```json
[
  {"title": "Event Description", "type": "category", "date": "YYYY-MM-DD"}
]
```

## Examples

### Example 1: Legal and Detention Events

**Input excerpt**: "On March 15, 2023, the military commission heard testimony in the case of Khalid Sheikh Mohammed. The defendant was originally captured in Pakistan in 2003 and transferred to Guantanamo in 2006."

**Output**:
```json
[
  {"title": "Military commission hearing for Khalid Sheikh Mohammed", "type": "legal", "date": "2023-03-15"},
  {"title": "Capture of Khalid Sheikh Mohammed in Pakistan", "type": "detention", "date": "2003-01-01"},
  {"title": "Transfer of Khalid Sheikh Mohammed to Guantanamo", "type": "detention", "date": "2006-01-01"}
]
```

### Example 2: Policy and Investigation Events

**Input excerpt**: "The FBI launched an investigation into interrogation practices in January 2004. Later that year, the Pentagon announced new detention policies following the Abu Ghraib scandal."

**Output**:
```json
[
  {"title": "FBI investigation into interrogation practices", "type": "investigation", "date": "2004-01-01"},
  {"title": "Pentagon announcement of new detention policies", "type": "policy_change", "date": "2004-01-01"}
]
```

### Example 3: Medical and Protest Events

**Input excerpt**: "In 2005, over 130 detainees participated in a hunger strike protesting their conditions. Dr. Smith examined several participants for medical complications on June 10, 2005."

**Output**:
```json
[
  {"title": "Mass hunger strike by 130+ detainees", "type": "protest", "date": "2005-01-01"},
  {"title": "Medical examination of hunger strike participants", "type": "medical", "date": "2005-06-10"}
]
```

### Example 4: Release and Military Events

**Input excerpt**: "Operation Enduring Freedom began in October 2001. The first detainee transfers to Guantanamo occurred on January 11, 2002. Three detainees were released to their home countries on December 20, 2016."

**Output**:
```json
[
  {"title": "Operation Enduring Freedom begins", "type": "military_operation", "date": "2001-10-01"},
  {"title": "First detainee transfers to Guantanamo", "type": "detention", "date": "2002-01-11"},
  {"title": "Release of three detainees to home countries", "type": "release", "date": "2016-12-20"}
]
```

## Important Guidelines

1. **Discrete Events**: Focus on specific, time-bound occurrences rather than ongoing states or conditions
2. **Concise Titles**: Use noun phrases that clearly identify the event without being a full sentence
3. **Date Precision**: Use the most specific date available (full date > month/year > year only)
4. **Multiple Events**: If an article mentions several related events, extract each separately
5. **Context Relevance**: Include events that are directly related to Guantanamo or have clear implications
6. **ASCII Only**: Convert any non-ASCII characters to their ASCII equivalents

## Date Format Guidelines

- Full date: "2023-03-15" (when exact date is known)
- Month/year: "2023-03-01" (use first day of month when only month/year known)
- Year only: "2023-01-01" (use January 1st when only year is known)
- No date: null (when no temporal information is provided)

## Common Event Categories

### Legal Events
- Court hearings and trials
- Legal rulings and decisions
- Filing of lawsuits or appeals
- Habeas corpus proceedings
- Military commission sessions

### Detention Events
- Initial captures and arrests
- Transfers between facilities
- Arrival at Guantanamo
- Change in detention status
- Classification reviews

### Policy Events
- New military directives
- Changes in detention procedures
- Congressional legislation
- Executive orders
- International agreements

### Investigation Events
- FBI inquiries
- Inspector General reports
- Congressional investigations
- Military investigations
- International monitoring

### Medical Events
- Health crises or incidents
- Medical examinations
- Treatment programs
- Psychological assessments
- Health policy changes

## Common Mistakes to Avoid

- Including ongoing states rather than discrete events (e.g., "being detained" vs. "transfer to detention")
- Using full sentences instead of concise titles
- Missing dates that are clearly mentioned in the text
- Extracting too many minor or irrelevant events
- Confusing event types (e.g., a court ruling about policy is "legal" not "policy_change")
- Including vague or unclear event descriptions
- Forgetting to extract events mentioned in passing or background context

## Special Considerations

### Timeline Events
- Pay attention to historical context and chronological information
- Include both recent and historical events mentioned

### Multiple Participants
- When events involve multiple people, focus on the event itself rather than individual actions

### Causal Relationships
- Extract both triggering events and resulting events when both are mentioned

### International Events
- Include events that occurred outside the US but are relevant to Guantanamo matters