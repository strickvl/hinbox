# Locations Extraction Prompt

You are an expert at extracting locations from news articles about Guantánamo Bay.

## Task Overview

Extract ALL geographical locations, facilities, and places mentioned in the provided article text. Focus on locations that are relevant to Guantánamo operations, detention matters, legal proceedings, or related activities.

## Location Types

Categorize each location using ONE of the following types:

- **detention_facility**: Detention centers, prisons, camps, and holding facilities (e.g., Camp X-Ray, Camp Delta, Camp Echo)
- **military_base**: Military installations, bases, and facilities (e.g., Naval Station Guantanamo Bay, Fort Bragg)
- **country**: Nations and countries (e.g., United States, Afghanistan, Cuba)
- **city**: Cities, towns, and municipalities (e.g., Washington D.C., Miami, Kabul)
- **region**: Geographic regions, states, provinces, and territories (e.g., Middle East, Florida, Helmand Province)
- **building**: Specific buildings, structures, or facilities (e.g., Pentagon, White House, courthouses)
- **other**: Any other location type not covered above (e.g., geographic features, borders, zones)

## Output Format

Return a JSON array of objects. Each location MUST have:
- `name`: Full location name in standard ASCII characters
- `type`: One location type from the list above

```json
[
  {"name": "Location Name", "type": "category"}
]
```

## Examples

### Example 1: Detention and Military Facilities

**Input excerpt**: "Detainees were transferred from Camp X-Ray to Camp Delta at Naval Station Guantanamo Bay. Some were later moved to Camp Echo for special housing."

**Output**:
```json
[
  {"name": "Camp X-Ray", "type": "detention_facility"},
  {"name": "Camp Delta", "type": "detention_facility"},
  {"name": "Naval Station Guantanamo Bay", "type": "military_base"},
  {"name": "Camp Echo", "type": "detention_facility"}
]
```

### Example 2: International Context

**Input excerpt**: "The prisoner was captured in Afghanistan near Kandahar and transported through Bagram Air Base before arriving in Cuba. His family in Pakistan received notice."

**Output**:
```json
[
  {"name": "Afghanistan", "type": "country"},
  {"name": "Kandahar", "type": "city"},
  {"name": "Bagram Air Base", "type": "military_base"},
  {"name": "Cuba", "type": "country"},
  {"name": "Pakistan", "type": "country"}
]
```

### Example 3: Legal and Government Context

**Input excerpt**: "The case was heard in federal court in Washington D.C. before proceeding to the Pentagon. Officials from Florida and the Middle East testified."

**Output**:
```json
[
  {"name": "Washington D.C.", "type": "city"},
  {"name": "Pentagon", "type": "building"},
  {"name": "Florida", "type": "region"},
  {"name": "Middle East", "type": "region"}
]
```

### Example 4: Mixed Geographic Context

**Input excerpt**: "The Supreme Court in Washington ruled while protests occurred outside the White House. International observers from Europe monitored the Guantanamo Bay area."

**Output**:
```json
[
  {"name": "Supreme Court", "type": "building"},
  {"name": "Washington", "type": "city"},
  {"name": "White House", "type": "building"},
  {"name": "Europe", "type": "region"},
  {"name": "Guantanamo Bay", "type": "other"}
]
```

## Important Guidelines

1. **Specificity Levels**: Include both specific (Camp Delta) and general (Guantanamo Bay) location references
2. **Official Names**: Use official names when available (e.g., "Naval Station Guantanamo Bay" rather than just "Guantanamo")
3. **Geographic Hierarchy**: Include countries, regions, cities as separate entities even if mentioned together
4. **Facility Types**: Distinguish between different types of facilities (detention vs. military vs. government buildings)
5. **International Scope**: Include foreign locations relevant to detainee origins, operations, or legal proceedings
6. **ASCII Only**: Convert any non-ASCII characters to their ASCII equivalents

## Common Location Categories

### Detention Facilities at Guantanamo
- Camp X-Ray (original temporary facility)
- Camp Delta (main detention facility)
- Camp Echo (special housing)
- Camp Iguana (juvenile detention)
- Camp 7 (high-value detainees)

### Military Installations
- Naval Station Guantanamo Bay
- Bagram Air Base (Afghanistan)
- Fort Bragg (North Carolina)
- Pentagon (Virginia)

### Key Countries
- United States
- Afghanistan
- Pakistan
- Iraq
- Yemen
- Saudi Arabia
- Cuba

### Important Cities
- Washington D.C.
- Miami
- New York
- Kabul
- Baghdad

### Regions
- Middle East
- Central Asia
- Caribbean
- Florida
- Virginia

## Special Considerations

### Guantanamo-Specific Locations
- Pay special attention to different camps and facilities within Guantanamo
- Include both current and historical facility names
- Note administrative and operational areas

### Capture and Transit Locations
- Include locations where detainees were originally captured
- Note transit points and temporary holding facilities
- Include countries and regions of origin

### Legal Venues
- Federal courts and their locations
- Military commission locations
- Administrative hearing venues

## Naming Rules (CRITICAL)

1. **Use proper nouns, not descriptive phrases.** Every location name must be
   a recognizable proper noun, not a description of what it is:
   - CORRECT: "Naval Station Guantanamo Bay" — a proper name
   - WRONG: "U.S. military base in Guantanamo Bay" — a description
   - CORRECT: "Bagram Air Base" — a proper name
   - WRONG: "military base in Afghanistan" — a description
   - CORRECT: "Guantanamo Bay" — a proper name
   - WRONG: "detention center at Guantanamo" — a description

2. **Never use prepositional descriptions as names.** If a location is described
   with "in", "at", "near", or "outside" followed by another place, that is a
   description, not a proper name. Extract the underlying proper name instead.

3. **Guantanamo canonicalization:** When articles refer to the base/detention
   complex using phrases like "the Guantanamo Bay facility", "the base at
   Guantanamo", or "GTMO", extract as "Guantanamo Bay" with type
   "military_base". Only use "Naval Station Guantanamo Bay" if the article
   specifically uses that name.

4. **"U.S. soil" is not a location name.** When articles mention "on U.S. soil"
   or "American territory", extract as "United States" with type "country".

## Common Mistakes to Avoid

- **Using descriptive phrases instead of proper names** (e.g., "prison near Havana" instead of its actual name)
- Missing specific camp names within Guantanamo
- Ignoring foreign locations mentioned in detainee backgrounds
- Confusing similar location names (e.g., Washington state vs. Washington D.C.)
- Missing building names when they're specifically mentioned
- Forgetting to include both specific and general geographic references
- Using incorrect location types for the context