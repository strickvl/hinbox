# Organizations Extraction Prompt

You are an expert at extracting organizations from news articles about Guantánamo Bay.

## Task Overview

Extract ALL organizations, agencies, institutions, and formal groups mentioned in the provided article text. Focus on entities that play a role in Guantánamo-related matters, detention operations, legal proceedings, or policy decisions.

## Organization Types

Categorize each organization using ONE of the following types:

- **military**: Military organizations, units, and branches (e.g., JTF-GTMO, US Army, Navy, Marine Corps)
- **intelligence**: Intelligence agencies and services (e.g., CIA, FBI, NSA, DIA)
- **legal**: Legal organizations, law firms, and bar associations (e.g., ACLU, Center for Constitutional Rights, public defender offices)
- **humanitarian**: Organizations focused on humanitarian aid and human rights (e.g., Red Cross, Physicians for Human Rights, Amnesty International)
- **advocacy**: Advocacy groups and activist organizations (e.g., Human Rights Watch, Center for Justice & Accountability)
- **media**: Media organizations, news outlets, and publications (e.g., Miami Herald, New York Times, Reuters)
- **government**: Government entities, departments, and agencies (e.g., US DoD, DoJ, State Department, Congress)
- **intergovernmental**: International governmental bodies and treaties (e.g., UN, European Union, NATO)
- **other**: Any other organization type not covered above (e.g., academic institutions, religious organizations, private companies)

## Output Format

Return a JSON array of objects. Each organization MUST have:
- `name`: Full organization name in standard ASCII characters
- `type`: One organization type from the list above

```json
[
  {"name": "Organization Name", "type": "category"}
]
```

## Examples

### Example 1: Military and Government Context

**Input excerpt**: "Joint Task Force Guantanamo (JTF-GTMO) coordinated with the Department of Defense and FBI agents regarding interrogation protocols. The Senate Armed Services Committee held hearings."

**Output**:
```json
[
  {"name": "Joint Task Force Guantanamo", "type": "military"},
  {"name": "Department of Defense", "type": "government"},
  {"name": "FBI", "type": "intelligence"},
  {"name": "Senate Armed Services Committee", "type": "government"}
]
```

### Example 2: Legal and Advocacy Context

**Input excerpt**: "The American Civil Liberties Union and Center for Constitutional Rights filed briefs, while Human Rights Watch documented conditions. Reprieve, a UK-based organization, provided legal assistance."

**Output**:
```json
[
  {"name": "American Civil Liberties Union", "type": "legal"},
  {"name": "Center for Constitutional Rights", "type": "legal"},
  {"name": "Human Rights Watch", "type": "advocacy"},
  {"name": "Reprieve", "type": "legal"}
]
```

### Example 3: Media and International Context

**Input excerpt**: "The Miami Herald reported on the proceedings while CNN conducted interviews. The United Nations Special Rapporteur and International Committee of the Red Cross inspected facilities."

**Output**:
```json
[
  {"name": "Miami Herald", "type": "media"},
  {"name": "CNN", "type": "media"},
  {"name": "United Nations", "type": "intergovernmental"},
  {"name": "International Committee of the Red Cross", "type": "humanitarian"}
]
```

### Example 4: Academic and Other Context

**Input excerpt**: "Stanford Law School's clinic represented detainees while Physicians for Human Rights documented medical issues. Halliburton provided construction services."

**Output**:
```json
[
  {"name": "Stanford Law School", "type": "other"},
  {"name": "Physicians for Human Rights", "type": "humanitarian"},
  {"name": "Halliburton", "type": "other"}
]
```

## Important Guidelines

1. **Organization vs. Person**: Verify that each entry is truly an organization, not an individual person
2. **Full Names**: Use the complete organizational name when available (e.g., "American Civil Liberties Union" not just "ACLU")
3. **Abbreviations**: Include well-known abbreviations if that's how they appear in the text (e.g., "FBI" is acceptable)
4. **Subsidiaries**: Treat organizational units as separate entities if they have distinct roles (e.g., "Senate Armed Services Committee" separate from "US Senate")
5. **Government Levels**: Include federal, state, and local government entities
6. **International**: Include foreign governments and international organizations when relevant
7. **ASCII Only**: Convert any non-ASCII characters to their ASCII equivalents

## Common Organization Categories

### Military Organizations
- Joint Task Force Guantanamo (JTF-GTMO)
- US Army, Navy, Air Force, Marines
- Military police units
- Naval Station Guantanamo Bay

### Intelligence Agencies
- Central Intelligence Agency (CIA)
- Federal Bureau of Investigation (FBI)
- National Security Agency (NSA)
- Defense Intelligence Agency (DIA)

### Legal Organizations
- American Civil Liberties Union (ACLU)
- Center for Constitutional Rights
- Public defender offices
- Military defense organizations

### Government Entities
- Department of Defense (DoD)
- Department of Justice (DoJ)
- Department of State
- Congress, Senate, House committees
- Federal courts

### Advocacy/Humanitarian
- Human Rights Watch
- Amnesty International
- International Committee of the Red Cross
- Physicians for Human Rights

## Common Mistakes to Avoid

- Including individual people instead of organizations
- Missing organizational units mentioned in passing
- Using incorrect organization types
- Including informal groups that aren't actual organizations
- Forgetting to extract foreign or international organizations
- Missing media outlets and news organizations