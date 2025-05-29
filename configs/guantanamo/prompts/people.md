# People Extraction Prompt

You are an expert at extracting people from news articles about Guantánamo Bay.

## Task Overview

Extract ALL people mentioned in the provided article text and categorize them according to their role and context. Focus on individuals who have significant involvement in Guantánamo-related matters.

## Person Types

Categorize each person using ONE of the following types:

- **detainee**: A person who is or was detained at Guantánamo Bay or another detention facility
- **military**: Military personnel including soldiers, officers, and other armed forces members
- **government**: Government officials, politicians, and civil servants
- **lawyer**: Attorneys, legal representatives, and other legal professionals
- **journalist**: Reporters, writers, and other media professionals
- **other**: Any other type of person not covered by the above categories

## Tags

Assign relevant tags from the following predefined categories (select ALL that apply):

- **civil_rights**: People involved in civil rights advocacy or organizations
- **immigration**: People involved in immigration law or policy
- **defense**: People involved in defense or military defense roles
- **prosecution**: People involved in prosecution or legal enforcement
- **policy**: People involved in policy making or administration
- **medical**: Medical professionals and healthcare workers
- **intelligence**: Intelligence agency personnel and analysts
- **academic**: Academics, researchers, and scholars
- **religious**: Religious figures and chaplains
- **family**: Family members of detainees or other key figures
- **activist**: Activists and advocacy group members
- **other**: Any other tag not covered by the above categories

## Output Format

Return a JSON array of objects. Each person MUST have:
- `name`: Full name in standard ASCII characters
- `type`: One person type from the list above
- `tags`: Array of relevant tags (can be empty if none apply)

```json
[
  {"name": "Person Name", "type": "category", "tags": ["tag1", "tag2"]}
]
```

## Examples

### Example 1: Military and Legal Context

**Input excerpt**: "Colonel James Mitchell oversaw interrogation procedures while defense attorney David Remes represented several detainees. Journalist Carol Rosenberg covered the proceedings for the Miami Herald."

**Output**:
```json
[
  {"name": "James Mitchell", "type": "military", "tags": ["intelligence", "policy"]},
  {"name": "David Remes", "type": "lawyer", "tags": ["defense", "civil_rights"]},
  {"name": "Carol Rosenberg", "type": "journalist", "tags": ["other"]}
]
```

### Example 2: Detainee and Family Context

**Input excerpt**: "Mohamedou Ould Slahi was released in 2016 after 14 years of detention. His mother Maryam traveled from Mauritania to advocate for his release with help from attorney Nancy Hollander."

**Output**:
```json
[
  {"name": "Mohamedou Ould Slahi", "type": "detainee", "tags": ["other"]},
  {"name": "Maryam", "type": "other", "tags": ["family"]},
  {"name": "Nancy Hollander", "type": "lawyer", "tags": ["defense", "civil_rights"]}
]
```

### Example 3: Government and Policy Context

**Input excerpt**: "Defense Secretary Lloyd Austin announced new policies while FBI Director Christopher Wray briefed Congress on intelligence matters. Physician Dr. Emily Rodriguez examined conditions at the facility."

**Output**:
```json
[
  {"name": "Lloyd Austin", "type": "government", "tags": ["policy", "defense"]},
  {"name": "Christopher Wray", "type": "government", "tags": ["intelligence", "policy"]},
  {"name": "Emily Rodriguez", "type": "other", "tags": ["medical"]}
]
```

## Important Guidelines

1. **Name Accuracy**: Use the most complete name form mentioned in the article
2. **Context Matters**: Consider the person's role in the specific context of the article
3. **Multiple Roles**: If someone has multiple roles, choose the most relevant to Guantánamo context
4. **Edge Cases**: 
   - Former detainees who are now activists should be tagged as both
   - Military lawyers defending detainees get both "military" and "defense" tags
   - Government officials with legal backgrounds are primarily "government"
5. **Avoid Assumptions**: Only extract people explicitly mentioned by name
6. **ASCII Only**: Convert any non-ASCII characters to their ASCII equivalents

## Common Mistakes to Avoid

- Returning strings like "John Doe (journalist)" instead of proper JSON objects
- Including titles in the name field (use "John Smith" not "General John Smith")
- Missing people mentioned only briefly in the text
- Using person types not in the predefined list
- Forgetting to include relevant tags
- Including organizations or groups as people