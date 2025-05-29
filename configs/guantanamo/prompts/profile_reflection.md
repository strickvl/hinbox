# Profile Reflection Prompt

You are evaluating a profile for {entity_type} "{entity_name}". Check if it meets ALL requirements.

## Required Criteria

1. **Text length**: Minimum 100 characters (current profile should be substantial, not just a name)
2. **Citations**: Must have citations but use smart strategy:
   - At least one citation per paragraph/section
   - Format: ^[{article_id}] (single article ID only)
   - Don't need citation on every sentence, group logically
3. **Structure**: Must have section headers (### Background, ### Role, etc.)
4. **Content**: Must contain specific facts from the article, not generic information
5. **JSON format**: Must be valid JSON with "text", "tags", "confidence", "sources" fields
6. **Tags**: Must include at least 2 relevant tags
7. **Confidence**: Must be between 0.0 and 1.0

## Common Failures

- No citations anywhere in text
- Too short/generic text (under 100 chars)
- No section headers
- Invalid JSON structure
- Empty or missing tags array

## Evaluation Response

Mark `valid=true` ONLY if ALL criteria are met. If any fail, mark `valid=false` and specify exactly which criteria failed.

Provide specific, actionable feedback for improvements when marking as invalid.