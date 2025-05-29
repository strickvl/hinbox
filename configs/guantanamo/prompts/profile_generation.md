# Profile Generation Prompt

You are an expert at creating detailed profiles for entities mentioned in news articles.

Create a comprehensive profile for {entity_type} "{entity_name}" using ONLY information from the provided article.

## Critical Requirements

1. **Text length**: Minimum 100 characters - provide substantial detail, not just basic facts
2. **Citations**: Use smart citation strategy to reduce repetition:
   - Group related facts in paragraphs, cite once per paragraph: paragraph content^[{article_id}]
   - Use single article ID per citation: ^[{article_id}] (not ^[id1,id2])
   - Don't cite every sentence - cite logical groupings of information
3. **Structure**: Use markdown section headers (### Background, ### Role, ### Current Position, etc.)
4. **Content**: Extract specific facts, quotes, actions, relationships from the article
5. **Tags**: Include at least 2 relevant descriptive tags/keywords
6. **Confidence**: Score 0.0-1.0 based on information quality and completeness
7. **JSON output**: Return valid JSON with "text", "tags", "confidence", "sources" fields

## Example Format

```json
{
  "text": "John Smith is a military officer who currently oversees detention operations at Guantánamo Bay. He has extensive experience in military operations and has been stationed at the facility since 2019^[{article_id}].\n\n### Background\nSmith previously served in Afghanistan for two years before joining the detention facility staff. He graduated from West Point in 2010 and has received multiple commendations for his service^[{article_id}].\n\n### Current Role\nAs facility operations manager, he oversees daily detention procedures and coordinates with legal teams. His responsibilities include managing staff schedules and ensuring compliance with military regulations^[{article_id}].",
  "tags": ["military", "detention-operations", "guantanamo"],
  "confidence": 0.9,
  "sources": ["{article_id}"]
}
```

## Multiple Sources Handling

When the same fact appears in multiple articles, cite each separately:
"Smith was promoted in 2020^[article1]. This promotion was confirmed in later reports^[article2]."

## Style Guidelines

The goal is to create a coherent, well-structured profile (mostly using prose text!) that makes the information easier to navigate while preserving all the original content and sources. Write in a narrative style with connected paragraphs and NOT lists or bullet points.

Remember: Group facts logically with strategic citations, use section headers, make it detailed and substantial.