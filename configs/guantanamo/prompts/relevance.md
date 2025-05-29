# Relevance Filtering Prompt

You are tasked with determining if an article is relevant to Guantánamo Bay detention issues.

## Relevance Criteria

An article is considered relevant if it discusses:

- Guantánamo Bay detention facility or detainees
- Military commissions or tribunals at Guantánamo
- Detention policies related to the War on Terror
- Legal cases involving Guantánamo detainees
- Conditions at Guantánamo Bay
- Torture or interrogation at Guantánamo or related facilities
- Transfer or release of Guantánamo detainees
- Government policies regarding Guantánamo Bay

## Guidelines

The article should have substantial content about these topics, not just passing mentions.

## Output Format

You must return a boolean decision with reasoning:

```json
{
  "is_relevant": true,
  "reason": "Brief explanation of why the article is or is not relevant"
}
```