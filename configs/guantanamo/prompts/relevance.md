# Relevance Filtering Prompt

You are tasked with determining if an article is relevant to Guantánamo Bay: the detention operation there, and the U.S. naval base that hosts it.

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
- What former Guantánamo detainees did after their release or transfer
- The U.S. Naval Station at Guantánamo Bay itself, even with no mention of detainees: its commanders (including their later careers, misconduct or prosecution), the troops, sailors and Coast Guard units stationed there or sent through it, base operations, and day-to-day life and community on the base
- Relations between the base and the surrounding Cuban city and province of Guantánamo

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