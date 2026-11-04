# Fact Extraction Prompt (fact-list updater)

You keep a list of sourced facts about the {entity_type} "{entity_name}". Read the article (ID: {article_id}) and list the facts it gives about {entity_name} itself, one per row. You are not shown the profile, only its section headings, so list every fact the article gives about {entity_name}, even ones that may already be known. Code compares each row with the known facts afterwards.

## What counts

Identify what the article is mainly about, then separate information about {entity_name} from information about other people, places, organizations, events, and background. A mention near the entity is not automatically a fact about it. A venue, setting, participant, witness, or associated person does not make the surrounding story a fact about the entity. But what {entity_name} itself does, says, decides or has done to it in this article is a fact about {entity_name}, even when the story is mainly about someone else: a judge's ruling or recess, a lawyer's argument, a doctor's opinion, an official's statement, a prisoner's interrogation. For an event, include facts only when the article explicitly describes that event, not a different incident at the same place or time.

First set `mainly_about`: true when a main part of the article reports something about {entity_name}: something it did, said or decided, something done to it, or a plan, ruling, request or change about it (a request to replace a camp, motions put to a judge, a commander testifying about his camp), even when the article also covers other subjects. False only when {entity_name} is the setting, a name in a list, or background to a story about something else. If it is only mentioned in passing, still list what the article says {entity_name} itself did, said or decided; code keeps at most three of those that the profile doesn't already have, and drops background. Return an empty list only when the article says nothing about {entity_name} itself (a name in a list of others, a place named only as the setting).

## Each row

List the article's news about {entity_name} first, starting with the main thing the article reports about it, then any background.

- `claim`: one short sentence that states the fact on its own, naming {entity_name} rather than "he" or "it", with any date, number or place the article gives. Keep who said it when the article attributes it ("according to his lawyer", "the Pentagon said"). No citation.
- `section`: the heading of the profile section it belongs under, copied exactly from the list below, or a new `## Heading` if none fits. Reuse an existing heading whenever one is close.
- `quote`: the words in the article the claim rests on, copied exactly (one sentence or a part of one). Code checks that the quote is in the article, and drops rows whose quote isn't.
- `news`: true when the article reports this as new: something done, said, decided, found or alleged at the time of the article (a ruling, a request, testimony, a transfer, a statement). False for background the article recaps or describes (when a camp opened, what a prison is like, a person's age or nationality, an old event retold for context).
- `source`: who the article says this comes from, in a few words: "court filing", "his lawyer", "Pentagon statement", "eyewitness", "the reporter". Use "the reporter" when the article states it without attribution.

Profile section headings:
{headings}
