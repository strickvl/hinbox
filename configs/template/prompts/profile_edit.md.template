# Profile Edit Prompt

You maintain a profile of the {entity_type} "{entity_name}". A new article has arrived. Do not rewrite the profile. Return only the smallest set of edits needed to add genuinely new information about {entity_name}. Code applies the edits; every sentence you don't mention stays exactly as it is.

## Assess relevance before extracting facts

Identify what the article is mainly about, then separate information about {entity_name} from information about other people, places, organizations, events, and background. A mention near the entity is not automatically a fact about it. A venue, setting, participant, witness, or associated person does not make the surrounding story a fact about the entity. For an event in particular, include facts only when the article explicitly describes that event; do not fold in a different protest, incident, or activity at the same place merely because it occurred nearby or around the same time. Do not infer that distinct events are connected unless the article makes that connection.

If the article gives no new information about {entity_name}, return empty `disagreements`, `new_facts`, and `edits` lists. If the article is mainly about another subject, add at most one or two concise sentences in total, and only for directly relevant, genuinely new facts about {entity_name}. This is a ceiling, not a target: do not add unrelated context to use up the allowance. Prefer no edit over a tenuous connection.

## Before writing edits

Fill in two lists:

- `disagreements`: compare each date, number, name, and outcome in the article that concerns {entity_name} with the profile. Include an item only if both give incompatible values for the very same fact or event. Copy the conflicting profile sentence exactly and state the article's version. A different year for the same event counts. Do not treat these as disagreements: separate events or stages, a report that something had not happened yet at an earlier time, details absent from the profile, approximate versus exact figures, or descriptions that can both be true. When in doubt, leave it out; most articles have no disagreements. Never include an item just to say there is no conflict.
- `new_facts`: list only genuinely new facts about {entity_name} itself that the article supports and the profile does not already contain. Do not list facts about other subjects, incidental venue or setting details, facts about a different event, or paraphrases of what the profile already says.

Then write `edits` covering every item in both lists: one `add` per disagreement, plus minimal `add` operations for the new facts. Avoid redundant sentences and, where possible, combine closely related new facts into one concise sentence. If both lists are empty, `edits` must be empty.

## Edits you can make

- `add`: insert new sentence(s). Set `anchor` to an existing sentence copied exactly from the profile (the new text goes right after it), a section heading such as `## Legal Proceedings` (the new text goes at the end of that section), or `END` (the new text goes at the end of the profile; start it with a new `## Heading` if needed).
- `replace`: change one existing sentence, only when the article makes that same sentence more precise. Set `anchor` to the exact existing sentence and `text` to its new version. Keep every citation the old sentence had. Do not use a replace to resolve a disagreement or to overwrite history.

## Rules

1. Add only facts about {entity_name}. Keep edits narrowly relevant and minimal. Do not turn a passing mention into a summary of the article. For an article mainly about a different subject, do not add facts about the article's main subject, other people, or the location just because they are mentioned alongside {entity_name}. If {entity_name} is only the venue, setting, or a participant, add only a genuinely new fact about it; if there is none, make no edit.
2. Every sentence you add or change ends with the citation `^[{article_id}]`, plus any citations it already had. Never remove or alter an existing citation, and never cite an id that isn't in the profile or this article.
3. For each item in `disagreements`, and only for those, don't pick a side and don't overwrite the existing sentence. Add a sentence right after the conflicting one that names the fact, gives the article's value, cites the article, and says the sources disagree, for example: "This article dates the camp's opening to 2003 rather than 2002; the sources disagree.^[{article_id}]" The article's value must appear in the article. Never say the sources differ or disagree anywhere else.
4. When the article reports a genuine change over time (a new role, a transfer, a newly scheduled proceeding), add the new state with its date and citation; keep the old sentence as history. A later development is not a conflict with an earlier status report.
5. Copy anchors character for character, including their citations, so code can find them. An anchor can be used more than once, but don't produce redundant adds.
6. Keep `new_facts` and edits concise and factual. Do not infer, embellish, or introduce anything the article does not support. Every listed new fact gets an edit, and every edit corresponds to a listed disagreement or new fact.
