# Profile Edit Prompt

You maintain a profile of the {entity_type} "{entity_name}". A new article has arrived. Do not rewrite the profile. Instead, list the smallest set of edits that brings it up to date. Code will apply your edits to the existing text, and every sentence you don't mention stays exactly as it is.

## Edits you can make

- `add`: insert new sentences. Set `anchor` to an existing sentence copied exactly from the profile (the new text goes right after it), or to a section heading such as `## Legal Proceedings` (the new text goes at the end of that section), or to `END` (the new text goes at the end of the profile; start it with a new `## Heading` if it belongs in a new section).
- `replace`: change one existing sentence. Set `anchor` to that sentence, copied exactly, and `text` to its new version. Use this only to sharpen a sentence the article makes more precise. Keep every citation the old sentence had.

There is no delete. Information already in the profile stays.

## Before writing edits

Fill in two lists first:

- `disagreements`: compare every date, number, name and outcome in the article that concerns {entity_name} with the profile. List one only when both give a different value for the very same thing, so that both can't be true: the same event on a different date, the same count with a different number. Copy the profile sentence exactly and say what the article says instead. A different year for something the profile already dates counts, even if the difference looks small.

  These are not disagreements, so leave them out (most go in `new_facts`):
  - two different events or steps, such as charges filed in one month and a trial ordered in another;
  - an older article saying something hadn't happened yet;
  - detail the profile doesn't have;
  - an approximate number against an exact one ("about 400" and "391");
  - two descriptions that could both be true, such as two roles.

  When in doubt, it is not a disagreement. Most articles have none, so this list is usually empty. Never add an entry just to say there is no conflict.
- `new_facts`: the facts about {entity_name} in the article that the profile doesn't have yet. When the article is mainly about {entity_name}, there is usually at least one.

Then write `edits` that cover every item in both lists: one `add` per disagreement (rule 3) and adds for the new facts.

## Rules

1. Only add facts about {entity_name}. Most articles are about something else and mention {entity_name} in passing: then add at most a sentence or two, or return no edits at all. Don't copy in background about other people, places or events.
2. Every sentence you add or change ends with the citation `^[{article_id}]`, plus any citations it already had. Never change or remove an existing citation, and never cite an id that isn't in the profile or this article.
3. For each entry in `disagreements`, and only for those, don't pick a side and don't overwrite the existing sentence. `add` a sentence right after it that gives the article's version with its citation and says the sources disagree, for example: "A later report gives the year as 2003 instead.^[{article_id}]" Never write that sources differ or disagree anywhere else.
4. When the article updates something that has genuinely changed (a new trial date, a new role), `add` the new state with its date and citation; keep the old sentence as history.
5. Copy anchors character for character, including their citations, so code can find them.
6. If both lists are empty, return an empty list of edits.
