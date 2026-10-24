# Profile Edit Prompt

You maintain a profile of the {entity_type} "{entity_name}". A new article has arrived. Do not rewrite the profile. Instead, list the smallest set of edits that brings it up to date. Code will apply your edits to the existing text, and every sentence you don't mention stays exactly as it is.

## Edits you can make

- `add`: insert new sentences. Set `anchor` to an existing sentence copied exactly from the profile (the new text goes right after it), or to a section heading such as `## Legal Proceedings` (the new text goes at the end of that section), or to `END` (the new text goes at the end of the profile; start it with a new `## Heading` if it belongs in a new section).
- `replace`: change one existing sentence. Set `anchor` to that sentence, copied exactly, and `text` to its new version. Use this only to sharpen a sentence the article makes more precise. Keep every citation the old sentence had.

There is no delete. Information already in the profile stays.

## Rules

1. Only add facts about {entity_name}. Most articles are about something else and mention {entity_name} in passing: then add at most a sentence or two, or return no edits at all. Don't copy in background about other people, places or events.
2. Every sentence you add or change ends with the citation `^[{article_id}]`, plus any citations it already had. Never change or remove an existing citation, and never cite an id that isn't in the profile or this article.
3. When the article contradicts the profile (a different date, number, name or outcome), don't pick a side and don't overwrite the existing sentence. `add` a sentence right after it that gives the article's version with its citation and says the sources disagree, for example: "A later report gives the year as 2003 instead.^[{article_id}]"
4. When the article updates something that has genuinely changed (a new trial date, a new role), `add` the new state with its date and citation; keep the old sentence as history.
5. Copy anchors character for character, including their citations, so code can find them.
6. If nothing in the article is new about {entity_name}, return an empty list of edits.
