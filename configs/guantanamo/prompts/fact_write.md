# Section Write Prompt (fact-list updater)

You maintain the section "{section}" of a profile of the {entity_type} "{entity_name}". Some new facts belong in this section. For each one, write a sentence that fits the section and say where it goes. Code inserts your sentences; every existing sentence stays exactly as it is.

## For each sentence

- `facts`: the numbers of the facts the sentence states. Join facts about the same thing into one sentence (a statement and who made it, an outcome and its date); keep unrelated facts apart.
- `after`: the existing sentence of the section the new sentence should follow, copied exactly with its citations, so the section stays in order (usually by date). Use `END` to put it at the end of the section.
- `sentence`: one sentence in the section's style. Where the section has already named {entity_name}, refer to it the way the section does (a short name, "he", "the group") rather than repeating the full name. Keep every date, number, name and attribution the fact gives, and add nothing it doesn't say. End it with the citation shown for the fact, such as `^[{article_id}]`, and cite nothing else.

Every fact must be in exactly one sentence.
