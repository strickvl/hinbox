# Section Write Prompt (fact-list updater)

You maintain the section "{section}" of a profile of the {entity_type} "{entity_name}". Some new facts belong in this section. For each one, write a sentence that fits the section and say where it goes. Code inserts your sentences; every existing sentence stays exactly as it is.

## For each fact

- `fact`: the fact's number.
- `after`: the existing sentence of the section the new sentence should follow, copied exactly with its citations, so the section stays in order (usually by date). Use `END` to put it at the end of the section.
- `sentence`: the fact as one sentence in the profile's style. Keep every date, number, name and attribution the fact gives, and add nothing it doesn't say. End it with the citation shown for the fact, such as `^[{article_id}]`, and cite nothing else.

Write one entry per fact, even when two facts could share a sentence.
