# Date Check Prompt (fact-list updater)

You keep a list of sourced facts about the {entity_type} "{entity_name}". A new article mentions dates near facts the list already has. Each pair below puts one known fact beside one sentence from the article that gives a different date and uses some of the same words. Say what each pair is.

## Verdicts

- `conflict`: both describe the very same completed event (an arrival, a capture, a bombing, a transfer, charges being filed, a plea, a sentence, a camp opening) and give it different dates, so both can't be true. Write `sentence`: the article's version as one sentence about {entity_name}, keeping the article's date and nothing the article doesn't say. No citation.
- `different`: two different events or steps, even if they are related: charges sworn in one year and approved in another, a first visit and a later one, a hearing and the trial it led to, or the article's own publication date against an event date.
- `later`: a plan, schedule, estimate, status or count that changed over time, so both were true when said: a proposed trial date and a later estimate, a deadline that moved, "still deciding" and then a decision, the number held at two different times.

Most pairs are `different` or `later`. Use `conflict` only when you can name the one event both sentences date, and each date is stated as when that event happened. Give one verdict per pair, using the pair's number.
