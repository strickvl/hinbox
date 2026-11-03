# Fact Comparison Prompt (fact-list updater)

You keep a list of sourced facts about the {entity_type} "{entity_name}". A new article gave the claims below. For each claim, code has found the few known facts most likely to be about the same thing. Say how each claim relates to them.

## Verdicts

- `same`: one of the known facts already says this, possibly in other words. Give its id in `row`. Set `new_evidence` to true only when the claim comes from a different kind of source than the known fact and that matters to a historian: a court record, official document or first-hand witness where the known fact rested on a secondhand report, or the reverse. A second newspaper repeating the same thing is not new evidence.
- `conflict`: one known fact gives a different value for the very same thing, so both can't be true: the same event on a different date, the same count with a different number. Give its id in `row`.
- `new`: none of the known facts says this. Also use `new` when the claim adds real detail (a date, an outcome, a new step) to a known fact.

These are not conflicts (most are `new`):
- two different events or steps, such as charges filed in one month and a trial ordered in another;
- an older report saying something hadn't happened yet;
- a later development, such as a new role, a transfer, or a rescheduled hearing;
- an approximate number against an exact one ("about 400" and "391");
- two descriptions that could both be true.

When in doubt, it is not a conflict. Give one verdict per claim, using the claim's number.
