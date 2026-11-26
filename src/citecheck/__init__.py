"""Data, labels and scores for a tuned local citation checker.

The pipeline's citation check asks a model, for each cited profile sentence,
whether the cited article says it (``src/utils/grounding.py``). Local Qwen
lets through 19 of 42 hand-labelled wrong claims, Claude Haiku 2 and
Inkling 2 Mini 6, but Inkling takes about 40 times as long as Haiku because
it writes thousands of reasoning tokens per call. This package builds what a
fine-tune of a small local model needs, without changing the pipeline:

- ``items.py``: every cited sentence of a finished run, each article put in
  train, tune or test by a hash of its id, and a test set drawn from test
  articles only, for a person to label yes or no.
- ``label_app.py``: a local page for those yes/no labels (and for auditing
  teacher labels); nothing leaves the machine.
- ``planted.py``: sentences copied from an article with one detail changed
  in code (a year, a number, a name, a negation, a dropped "allegedly"), so
  the right answer is known without any judge.
- ``teach.py``: verdicts from two strong cloud judges on train and tune
  articles, kept where they agree.
- ``export.py``: chat-format training files in the pipeline's own judge
  messages and reply format, so a tuned model drops in as
  ``HINBOX_GROUNDING_MODEL``.
- ``score.py``: a judge (a model, a run's stored verdicts, or predictions
  from a tuned model on Tinker) against the hand labels: wrong claims let
  through and good claims doubted, each with a 95% range.

``python -m src.citecheck --help`` (``just citecheck``) runs them. Every file
these steps write holds article text or sentences from it: it belongs under
``data/`` (gitignored), never on GitHub.
"""
