"""Kitaru evaluator: the profile-update eval's model-free checks on one step session.

Scores the session's updated profile (outputs) against the old one (inputs)
with ``src/evals/kitaru_checks.py``: collapse, kept facts, invented citations
and the reported counts. Works on imported step sessions and on replays of
them (``scripts/kitaru_step.py``).

It declares no dependencies, so the worker runs it on its own Python. Start
the worker from the Hinbox checkout, in Hinbox's environment, so ``src`` is
importable::

    uv run --extra kitaru kitaru worker start

Register it once (scoped to the step agent)::

    uvx --from 'kitaru[cli]' kitaru evaluator register hinbox-profile-update-checks \
        --script scripts/kitaru_evaluators/profile_update_checks.py \
        --entrypoint evaluate --agent-id <hinbox-profile-update agent id>

Then ``just kitaru-describe`` gives it the ``DESCRIPTION`` below, which the
dashboard shows on its page and beside experiment results.
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView

# What the Kitaru dashboard shows for this evaluator (`just kitaru-describe` sets it)
NAME = "hinbox-profile-update-checks"
DESCRIPTION = """\
Checks one profile update without a model, by comparing the updated profile with the old one.
update_ok: the profile did not collapse, kept its cited sentences and invented no citations.
length_ratio: new length over old; far below 1 means the profile collapsed.
fact_retention: share of the old cited sentences still there.
invented_citations: citations to articles the profile does not have.
unchanged_share: share of old sentences kept word for word.
added_sentences: new sentences with no close match in the old profile.
cites_article: the new article is cited. updated: it joined the profile's sources.
dispute_claims: new "sources disagree" sentences.
unsupported_numbers: new sentences citing the article with numbers it lacks.
case_ok: for an eval case, the profile-update eval's own pass verdict.
"""


def evaluate(session: SessionView, **params: Any) -> List[EvaluationResult]:
    try:
        from src.evals.kitaru_checks import step_results
    except ImportError as e:
        raise RuntimeError(
            "Can't import Hinbox's checks: start the worker from the Hinbox "
            "checkout with `uv run --extra kitaru kitaru worker start`"
        ) from e
    s = session.session
    inputs = s.inputs or {}
    # A step session holds its eval case's labels in its inputs, a replay too
    rows = step_results(inputs, s.outputs, inputs.get("labels") or s.metadata)
    return [EvaluationResult(**row) for row in rows]
