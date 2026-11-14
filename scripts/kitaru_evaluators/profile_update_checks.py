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
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView


def evaluate(session: SessionView, **params: Any) -> List[EvaluationResult]:
    try:
        from src.evals.kitaru_checks import step_results
    except ImportError as e:
        raise RuntimeError(
            "Can't import Hinbox's checks: start the worker from the Hinbox "
            "checkout with `uv run --extra kitaru kitaru worker start`"
        ) from e
    s = session.session
    rows = step_results(s.inputs or {}, s.outputs, s.metadata)
    return [EvaluationResult(**row) for row in rows]
