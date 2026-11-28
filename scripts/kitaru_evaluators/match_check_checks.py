"""Kitaru evaluator: a ``match_check`` step session's verdict against its label.

Scores a merge eval pair's step session (``scripts/kitaru_traces.py --step
match_check``), or a replay of it (``scripts/kitaru_step.py``), with
``match_results()`` in ``src/evals/kitaru_checks.py``: what the model
answered, and for a labelled pair whether that agrees (``match_ok``) and
which way it was wrong (``false_merge``, ``missed_merge``).

Like ``profile_update_checks.py`` it declares no dependencies, so the worker
runs it on its own Python: start the worker with ``just kitaru-worker``.
Register it once (scoped to the step agent)::

    uvx --from 'kitaru[cli]' kitaru evaluator register hinbox-match-check-checks \
        --script scripts/kitaru_evaluators/match_check_checks.py \
        --entrypoint evaluate --agent-id <hinbox-match-check agent id>

Then ``just kitaru-describe`` gives it the ``DESCRIPTION`` below, which the
dashboard shows on its page and beside experiment results.
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView

# What the Kitaru dashboard shows for this evaluator (`just kitaru-describe` sets it)
NAME = "hinbox-match-check-checks"
DESCRIPTION = """\
Scores one merge match check: are these two entities the same one?
is_match: the model said they are the same.
confidence: how sure it said it was, 0 to 1.
For a hand-labelled pair:
match_ok: the answer matches the label.
false_merge: it said the same for two different entities, so two pages would be wrongly joined.
missed_merge: it said different for one entity, so a duplicate page would be left.
"""


def evaluate(session: SessionView, **params: Any) -> List[EvaluationResult]:
    try:
        from src.evals.kitaru_checks import match_results
    except ImportError as e:
        raise RuntimeError(
            "Can't import Hinbox's checks: start the worker from the Hinbox "
            "checkout with `just kitaru-worker`"
        ) from e
    s = session.session
    inputs = s.inputs or {}
    # A step session holds its pair's label in its inputs, a replay too
    rows = match_results(s.outputs, inputs.get("labels") or s.metadata)
    return [EvaluationResult(**row) for row in rows]
