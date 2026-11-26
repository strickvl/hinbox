"""Kitaru evaluator: verdict and passage checks on one ``grounding`` step session.

Scores a grounding step session (``scripts/kitaru_traces.py --step grounding``),
or a replay of it (``scripts/kitaru_step.py``), with ``grounding_results()``
in ``src/evals/kitaru_checks.py``: completion, missing verdicts, verdict
counts and shares, verified quote rate, and label agreement when expected
levels are provided.

Like ``profile_update_checks.py`` it declares no dependencies, so the worker
runs it on its own Python: start the worker with ``just kitaru-worker``.
Register it once (scoped to the step agent)::

    uvx --from 'kitaru[cli]' kitaru evaluator register hinbox-grounding-checks \
        --script scripts/kitaru_evaluators/grounding_checks.py \
        --entrypoint evaluate --agent-id <hinbox-grounding agent id>
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView


def evaluate(session: SessionView, **params: Any) -> List[EvaluationResult]:
    try:
        from src.evals.kitaru_checks import grounding_results
    except ImportError as e:
        raise RuntimeError(
            "Can't import Hinbox's checks: start the worker from the Hinbox "
            "checkout with `just kitaru-worker`"
        ) from e
    s = session.session
    inputs = s.inputs or {}
    rows = grounding_results(inputs, s.outputs, inputs.get("labels") or s.metadata)
    return [EvaluationResult(**row) for row in rows]
