"""Kitaru evaluator: health and QC checks on one ``extract.<type>`` step session.

Scores an extraction step session (``scripts/kitaru_traces.py --step extract``
or ``--step extract.<type>``), or a replay of it (``scripts/kitaru_step.py``),
with ``extraction_results()`` in ``src/evals/kitaru_checks.py``: completion,
entity/raw counts, QC dedups/drops/flags, retries, and mention rate in the
article.

Like ``profile_update_checks.py`` it declares no dependencies, so the worker
runs it on its own Python: start the worker with ``just kitaru-worker``.
Register it once (scoped to the step agent)::

    uvx --from 'kitaru[cli]' kitaru evaluator register hinbox-extract-checks \
        --script scripts/kitaru_evaluators/extraction_checks.py \
        --entrypoint evaluate --agent-id <hinbox-extract agent id>
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView


def evaluate(session: SessionView, **params: Any) -> List[EvaluationResult]:
    try:
        from src.evals.kitaru_checks import extraction_results
    except ImportError as e:
        raise RuntimeError(
            "Can't import Hinbox's checks: start the worker from the Hinbox "
            "checkout with `just kitaru-worker`"
        ) from e
    s = session.session
    inputs = s.inputs or {}
    rows = extraction_results(inputs, s.outputs, inputs.get("labels") or s.metadata)
    return [EvaluationResult(**row) for row in rows]
