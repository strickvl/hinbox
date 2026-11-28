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

Then ``just kitaru-describe`` gives it the ``DESCRIPTION`` below, which the
dashboard shows on its page and beside experiment results.
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView

# What the Kitaru dashboard shows for this evaluator (`just kitaru-describe` sets it)
NAME = "hinbox-extract-checks"
DESCRIPTION = """\
Checks one extraction step (the people, organizations, locations or events found in one article) without a model.
extract_ok: the step finished without an error.
entity_count: entities kept. raw_count: entities the model returned.
deduped: duplicates merged by quality control. dropped_missing_required: entities dropped for missing fields.
retry_attempted: the step asked the model a second time.
qc_flags: quality-control flags raised.
mention_rate: share of the kept names that appear in the article.
"""


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
