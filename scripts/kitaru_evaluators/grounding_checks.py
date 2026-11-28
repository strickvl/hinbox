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

Then ``just kitaru-describe`` gives it the ``DESCRIPTION`` below, which the
dashboard shows on its page and beside experiment results.
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView

# What the Kitaru dashboard shows for this evaluator (`just kitaru-describe` sets it)
NAME = "hinbox-grounding-checks"
DESCRIPTION = """\
Checks one citation-check step: one article against the profile sentences that cite it.
grounding_ok: the step finished and every sentence got a verdict.
missing_verdicts: sentences with no verdict.
supported, partial, not_supported, unclear: how many sentences got each verdict.
supported_share: share called supported. accepted_share: share called supported or partial.
passage_rate: share of verdicts quoting a passage found in the article.
label_agreement: share of verdicts matching the expected ones, when the session has them.
"""


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
