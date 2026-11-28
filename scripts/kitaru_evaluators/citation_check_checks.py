"""Kitaru evaluator: a citation check's verdicts against the hand labels,
with its time, tokens and cost.

Scores a ``grounding`` step session (``just citecheck kitaru-sessions``, or
``scripts/kitaru_traces.py --step grounding``), or a replay of one by
``scripts/kitaru_step.py`` with another judge, with ``citation_results()`` in
``src/citecheck/kitaru_eval.py``: ``wrong_passed``, ``good_doubted``,
``agrees``, ``answered``, ``seconds``, ``input_tokens``, ``output_tokens`` and
``cost_usd``. Pass ``prices`` (model -> [dollars per million input tokens,
per million output tokens]) as a parameter to price a model Kitaru has no
price for, such as a Tinker checkpoint, or a local one at 0.

Like the other Hinbox evaluators it declares no dependencies, so the worker
runs it on its own Python: start the worker with ``just kitaru-worker``.
Register it once (scoped to the step agent)::

    uvx --from 'kitaru[cli]' kitaru evaluator register hinbox-citation-check-checks \
        --script scripts/kitaru_evaluators/citation_check_checks.py \
        --entrypoint evaluate --agent-id <hinbox-citation-check agent id>

Then ``just kitaru-describe`` gives it the ``DESCRIPTION`` below, which the
dashboard shows on its page and beside experiment results.
"""

from typing import Any, List

from kitaru.task.evaluator import EvaluationResult, SessionView

# What the Kitaru dashboard shows for this evaluator (`just kitaru-describe` sets it)
NAME = "hinbox-citation-check-checks"
DESCRIPTION = """\
Scores one citation check against the hand labels (the answer key).
wrong_passed: a sentence labelled wrong that the checker called supported. This is the error that matters most; passes when it did not happen.
good_doubted: a sentence labelled right that the checker called partly or not supported; passes when it did not happen.
agrees: the checker's verdict matches the label.
answered: the checker gave a verdict for every sentence.
verdict: the checker's answer (supported, partial, not_supported, unclear).
seconds, input_tokens, output_tokens: how long the call took and how big it was.
cost_usd: dollars, from the prices parameter (dollars per million input and output tokens), else Kitaru's own price.
"""


def _seconds(s: Any) -> Any:
    if s.started_at and s.ended_at:
        return (s.ended_at - s.started_at).total_seconds()
    return None


def evaluate(session: SessionView, **params: Any) -> List[EvaluationResult]:
    try:
        from src.citecheck.kitaru_eval import citation_results
    except ImportError as e:
        raise RuntimeError(
            "Can't import Hinbox's checks: start the worker from the Hinbox "
            "checkout with `just kitaru-worker`"
        ) from e
    s = session.session
    base = getattr(session, "baseline", None)
    tokens = s.tokens
    rows = citation_results(
        s.inputs or {},
        s.outputs,
        seconds=_seconds(s),
        input_tokens=getattr(tokens, "input_tokens", None),
        output_tokens=getattr(tokens, "output_tokens", None),
        kitaru_cost=float(s.cost) if s.cost is not None else None,
        baseline_metadata=getattr(base, "metadata", None) if base else None,
        prices=params.get("prices"),
    )
    return [EvaluationResult(**row) for row in rows]
