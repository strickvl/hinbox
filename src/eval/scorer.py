"""Score pipeline merge decisions against gold labels.

Loads trace events (stage=merge.decision) and compares them against
annotated gold labels to produce precision/recall/F1 metrics.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel

from .candidates import candidate_key_from_names
from .gold_labels import GoldLabel, load_gold_labels


class EvalResult(BaseModel):
    """Confusion matrix and derived metrics for merge evaluation."""

    true_positives: int = 0  # pipeline=merge, gold=yes
    false_positives: int = 0  # pipeline=merge, gold=no
    true_negatives: int = 0  # pipeline=skip, gold=no
    false_negatives: int = 0  # pipeline=skip, gold=yes
    skipped_unsure: int = 0
    skipped_no_gold: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def total_evaluated(self) -> int:
        return (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )


def _is_merge_decision(decision: str) -> bool:
    """Check if a pipeline decision represents a merge."""
    return decision.lower() in ("merge", "merged", "yes")


def _is_skip_decision(decision: str) -> bool:
    """Check if a pipeline decision represents a skip/no-merge."""
    return decision.lower() in ("skip", "skipped", "no", "new")


def score_decisions(
    decisions: List[Dict],
    gold: Dict[str, GoldLabel],
) -> EvalResult:
    """Score a list of merge decision dicts against gold labels.

    Each decision dict should have keys: entity_name, candidate_name,
    entity_type, decision.

    Args:
        decisions: List of dicts with merge decision info.
        gold: Dict of gold labels keyed by candidate_key.

    Returns:
        EvalResult with confusion matrix counts.
    """
    result = EvalResult()

    for d in decisions:
        entity_name = str(d.get("entity_name", ""))
        candidate_name = str(d.get("candidate_name", ""))
        entity_type = str(d.get("entity_type", ""))
        decision = str(d.get("decision", ""))

        if not entity_name or not candidate_name or not entity_type:
            continue

        key = candidate_key_from_names(entity_name, candidate_name, entity_type)
        gold_label = gold.get(key)

        if gold_label is None:
            result.skipped_no_gold += 1
            continue

        if gold_label.label == "unsure":
            result.skipped_unsure += 1
            continue

        pipeline_merged = _is_merge_decision(decision)
        pipeline_skipped = _is_skip_decision(decision)
        gold_yes = gold_label.label == "yes"

        if not pipeline_merged and not pipeline_skipped:
            # Unknown decision type — skip
            result.skipped_no_gold += 1
            continue

        if pipeline_merged and gold_yes:
            result.true_positives += 1
        elif pipeline_merged and not gold_yes:
            result.false_positives += 1
        elif pipeline_skipped and not gold_yes:
            result.true_negatives += 1
        elif pipeline_skipped and gold_yes:
            result.false_negatives += 1

    return result


def score_run(
    domain: str,
    run_id: Optional[str] = None,
) -> EvalResult:
    """Score a trace run's merge decisions against gold labels.

    Loads trace events with stage=merge.decision from the specified run
    (or latest run) and compares against gold labels.

    Args:
        domain: Domain name (e.g. "guantanamo").
        run_id: Specific run ID, or None for latest.

    Returns:
        EvalResult with confusion matrix and derived metrics.
    """
    # Import trace helpers lazily to avoid circular deps
    from scripts.traces import (
        _filter_events,
        _load_events,
        _resolve_run_file,
        _resolve_runs_dir,
    )

    runs_dir = _resolve_runs_dir(domain)
    run_file = _resolve_run_file(runs_dir, run_id)

    if run_file is None:
        return EvalResult()

    events = _load_events(run_file)
    merge_events = _filter_events(events, stage="merge.decision")
    gold = load_gold_labels(domain)

    return score_decisions(merge_events, gold)
