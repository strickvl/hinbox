"""Binary classification metrics for the scoreboard.

Gold sets are lopsided (e.g. 132 relevant vs 20 irrelevant articles), so plain
accuracy flatters a model that always gives the common answer. Balanced
accuracy averages the hit rate on each class, so "always relevant" scores 0.5
no matter how lopsided the set is. Errors (no prediction) count as wrong.
"""

from typing import Dict, List, Optional, Tuple


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_metrics(pairs: List[Tuple[bool, Optional[bool]]]) -> Dict[str, float]:
    """Score ``(gold, predicted)`` pairs; ``predicted=None`` means the call failed."""
    tp = sum(1 for gold, pred in pairs if gold and pred is True)
    fn = sum(1 for gold, pred in pairs if gold and pred is not True)
    tn = sum(1 for gold, pred in pairs if not gold and pred is False)
    fp = sum(1 for gold, pred in pairs if not gold and pred is not False)
    errors = sum(1 for _, pred in pairs if pred is None)
    n = len(pairs)

    recall_pos = _safe_div(tp, tp + fn)
    recall_neg = _safe_div(tn, tn + fp)
    precision_pos = _safe_div(tp, tp + fp)
    positives = tp + fn
    return {
        "n": n,
        "errors": errors,
        "accuracy": _safe_div(tp + tn, n),
        "balanced_accuracy": (recall_pos + recall_neg) / 2,
        "precision": precision_pos,
        "recall": recall_pos,
        "f1": _safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos),
        "true_negative_rate": recall_neg,
        "majority_baseline": _safe_div(max(positives, n - positives), n),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
