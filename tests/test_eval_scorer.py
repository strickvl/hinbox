"""Tests for merge evaluation scorer — confusion matrix, metrics, edge cases."""

from src.eval.candidates import candidate_key_from_names
from src.eval.gold_labels import GoldLabel
from src.eval.scorer import EvalResult, score_decisions


def _make_gold(name_a: str, name_b: str, entity_type: str, label: str) -> GoldLabel:
    """Create a gold label for testing."""
    return GoldLabel(
        entity_name=name_a,
        candidate_name=name_b,
        entity_type=entity_type,
        label=label,
        timestamp="2025-01-01T00:00:00Z",
    )


def _make_decision(name_a: str, name_b: str, entity_type: str, decision: str) -> dict:
    """Create a pipeline decision dict for testing."""
    return {
        "entity_name": name_a,
        "candidate_name": name_b,
        "entity_type": entity_type,
        "decision": decision,
    }


def _gold_dict(*labels):
    """Build a gold dict from a list of GoldLabel objects."""
    result = {}
    for gl in labels:
        key = candidate_key_from_names(
            gl.entity_name, gl.candidate_name, gl.entity_type
        )
        result[key] = gl
    return result


# ── Confusion matrix math ──────────────────────────────────


def test_perfect_score():
    """All correct predictions should give perfect metrics."""
    gold = _gold_dict(
        _make_gold("A", "B", "person", "yes"),
        _make_gold("C", "D", "person", "no"),
    )
    decisions = [
        _make_decision("A", "B", "person", "merge"),
        _make_decision("C", "D", "person", "skip"),
    ]
    result = score_decisions(decisions, gold)
    assert result.true_positives == 1
    assert result.true_negatives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_all_wrong():
    """All incorrect predictions should give zero metrics."""
    gold = _gold_dict(
        _make_gold("A", "B", "person", "no"),
        _make_gold("C", "D", "person", "yes"),
    )
    decisions = [
        _make_decision("A", "B", "person", "merge"),
        _make_decision("C", "D", "person", "skip"),
    ]
    result = score_decisions(decisions, gold)
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_mixed_results():
    """Mixed predictions should produce correct precision/recall."""
    gold = _gold_dict(
        _make_gold("A", "B", "person", "yes"),
        _make_gold("C", "D", "person", "yes"),
        _make_gold("E", "F", "person", "no"),
    )
    decisions = [
        _make_decision("A", "B", "person", "merge"),  # TP
        _make_decision("C", "D", "person", "skip"),  # FN
        _make_decision("E", "F", "person", "merge"),  # FP
    ]
    result = score_decisions(decisions, gold)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    # precision = 1/(1+1) = 0.5
    assert abs(result.precision - 0.5) < 1e-9
    # recall = 1/(1+1) = 0.5
    assert abs(result.recall - 0.5) < 1e-9
    # f1 = 2*0.5*0.5/(0.5+0.5) = 0.5
    assert abs(result.f1 - 0.5) < 1e-9


# ── Unsure exclusion ───────────────────────────────────────


def test_unsure_labels_excluded():
    """Pairs labelled 'unsure' should be excluded from scoring."""
    gold = _gold_dict(
        _make_gold("A", "B", "person", "unsure"),
        _make_gold("C", "D", "person", "yes"),
    )
    decisions = [
        _make_decision("A", "B", "person", "merge"),
        _make_decision("C", "D", "person", "merge"),
    ]
    result = score_decisions(decisions, gold)
    assert result.skipped_unsure == 1
    assert result.true_positives == 1
    assert result.total_evaluated == 1


# ── Missing gold labels ───────────────────────────────────


def test_no_gold_skipped():
    """Decisions without matching gold labels should be skipped."""
    gold = _gold_dict()
    decisions = [
        _make_decision("A", "B", "person", "merge"),
    ]
    result = score_decisions(decisions, gold)
    assert result.skipped_no_gold == 1
    assert result.total_evaluated == 0


# ── Name normalization / order independence ────────────────


def test_decision_name_order_matches_gold():
    """Decision names in reverse order should still match gold labels."""
    gold = _gold_dict(
        _make_gold("Alice", "Bob", "person", "yes"),
    )
    # Decision has names in reversed order
    decisions = [
        _make_decision("Bob", "Alice", "person", "merge"),
    ]
    result = score_decisions(decisions, gold)
    assert result.true_positives == 1


def test_case_insensitive_matching():
    """Name matching should be case-insensitive."""
    gold = _gold_dict(
        _make_gold("alice smith", "BOB JONES", "person", "yes"),
    )
    decisions = [
        _make_decision("Alice Smith", "Bob Jones", "person", "merge"),
    ]
    result = score_decisions(decisions, gold)
    assert result.true_positives == 1


# ── Empty inputs ───────────────────────────────────────────


def test_empty_decisions():
    """Scoring with no decisions should return zeroes."""
    gold = _gold_dict(
        _make_gold("A", "B", "person", "yes"),
    )
    result = score_decisions([], gold)
    assert result.total_evaluated == 0
    assert result.precision == 0.0
    assert result.f1 == 0.0


# ── EvalResult property edge cases ─────────────────────────


def test_eval_result_zero_denominator():
    """Metrics with zero denominators should return 0.0, not error."""
    result = EvalResult()
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.total_evaluated == 0


# ── Decision synonym handling ──────────────────────────────


def test_merged_synonym():
    """'merged' should be treated as a merge decision."""
    gold = _gold_dict(_make_gold("A", "B", "person", "yes"))
    decisions = [_make_decision("A", "B", "person", "merged")]
    result = score_decisions(decisions, gold)
    assert result.true_positives == 1


def test_new_synonym():
    """'new' should be treated as a skip/no-merge decision."""
    gold = _gold_dict(_make_gold("A", "B", "person", "no"))
    decisions = [_make_decision("A", "B", "person", "new")]
    result = score_decisions(decisions, gold)
    assert result.true_negatives == 1
