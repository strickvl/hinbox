"""Tests for MergeCandidate model, candidate key generation, and seed logic."""

from src.eval.candidates import (
    MergeCandidate,
    candidate_key,
    candidate_key_from_names,
    load_candidates,
    save_candidates,
)
from src.eval.gold_labels import (
    append_gold_label,
    load_gold_labels,
    make_gold_label,
)

# ── candidate_key ──────────────────────────────────────────


def test_candidate_key_order_independent():
    """A-vs-B should produce the same key as B-vs-A."""
    c1 = MergeCandidate(
        entity_name="Alice Smith",
        candidate_name="Bob Jones",
        entity_type="person",
        source="seed",
    )
    c2 = MergeCandidate(
        entity_name="Bob Jones",
        candidate_name="Alice Smith",
        entity_type="person",
        source="seed",
    )
    assert candidate_key(c1) == candidate_key(c2)


def test_candidate_key_case_insensitive():
    """Keys should be case-insensitive."""
    c1 = MergeCandidate(
        entity_name="ALICE SMITH",
        candidate_name="bob jones",
        entity_type="person",
        source="seed",
    )
    c2 = MergeCandidate(
        entity_name="alice smith",
        candidate_name="Bob Jones",
        entity_type="person",
        source="seed",
    )
    assert candidate_key(c1) == candidate_key(c2)


def test_candidate_key_different_types_differ():
    """Same names but different entity types should produce different keys."""
    c1 = MergeCandidate(
        entity_name="Alpha",
        candidate_name="Beta",
        entity_type="person",
        source="seed",
    )
    c2 = MergeCandidate(
        entity_name="Alpha",
        candidate_name="Beta",
        entity_type="organization",
        source="seed",
    )
    assert candidate_key(c1) != candidate_key(c2)


def test_candidate_key_strips_whitespace():
    """Leading/trailing whitespace should be ignored."""
    c1 = MergeCandidate(
        entity_name="  Alice  ",
        candidate_name="Bob",
        entity_type="person",
        source="seed",
    )
    c2 = MergeCandidate(
        entity_name="Alice",
        candidate_name="Bob",
        entity_type="person",
        source="seed",
    )
    assert candidate_key(c1) == candidate_key(c2)


def test_candidate_key_from_names_matches_model():
    """The convenience function should match the model-based function."""
    c = MergeCandidate(
        entity_name="Alice",
        candidate_name="Bob",
        entity_type="person",
        source="seed",
    )
    assert candidate_key(c) == candidate_key_from_names("Alice", "Bob", "person")


# ── load/save round-trip ───────────────────────────────────


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """Candidates should survive a save→load round-trip."""
    # Patch _eval_dir to use tmp_path
    monkeypatch.setattr(
        "src.eval.candidates._eval_dir",
        lambda domain: tmp_path,
    )
    candidates = [
        MergeCandidate(
            entity_name="Alice",
            candidate_name="Bob",
            entity_type="person",
            source="seed",
            similarity_score=72.5,
            entity_context="context A",
        ),
        MergeCandidate(
            entity_name="Org One",
            candidate_name="Org Two",
            entity_type="organization",
            source="trace",
            pipeline_decision="merge",
            pipeline_confidence=0.91,
        ),
    ]
    save_candidates("test", candidates)
    loaded = load_candidates("test")
    assert len(loaded) == 2
    assert loaded[0].entity_name == "Alice"
    assert loaded[0].similarity_score == 72.5
    assert loaded[1].pipeline_decision == "merge"


def test_load_missing_file_returns_empty(tmp_path, monkeypatch):
    """Loading from a non-existent file should return empty list."""
    monkeypatch.setattr(
        "src.eval.candidates._eval_dir",
        lambda domain: tmp_path / "nonexistent",
    )
    assert load_candidates("test") == []


# ── gold label persistence ─────────────────────────────────


def test_gold_label_append_and_load(tmp_path, monkeypatch):
    """Appending labels should accumulate in the file."""
    monkeypatch.setattr(
        "src.eval.gold_labels._eval_dir",
        lambda domain: tmp_path,
    )
    gl1 = make_gold_label("Alice", "Bob", "person", "yes")
    gl2 = make_gold_label(
        "Org A", "Org B", "organization", "no", notes="clearly different"
    )

    append_gold_label("test", gl1)
    append_gold_label("test", gl2)

    labels = load_gold_labels("test")
    assert len(labels) == 2
    keys = list(labels.keys())
    assert labels[keys[0]].label in ("yes", "no")
    assert labels[keys[1]].label in ("yes", "no")


def test_gold_label_last_write_wins(tmp_path, monkeypatch):
    """Labelling the same pair twice should keep only the last label."""
    monkeypatch.setattr(
        "src.eval.gold_labels._eval_dir",
        lambda domain: tmp_path,
    )
    gl1 = make_gold_label("Alice", "Bob", "person", "yes")
    gl2 = make_gold_label("Alice", "Bob", "person", "no", notes="changed my mind")

    append_gold_label("test", gl1)
    append_gold_label("test", gl2)

    labels = load_gold_labels("test")
    # Only one key for this pair
    key = candidate_key_from_names("Alice", "Bob", "person")
    assert key in labels
    assert labels[key].label == "no"
    assert labels[key].notes == "changed my mind"


def test_gold_label_order_independent(tmp_path, monkeypatch):
    """Labelling A-vs-B and B-vs-A should map to the same key."""
    monkeypatch.setattr(
        "src.eval.gold_labels._eval_dir",
        lambda domain: tmp_path,
    )
    gl1 = make_gold_label("Alice", "Bob", "person", "yes")
    gl2 = make_gold_label("Bob", "Alice", "person", "no")

    append_gold_label("test", gl1)
    append_gold_label("test", gl2)

    labels = load_gold_labels("test")
    key = candidate_key_from_names("Alice", "Bob", "person")
    assert labels[key].label == "no"  # last write wins
