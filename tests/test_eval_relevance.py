"""Tests for RelevanceCandidate/RelevanceLabel models and JSONL persistence."""

from src.eval.relevance_candidates import (
    RelevanceCandidate,
    load_relevance_candidates,
    save_relevance_candidates,
)
from src.eval.relevance_labels import (
    append_relevance_label,
    load_relevance_labels,
    make_relevance_label,
)

# ── RelevanceCandidate model ──────────────────────────────


def test_relevance_candidate_creation():
    """Basic model creation with required and optional fields."""
    c = RelevanceCandidate(
        article_id="abc-123",
        title="Test Article",
        content_snippet="First 600 chars of content...",
        url="https://example.com/article",
        published_date="2024-01-15",
        content_length=5000,
    )
    assert c.article_id == "abc-123"
    assert c.content_length == 5000
    assert c.url == "https://example.com/article"


def test_relevance_candidate_defaults():
    """Optional fields should have sensible defaults."""
    c = RelevanceCandidate(
        article_id="xyz",
        title="Minimal",
        content_snippet="snippet",
    )
    assert c.url is None
    assert c.published_date is None
    assert c.content_length == 0


# ── save/load round-trip ──────────────────────────────────


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    """Candidates should survive a save->load round-trip."""
    monkeypatch.setattr(
        "src.eval.relevance_candidates._eval_dir",
        lambda domain: tmp_path,
    )
    candidates = [
        RelevanceCandidate(
            article_id="id-1",
            title="Article One",
            content_snippet="Content of article one",
            url="https://example.com/1",
            content_length=1200,
        ),
        RelevanceCandidate(
            article_id="id-2",
            title="Article Two",
            content_snippet="Content of article two",
            published_date="2024-03-10",
            content_length=3500,
        ),
    ]
    save_relevance_candidates("test", candidates)
    loaded = load_relevance_candidates("test")
    assert len(loaded) == 2
    assert loaded[0].article_id == "id-1"
    assert loaded[0].content_length == 1200
    assert loaded[1].published_date == "2024-03-10"


def test_load_missing_file_returns_empty(tmp_path, monkeypatch):
    """Loading from a non-existent file should return empty list."""
    monkeypatch.setattr(
        "src.eval.relevance_candidates._eval_dir",
        lambda domain: tmp_path / "nonexistent",
    )
    assert load_relevance_candidates("test") == []


# ── RelevanceLabel persistence ────────────────────────────


def test_relevance_label_append_and_load(tmp_path, monkeypatch):
    """Appending labels should accumulate in the file."""
    monkeypatch.setattr(
        "src.eval.relevance_labels._eval_dir",
        lambda domain: tmp_path,
    )
    rl1 = make_relevance_label("article-1", "relevant")
    rl2 = make_relevance_label("article-2", "irrelevant", notes="off-topic")

    append_relevance_label("test", rl1)
    append_relevance_label("test", rl2)

    labels = load_relevance_labels("test")
    assert len(labels) == 2
    assert labels["article-1"].label == "relevant"
    assert labels["article-2"].label == "irrelevant"
    assert labels["article-2"].notes == "off-topic"


def test_relevance_label_last_write_wins(tmp_path, monkeypatch):
    """Re-labelling the same article should keep only the last label."""
    monkeypatch.setattr(
        "src.eval.relevance_labels._eval_dir",
        lambda domain: tmp_path,
    )
    rl1 = make_relevance_label("article-1", "relevant")
    rl2 = make_relevance_label("article-1", "noisy", notes="has sidebar junk")

    append_relevance_label("test", rl1)
    append_relevance_label("test", rl2)

    labels = load_relevance_labels("test")
    assert len(labels) == 1
    assert labels["article-1"].label == "noisy"
    assert labels["article-1"].notes == "has sidebar junk"


def test_relevance_label_load_missing_file(tmp_path, monkeypatch):
    """Loading from a non-existent gold file should return empty dict."""
    monkeypatch.setattr(
        "src.eval.relevance_labels._eval_dir",
        lambda domain: tmp_path / "nonexistent",
    )
    assert load_relevance_labels("test") == {}


def test_make_relevance_label_has_timestamp():
    """Factory function should include a UTC timestamp."""
    rl = make_relevance_label("article-1", "relevant")
    assert rl.timestamp  # non-empty
    assert "T" in rl.timestamp  # ISO format
