"""Offline tests for the speed baseline script (no model, no pipeline run)."""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import speed_baseline as sb


def test_settings_env_keeps_settings_and_drops_secrets(monkeypatch):
    monkeypatch.setenv("HINBOX_OLLAMA_NUM_CTX", "32768")
    monkeypatch.setenv("OLLAMA_API_URL", "http://beast:8080/v1")
    monkeypatch.setenv("HINBOX_SOME_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    env = sb.settings_env()
    assert env["HINBOX_OLLAMA_NUM_CTX"] == "32768"
    assert env["OLLAMA_API_URL"] == "http://beast:8080/v1"
    assert "sk-secret" not in json.dumps(env)


def test_cached_pass_gets_extraction_cache_and_only_not_relevant_verdicts(tmp_path):
    fresh, cached = tmp_path / "fresh", tmp_path / "cached"
    (fresh / "cache" / "extractions").mkdir(parents=True)
    (fresh / "cache" / "extractions" / "x.json").write_text("{}")
    status = {
        "a1": {"processed": True, "content_hash": "h1"},
        "a2": {"processed": False, "reason": "not_relevant", "content_hash": "h2"},
        "a3": {"processed": False, "reason": "incomplete"},
    }
    (fresh / "processing_status.json").write_text(json.dumps(status))
    cached.mkdir()
    sb.seed_cached_pass(str(fresh), str(cached))
    assert (cached / "cache" / "extractions" / "x.json").exists()
    kept = json.loads((cached / "processing_status.json").read_text())
    # Processed articles must be merged again, so they are not carried over
    assert list(kept) == ["a2"]


def test_pass_quality_counts_outcomes_entities_and_grounding(tmp_path):
    rows = [
        {
            "name": "Camp Delta",
            "articles": [{"article_id": "a1"}],
            "profile": {"text": "Camp Delta is a camp.^[a1]", "qc": {"passed": True}},
            "profile_grounding": {
                "grounding_score": 0.5,
                "verified": 1,
                "total_citations": 2,
            },
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "locations.parquet")
    status = {
        "a1": {"processed": True},
        "a2": {"processed": False, "reason": "not_relevant"},
        "a3": {"processed": False, "reason": "incomplete"},
    }
    (tmp_path / "processing_status.json").write_text(json.dumps(status))
    q = sb.pass_quality(str(tmp_path), ["a1", "a2", "a3"])
    assert q["articles"] == {"processed": 1, "not_relevant": 1, "incomplete": 1}
    assert q["entities"]["locations"] == 1 and q["entities"]["people"] == 0
    assert q["grounding"] == {
        "profiles_scored": 1,
        "mean_score": 0.5,
        "verified": 1,
        "citations": 2,
    }


def test_render_one_row_per_pass():
    p = {
        "pass": "fresh",
        "wall_seconds": 100.0,
        "seconds_per_merged_article": 10.0,
        "articles_merged": 10,
        "llm": {"calls": 50, "failed": 1, "wait_seconds": 3.0},
        "quality": {
            "entities": {"people": 3, "locations": 2},
            "articles": {"incomplete": 1},
            "audit_totals": {"repeated_sentences": 4},
            "grounding": {"mean_score": 0.9},
        },
    }
    s = {
        "label": "w8",
        "git": {"commit": "abcdef123"},
        "overrides": {"max_buffered_articles": "8"},
        "passes": [p, {**p, "pass": "cached"}],
    }
    table = sb.render([s])
    assert table.count("| w8 |") == 2
    assert "max_buffered_articles=8" in table and "abcdef1" in table


def test_latest_timing_ignores_partial(tmp_path):
    tdir = tmp_path / "timings"
    tdir.mkdir()
    (tdir / "partial.json").write_text('{"x": 1}')
    (tdir / "run-20261105-100000.json").write_text('{"wall_seconds": 1}')
    (tdir / "run-20261105-110000.json").write_text('{"wall_seconds": 2}')
    assert sb.latest_timing(str(tmp_path)) == {"wall_seconds": 2}
    assert os.path.exists(tdir / "partial.json")
