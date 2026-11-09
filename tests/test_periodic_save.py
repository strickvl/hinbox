"""Saving the entity tables during a run (``HINBOX_SAVE_EVERY``).

Scenario: a 1,620-article local run takes about 30 hours. It used to save
only at its end, on Ctrl+C or when the model server was gone, so a crash,
`kill` or shutdown at hour 25 lost every merge and profile call made so far.
"""

import os
import signal
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from src import process_and_extract as pae


def _run(tmp_path, monkeypatch, count: int, skipped=()):
    saves = []
    config = MagicMock()
    config.get_concurrency_config.return_value = {
        "extract_workers": 1, "extract_per_article": 1, "max_buffered_articles": 1,
    }  # fmt: skip
    config.get_cache_config.return_value = {}
    config.get_output_dir.return_value = str(tmp_path)
    monkeypatch.setattr(
        pae, "extract_single_article_only",
        lambda row, *a, **k: MagicMock(row=row, was_extracted=row["id"] not in skipped),
    )  # fmt: skip
    monkeypatch.setattr(pae, "merge_and_finalize", lambda result, **k: None)
    monkeypatch.setattr(pae, "article_scope", lambda span, result: nullcontext())
    monkeypatch.setattr(pae, "write_partial_report", lambda *a: None)
    monkeypatch.setattr(
        pae, "save_entities_then_status", lambda e, d, s: saves.append(d)
    )
    rows = [{"id": f"a{i}"} for i in range(count)]
    pae.process_articles_batch(
        rows, MagicMock(), {}, MagicMock(limit=count), domain_config=config
    )
    return saves


def test_the_tables_are_saved_every_few_merged_articles(tmp_path, monkeypatch):
    monkeypatch.setenv("HINBOX_SAVE_EVERY", "2")
    assert _run(tmp_path, monkeypatch, 5) == [str(tmp_path)] * 2  # after 2 and 4
    monkeypatch.setenv("HINBOX_SAVE_EVERY", "0")
    assert _run(tmp_path, monkeypatch, 5) == []


def test_skipped_articles_do_not_count_toward_a_save(tmp_path, monkeypatch):
    # A resumed run passes over its already-processed articles without merging
    monkeypatch.setenv("HINBOX_SAVE_EVERY", "2")
    assert _run(tmp_path, monkeypatch, 6, skipped={"a0", "a1", "a2", "a4"}) == [
        str(tmp_path)
    ]  # a3, a5
    assert len(_run(tmp_path, monkeypatch, 6, skipped={"a0", "a2"})) == 2
    every = {f"a{i}" for i in range(6)}
    assert _run(tmp_path, monkeypatch, 6, skipped=every) == []


def test_the_setting_defaults_to_25(monkeypatch):
    monkeypatch.delenv("HINBOX_SAVE_EVERY", raising=False)
    assert pae.save_every_articles() == 25
    monkeypatch.setenv("HINBOX_SAVE_EVERY", "x")
    assert pae.save_every_articles() == 25
    monkeypatch.setenv("HINBOX_SAVE_EVERY", "-3")
    assert pae.save_every_articles() == 0


def test_sigterm_stops_the_run_like_ctrl_c():
    old = signal.signal(signal.SIGTERM, pae._interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, old)
