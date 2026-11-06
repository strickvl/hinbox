"""Tests for the bounded submission window and the concurrency env overrides."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pytest

from src.config_loader import DomainConfig
from src.utils.article_window import ordered_window


def test_results_come_back_in_item_order_even_when_later_items_finish_first():
    def work(i: int) -> int:
        time.sleep(0.05 if i == 0 else 0)  # item 0 finishes last
        return i

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = [
            f.result()
            for f in ordered_window(range(6), lambda i: pool.submit(work, i), 3)
        ]
    assert got == list(range(6))


def test_never_more_than_window_items_submitted_but_not_consumed():
    submitted: List[int] = []
    consumed: List[int] = []
    max_ahead = 0

    def submit(i: int) -> int:
        submitted.append(i)
        return i

    for i in ordered_window(range(10), submit, 3):
        # Everything submitted so far, minus what's been merged already
        max_ahead = max(max_ahead, len(submitted) - len(consumed))
        assert i == len(consumed)
        consumed.append(i)
    assert consumed == list(range(10))
    assert max_ahead == 3


def test_next_item_is_submitted_only_when_caller_moves_on():
    submitted: List[int] = []
    gen = ordered_window(range(5), lambda i: submitted.append(i) or i, 2)
    assert submitted == []  # a generator does nothing until asked
    assert next(gen) == 0
    assert submitted == [0, 1]
    assert next(gen) == 1  # caller is done with 0, so 2 goes in
    assert submitted == [0, 1, 2]


def test_window_larger_than_items_and_empty_input():
    assert list(ordered_window(range(3), lambda i: i * 10, 50)) == [0, 10, 20]
    assert list(ordered_window([], lambda i: i, 4)) == []
    with pytest.raises(ValueError):
        list(ordered_window(range(3), lambda i: i, 0))


def test_extraction_stays_bounded_while_merging_is_slow():
    """With a slow consumer, workers can't race ahead of the window."""
    lock = threading.Lock()
    started: List[int] = []
    merged: List[int] = []
    peak: Dict[str, int] = {"ahead": 0}

    def extract(i: int) -> int:
        with lock:
            started.append(i)
            peak["ahead"] = max(peak["ahead"], len(started) - len(merged))
        return i

    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in ordered_window(range(20), lambda i: pool.submit(extract, i), 4):
            value = future.result()
            time.sleep(0.005)  # a slow merge
            with lock:
                merged.append(value)
    assert merged == list(range(20))
    assert peak["ahead"] <= 4


def test_concurrency_env_overrides_beat_config(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "performance:\n  concurrency:\n    extract_workers: 8\n"
        "  queue:\n    max_buffered_articles: 32\n"
    )
    monkeypatch.setattr(DomainConfig, "config_dir", str(tmp_path))
    cfg = DomainConfig("x")
    assert cfg.get_concurrency_config()["max_buffered_articles"] == 32
    monkeypatch.setenv("HINBOX_MAX_BUFFERED_ARTICLES", "6")
    monkeypatch.setenv("HINBOX_EXTRACT_WORKERS", "2")
    cc = cfg.get_concurrency_config()
    assert (cc["max_buffered_articles"], cc["extract_workers"]) == (6, 2)


def test_output_dir_env_override(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text("output:\n  directory: data/x\n")
    monkeypatch.setattr(DomainConfig, "config_dir", str(tmp_path))
    assert DomainConfig("x").get_output_dir() == "data/x"
    monkeypatch.setenv("HINBOX_OUTPUT_DIR", "/tmp/bench/fresh")
    assert DomainConfig("x").get_output_dir() == "/tmp/bench/fresh"
