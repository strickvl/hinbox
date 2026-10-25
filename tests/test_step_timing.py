"""Tests for per-step timings (src/utils/step_timing.py)."""

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor

from src.utils import step_timing
from src.utils.llm import cloud_llm_slot, configure_llm_concurrency
from src.utils.step_timing import timed, timed_step


def setup_function():
    step_timing.reset_timings()


def test_steps_nest_and_llm_calls_land_on_innermost_step():
    configure_llm_concurrency(cloud_in_flight=2)
    with timed_step("merge.people"):
        with timed_step("profile.update"):
            with cloud_llm_slot():
                pass
            with cloud_llm_slot():
                pass
    report = step_timing.timing_report(articles=2, wall_seconds=10.0)
    assert set(report["steps"]) == {"merge.people", "merge.people/profile.update"}
    assert report["llm_calls"]["merge.people/profile.update"]["calls"] == 2
    assert report["wall_seconds_per_article"] == 5.0


def test_failed_llm_call_is_counted():
    with timed_step("relevance"):
        try:
            with cloud_llm_slot():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert step_timing.timing_report(1, 1.0)["llm_calls"]["relevance"]["failed"] == 1


def test_decorator_and_thread_pool_with_copied_context():
    @timed("extract.people")
    def extract():
        with cloud_llm_slot():
            return 1

    with timed_step("extraction"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(contextvars.copy_context().run, extract).result()
            pool.submit(extract).result()  # no copied context: top-level step
    steps = step_timing.timing_report(1, 1.0)["steps"]
    assert "extraction/extract.people" in steps
    assert "extract.people" in steps


def test_save_timing_report_writes_json(tmp_path):
    with timed_step("write"):
        pass
    path = step_timing.save_timing_report(str(tmp_path), articles=1, wall_seconds=2.0)
    data = json.loads(open(path).read())
    assert data["steps"]["write"]["count"] == 1
