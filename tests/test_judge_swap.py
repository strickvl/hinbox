"""Swap commands around the citation check (`src/engine/judge_swap.py`)."""

from typing import List

import pytest

from src.engine import judge_swap
from src.engine.judge_swap import (
    JudgeSwapFailed,
    judge_server,
    swap_back,
    swap_to_judge,
    wait_for_server,
)
from src.utils import server_health
from src.utils.server_health import ModelServerUnreachable


@pytest.fixture
def no_commands(monkeypatch):
    for name in (judge_swap.BEFORE_ENV, judge_swap.AFTER_ENV, judge_swap.WAIT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HINBOX_GROUNDING_MODEL", raising=False)


def _touch(path) -> str:
    return f"echo ran >> {path}"


def test_nothing_runs_and_nothing_is_saved_without_a_command(no_commands, monkeypatch):
    saved: List[str] = []
    monkeypatch.setattr(
        judge_swap, "server_answers", lambda: pytest.fail("no wait expected")
    )
    with judge_server("ollama", lambda: saved.append("saved")):
        pass
    assert saved == []


def test_saves_then_swaps_then_swaps_back(no_commands, monkeypatch, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    monkeypatch.setenv(judge_swap.BEFORE_ENV, _touch(before))
    monkeypatch.setenv(judge_swap.AFTER_ENV, _touch(after))
    monkeypatch.setattr(judge_swap, "server_answers", lambda: True)
    order: List[str] = []
    with judge_server("ollama", lambda: order.append(f"saved:{before.exists()}")):
        order.append(f"check:{before.exists()}:{after.exists()}")
    # Saved before the command ran; the check ran between the two commands
    assert order == ["saved:False", "check:True:False"]
    assert after.read_text().strip() == "ran"


def test_swap_back_runs_when_the_check_raises(no_commands, monkeypatch, tmp_path):
    after = tmp_path / "after"
    monkeypatch.setenv(judge_swap.AFTER_ENV, _touch(after))
    with pytest.raises(KeyboardInterrupt):
        with judge_server("ollama", lambda: None):
            raise KeyboardInterrupt
    assert after.exists()


def test_failed_swap_command_stops_like_an_unreachable_server(
    no_commands, monkeypatch, tmp_path
):
    after = tmp_path / "after"
    monkeypatch.setenv(judge_swap.BEFORE_ENV, "exit 3")
    monkeypatch.setenv(judge_swap.AFTER_ENV, _touch(after))
    saved: List[str] = []
    with pytest.raises(ModelServerUnreachable, match="exited with 3"):
        with judge_server("ollama", lambda: saved.append("saved")):
            pytest.fail("the check must not run after a failed swap")
    # Merged work was saved first, and the pipeline's server is asked back
    assert saved == ["saved"] and after.exists()


def test_waits_for_the_server_then_gives_up():
    answers = iter([False, False, True])
    slept: List[float] = []
    wait_for_server(lambda: next(answers), limit=60, pause=2, sleep=slept.append)
    assert slept == [2, 2]

    now = iter(range(0, 1000, 10))
    with pytest.raises(JudgeSwapFailed, match="did not answer within 30 s"):
        wait_for_server(
            lambda: False,
            limit=30,
            pause=1,
            sleep=lambda _: None,
            clock=lambda: next(now),
        )


def test_cloud_judge_is_not_waited_for(no_commands, monkeypatch):
    monkeypatch.setenv(judge_swap.BEFORE_ENV, "true")
    monkeypatch.setattr(
        judge_swap, "server_answers", lambda: pytest.fail("no wait expected")
    )
    assert swap_to_judge(local_judge=False) is True


def test_swap_resets_the_connection_failure_count(no_commands, monkeypatch):
    monkeypatch.setenv(judge_swap.BEFORE_ENV, "true")
    monkeypatch.setattr(judge_swap, "server_answers", lambda: True)
    monkeypatch.setattr(server_health, "_in_a_row", 50)
    with judge_server("ollama", lambda: None):
        assert server_health.unreachable() is None


def test_failed_swap_back_is_logged_not_raised(no_commands, monkeypatch):
    monkeypatch.setenv(judge_swap.AFTER_ENV, "exit 1")
    assert swap_back() is True
