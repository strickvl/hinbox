"""Tests for per-task model selection (``HINBOX_<TASK>_MODEL``)."""

from unittest.mock import patch

import pytest

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.engine.article_processor import ArticleProcessor
from src.utils import task_models
from src.utils.task_models import check_task_models, task_model


@pytest.fixture(autouse=True)
def _no_overrides(monkeypatch):
    for task in task_models.TASKS:
        monkeypatch.delenv(f"HINBOX_{task.upper()}_MODEL", raising=False)


def test_defaults_follow_the_run_type():
    assert task_model("match", "gemini") == CLOUD_MODEL
    assert task_model("match", "ollama") == OLLAMA_MODEL


def test_override_applies_to_one_task_only(monkeypatch):
    monkeypatch.setenv("HINBOX_MATCH_MODEL", "ollama/qwen3.6:35b")
    assert task_model("match", "gemini") == "ollama/qwen3.6:35b"
    assert task_model("profile", "gemini") == CLOUD_MODEL


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        task_model("summarise", "gemini")


def test_local_run_refuses_cloud_overrides(monkeypatch):
    monkeypatch.setenv("HINBOX_PROFILE_MODEL", "openai/gpt-6-luna")
    with pytest.raises(ValueError, match="HINBOX_PROFILE_MODEL"):
        check_task_models("ollama")
    assert check_task_models("gemini")["profile"] == "openai/gpt-6-luna"


def test_processor_routes_relevance_by_the_task_model_prefix(monkeypatch):
    monkeypatch.setenv("HINBOX_RELEVANCE_MODEL", "ollama/qwen3.6:35b")
    processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
    assert processor.extraction_model == CLOUD_MODEL

    class Verdict:
        is_relevant = True
        reason = ""

    with patch(
        "src.engine.article_processor.ollama_check_relevance", return_value=Verdict()
    ) as local_check:
        outcome = processor.check_relevance("text", "art-1")
    assert outcome.value is True
    assert local_check.call_args.kwargs["model"] == "ollama/qwen3.6:35b"
