"""Prompt files chosen by model: family, then local/cloud, then the default."""

import os

import pytest

from src.config_loader import DomainConfig
from src.evals.__main__ import _print_grid, _prompt_file


@pytest.fixture
def config(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (tmp_path / "config.yaml").write_text(
        "prompt_families:\n  qwen: [qwen]\n  bolt: [bolt, nemotron]\n"
    )
    for name in ("relevance", "relevance.local", "relevance.qwen", "people"):
        (prompts / f"{name}.md").write_text(name)
    (prompts / "people.cloud.md").write_text("people.cloud")
    monkeypatch.setattr(DomainConfig, "config_dir", str(tmp_path))
    monkeypatch.delenv("HINBOX_PROMPT_VARIANT", raising=False)
    return DomainConfig("test")


def test_family_beats_local_beats_default(config):
    assert config.load_prompt("relevance", "ollama/local-qwen36-16k") == (
        "relevance.qwen"
    )
    assert config.load_prompt("relevance", "ollama/local-bolt-16k") == (
        "relevance.local"
    )
    assert config.load_prompt("relevance", "openai/gpt-6-luna") == "relevance"
    assert config.load_prompt("relevance") == "relevance"


def test_cloud_variant_and_missing_variants_fall_back(config):
    assert config.load_prompt("people", "openai/gpt-6-luna") == "people.cloud"
    # A family with no file of its own falls through to the default
    assert config.load_prompt("people", "ollama/qwen3.6") == "people"


def test_variant_can_be_forced_for_the_eval(config, monkeypatch):
    monkeypatch.setenv("HINBOX_PROMPT_VARIANT", "qwen")
    assert config.load_prompt("relevance", "openai/gpt-6-luna") == "relevance.qwen"
    monkeypatch.setenv("HINBOX_PROMPT_VARIANT", "default")
    assert config.load_prompt("relevance", "ollama/qwen3.6") == "relevance"


def test_missing_prompt_raises(config):
    with pytest.raises(ValueError, match="Prompt file not found"):
        config.load_prompt("events", "ollama/qwen3.6")


def test_scoreboard_records_the_prompt_file(config):
    assert _prompt_file(config, "relevance", "ollama/qwen3.6") == "relevance.qwen.md"
    assert _prompt_file(config, "merge", "ollama/qwen3.6") == "built-in"


def test_real_domain_gives_local_models_the_tuned_edit_prompt(monkeypatch):
    monkeypatch.delenv("HINBOX_PROMPT_VARIANT", raising=False)
    path = DomainConfig("guantanamo").prompt_path("profile_edit", "ollama/x")
    assert os.path.basename(path) == "profile_edit.local.md"
    path = DomainConfig("guantanamo").prompt_path("profile_edit", "openai/x")
    assert os.path.basename(path) == "profile_edit.cloud.md"


def test_grid_keeps_the_latest_run_per_model_and_prompt(capsys):
    rows = [
        {"dataset": "relevance", "approach": "pipeline", "split": "test",
         "model": "m", "prompt": "relevance.md", "balanced_accuracy": 0.5, "n": 9},
        {"dataset": "relevance", "approach": "pipeline", "split": "test",
         "model": "m", "prompt": "relevance.md", "balanced_accuracy": 0.9, "n": 9},
        {"dataset": "relevance", "approach": "pipeline", "split": "test",
         "model": "m", "balanced_accuracy": 0.7, "n": 9},
    ]  # fmt: skip
    _print_grid(rows)
    out = capsys.readouterr().out
    assert "0.9 (n=9)" in out and "0.5" not in out and "0.7 (n=9)" in out
