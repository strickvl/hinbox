"""Offline tests for the DSPy merge experiment (run with ``just dspy-test``)."""

import importlib.util

import dspy

from experiments.dspy_opt import lm as lm_module
from experiments.dspy_opt.merge import (
    correct,
    gepa_metric,
    make_predictor,
    split_for_optimizer,
    to_example,
)
from src.evals.datasets import EvalItem


def _item(label: bool, n: int = 0, entity_type: str = "locations") -> EvalItem:
    return EvalItem(
        item_id=f"i{n}",
        label=label,
        inputs={
            "new_name": f"A{n}",
            "existing_name": f"B{n}",
            "new_profile_text": "new",
            "existing_profile_text": "old",
            "entity_type": entity_type,
        },
    )


def test_litellm_is_not_installed():
    assert importlib.util.find_spec("litellm") is None


def test_to_example_adds_type_rules_and_label():
    example = to_example(_item(True))
    assert "Location-Specific Rules" in example.rules
    assert example.is_match is True
    assert set(example.inputs().keys()) == {
        "entity_type",
        "rules",
        "new_name",
        "new_profile_text",
        "existing_name",
        "existing_profile_text",
    }


def test_split_for_optimizer_puts_half_the_positives_in_validation():
    pool = [to_example(_item(True, n)) for n in range(10)]
    pool += [to_example(_item(False, n)) for n in range(10, 100)]
    train, val = split_for_optimizer(pool, seed=1)
    assert len(train) + len(val) == 100
    assert sum(e.is_match for e in val) == 5
    assert len(val) == 20
    assert not {e.new_name for e in train} & {e.new_name for e in val}


def test_metrics():
    gold = to_example(_item(False))
    assert correct(gold, dspy.Prediction(is_match=False))
    wrong = gepa_metric(gold, dspy.Prediction(is_match=True))
    assert wrong.score == 0.0 and "different entities" in wrong.feedback


def test_predictor_passes_rules_to_the_program():
    seen = {}

    def program(**inputs):
        seen.update(inputs)
        return dspy.Prediction(is_match=True)

    assert make_predictor(program)(_item(False).inputs) is True
    assert "Location-Specific Rules" in seen["rules"]


def test_make_lm_points_ollama_at_the_configured_server(monkeypatch):
    monkeypatch.setattr(lm_module, "OLLAMA_API_URL", "http://box:11434/v1")
    monkeypatch.setattr(lm_module, "OLLAMA_REASONING_EFFORT", "none")
    lm = lm_module.make_lm("ollama/qwen3.6:35b")
    assert lm.kwargs["api_base"] == "http://box:11434/v1"
    assert lm.kwargs["reasoning_effort"] == "none"
    assert "api_base" not in lm_module.make_lm("openai/gpt-6-luna").kwargs
