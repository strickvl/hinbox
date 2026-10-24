"""Offline tests for the DSPy profile-update experiment (run with ``just dspy-test``)."""

import dspy

from experiments.dspy_opt.profile_update import (
    ProfileUpdater,
    base_instructions,
    make_updater,
    metric,
    score_and_feedback,
    to_example,
)
from src.engine.profile_edits import ProfileEdit

OLD = (
    "## Background\n"
    "Nashiri was captured in Dubai in 2002 by local security forces.^[a1] "
    "He was transferred to Guantanamo Bay detention camp in 2006 with others.^[a2]"
)


def _case(kind="new_fact", article_text="He was arraigned in 2011."):
    return {
        "case_id": f"{kind}:x",
        "kind": kind,
        "split": "train",
        "entity_name": "Nashiri",
        "entity_type": "person",
        "profile": {"text": OLD, "sources": ["a1", "a2"]},
        "article_id": "a9",
        "article_text": article_text,
        "expected": {"must_cite": "a9"} if kind == "new_fact" else {"keep_all": True},
    }


def test_base_instructions_have_no_placeholders_left():
    text = base_instructions("guantanamo")
    assert "{" not in text and "^[<article_id>]" in text and "the entity" in text


def test_passing_update_scores_one():
    new = OLD + " He was arraigned in 2011.^[a9]"
    result = score_and_feedback(_case(), new)
    assert result["score"] == 1.0 and result["feedback"].startswith("Passed")


def test_missed_new_fact_and_invented_values_explain_themselves():
    result = score_and_feedback(_case(), OLD)
    assert result["score"] < 1 and "^[a9]" in result["feedback"]
    made_up = OLD + " A later report gives the year as 2007 instead.^[a9]"
    result = score_and_feedback(_case(kind="irrelevant"), made_up)
    assert "numbers or years" in result["feedback"] and result["score"] == 0.75


def test_metric_and_updater_apply_the_predicted_edits():
    edit = ProfileEdit(op="add", anchor="END", text="He was arraigned in 2011.^[a9]")

    class Fake(dspy.Module):
        def forward(self, **inputs):
            assert inputs["profile_text"] == OLD and inputs["article_id"] == "a9"
            return dspy.Prediction(disagreements=[], new_facts=["x"], edits=[edit])

    case = _case()
    assert make_updater(Fake())(case).rstrip().endswith("2011.^[a9]")
    pred = Fake()(**to_example(case).inputs())
    assert metric(to_example(case), pred).score == 1.0
    assert metric(to_example(case), dspy.Prediction(edits=["junk"])).score == 0.0


def test_program_uses_the_given_instructions():
    program = ProfileUpdater("Be careful.")
    assert program.update.signature.instructions == "Be careful."
