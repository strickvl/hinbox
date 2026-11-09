"""The placement eval (src/evals/placement.py) and its classify prompt
(src/engine/section_place.py), with canned model replies."""

import json
from typing import Any, Dict, List

from src.engine.section_place import Placement, Placements, messages, read_reply
from src.engine.section_split import SectionSplit, SubSection
from src.evals.placement import (
    baseline,
    batches,
    load_labels,
    score_batches,
    summarize,
)

CONFIG = {
    "profiles": {
        "starter_headings": {
            "people": {
                "default": [
                    {"Background": "who they are and their earlier career"},
                    {"Role": "the posts they held, not what they did in them"},
                    {"Cases and decisions": "rulings, orders and actions"},
                ]
            }
        }
    }
}
STARTER = ["Background", "Role", "Cases and decisions"]


def _row(sentence: str, gold: str, **extra: Any) -> Dict[str, Any]:
    return {
        "entity": "Vance Spath", "entity_type": "people", "kind": "default",
        "sentence": sentence, "gold": gold, "confidence": "sure", **extra,
    }  # fmt: skip


ROWS = [
    _row("Spath served as an Air Force prosecutor.^[a1]", "Background",
         source="r22-luna", placed="Background"),
    _row("Spath suspended the USS Cole trial.^[a2]", "Cases and decisions",
         source="r22-qwen", placed="Background > Post-military career"),
    _row("He was chief judge of the trial judiciary.^[a3]", "Role",
         source="r22-qwen", placed="Role"),
]  # fmt: skip


def _write(tmp_path, rows: List[Dict[str, Any]]) -> str:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_only_sure_rows_with_a_starter_gold_are_scored(tmp_path):
    extra = [
        _row("Spath retired.^[a4]", "Role", confidence="unsure"),
        _row("Spath ruled.^[a5]", "Legal proceedings"),
        _row("Spath ruled again.^[a6]", "cases and decisions: rulings"),
    ]
    rows, skipped = load_labels(_write(tmp_path, ROWS + extra), CONFIG)
    assert len(rows) == 4 and skipped == {"unsure": 1, "bad_gold": 1}
    assert rows[-1]["gold"] == "Cases and decisions"
    assert len({r["id"] for r in rows}) == 4


def test_batches_are_the_same_for_every_approach():
    rows, _ = load_labels_from(ROWS)
    one, two = batches(rows, 2, seed=1), batches(rows, 2, seed=1)
    assert [[r["id"] for r in b] for b in one] == [[r["id"] for r in b] for b in two]
    assert sorted(len(b) for b in one) == [1, 2]


def load_labels_from(rows):
    return [{**r, "id": str(i)} for i, r in enumerate(rows)], {}


def test_classify_reads_headings_back_with_their_notes():
    msgs = messages(["A.", "B."], STARTER, {"Role": "posts"}, "person", "Spath")
    assert "- Role: posts" in msgs[0]["content"]
    assert "[2] B." in msgs[1]["content"]
    reply = Placements(
        placements=[
            Placement(sentence=2, about="a ruling", heading="Cases and decisions"),
            Placement(sentence=1, about="career", heading="Background: who they are"),
            Placement(sentence=1, about="again", heading="Role"),  # repeat ignored
            Placement(sentence=3, about="out of range", heading="Role"),
        ]
    )
    assert read_reply(reply, 2, STARTER) == ["Background", "Cases and decisions"]


def test_both_approaches_score_against_gold():
    rows, _ = load_labels_from(ROWS)
    groups = [rows]
    classify = lambda msgs: Placements(  # noqa: E731
        placements=[
            Placement(sentence=1, about="", heading="Background"),
            Placement(sentence=2, about="", heading="Cases and decisions"),
            Placement(sentence=3, about="", heading="Cases and decisions"),
        ]
    )
    records = score_batches("classify", groups, CONFIG, classify, workers=1)
    summary = summarize(records)
    assert summary["right"] == 2 and summary["per_heading"]["Role"] == "0/1"
    assert summary["balanced_accuracy"] == round(2 / 3, 4)
    assert summary["top_mistakes"] == {"Role -> Cases and decisions": 1}

    reorganise = lambda msgs: SectionSplit(  # noqa: E731
        subsections=[
            SubSection(heading="Background", paragraphs=[1, 2]),
            SubSection(heading="Trial matters", paragraphs=[3]),
        ]
    )
    records = score_batches("reorganise", groups, CONFIG, reorganise, workers=1)
    assert [r["answer"] for r in records] == [
        "Background", "Background", "(new) Trial matters",
    ]  # fmt: skip
    assert summarize(records)["right"] == 1


def test_a_failed_call_is_an_error_not_a_wrong_answer():
    def down(msgs):
        raise RuntimeError("down")

    rows, _ = load_labels_from(ROWS)
    summary = summarize(score_batches("classify", [rows], CONFIG, down, workers=1))
    assert summary["errors"] == 3 and summary["accuracy"] == 0.0


def test_baseline_counts_where_each_model_had_put_rows():
    assert baseline(ROWS, CONFIG) == {"luna": "1/1", "qwen": "1/2"}
