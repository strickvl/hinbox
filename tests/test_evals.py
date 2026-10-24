"""Tests for the evaluation scoreboard: gold loading, splits, metrics, run records."""

import json

from src.evals.datasets import (
    EvalItem,
    assign_split,
    filter_split,
    load_merge_items,
    load_relevance_items,
)
from src.evals.metrics import binary_metrics
from src.evals.runner import load_scoreboard, run_eval, save_run, summarize


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestMetrics:
    def test_always_majority_gets_half_balanced_accuracy(self):
        # 9 positives, 1 negative; a model that always says True
        pairs = [(True, True)] * 9 + [(False, True)]
        m = binary_metrics(pairs)
        assert m["accuracy"] == 0.9
        assert m["balanced_accuracy"] == 0.5
        assert m["majority_baseline"] == 0.9

    def test_errors_count_as_wrong(self):
        m = binary_metrics([(True, None), (False, None), (True, True), (False, False)])
        assert m["errors"] == 2
        assert m["accuracy"] == 0.5
        assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1)

    def test_perfect(self):
        m = binary_metrics([(True, True), (False, False)])
        assert m["balanced_accuracy"] == 1.0
        assert m["f1"] == 1.0


class TestSplits:
    def test_split_is_deterministic_and_roughly_60_20_20(self):
        ids = [f"item-{i}" for i in range(2000)]
        splits = [assign_split(i) for i in ids]
        assert splits == [assign_split(i) for i in ids]
        assert 0.55 < splits.count("train") / 2000 < 0.65
        assert 0.15 < splits.count("test") / 2000 < 0.25

    def test_filter_split_partitions_items(self):
        items = [EvalItem(item_id=f"x{i}", label=True) for i in range(100)]
        parts = [filter_split(items, s) for s in ("train", "dev", "test")]
        assert sum(len(p) for p in parts) == 100
        assert filter_split(items, "all") == items


class TestGoldLoading:
    def test_merge_uses_latest_label_drops_unsure_and_cleans_names(self, tmp_path):
        _write_jsonl(
            tmp_path / "merge_candidates.jsonl",
            [
                {
                    "entity_name": "A",
                    "candidate_name": "B",
                    "entity_type": "person",
                    "entity_context": "ctx A",
                    "candidate_context": "ctx B",
                },
                {
                    "entity_name": "US",
                    "candidate_name": {"name": "U.S. soil", "type": "country"},
                    "entity_type": "location",
                    "entity_context": "",
                    "candidate_context": "",
                },
            ],
        )
        _write_jsonl(
            tmp_path / "merge_gold.jsonl",
            [
                {
                    "entity_name": "A",
                    "candidate_name": "B",
                    "entity_type": "person",
                    "label": "no",
                    "timestamp": "2026-01-01T00:00:01",
                },
                {
                    "entity_name": "A",
                    "candidate_name": "B",
                    "entity_type": "person",
                    "label": "yes",
                    "timestamp": "2026-01-01T00:00:02",
                },
                {
                    "entity_name": "US",
                    "candidate_name": "{'name': 'U.S. soil', 'type': 'country'}",
                    "entity_type": "location",
                    "label": "yes",
                    "timestamp": "t",
                },
                {
                    "entity_name": "C",
                    "candidate_name": "D",
                    "entity_type": "person",
                    "label": "unsure",
                    "timestamp": "t",
                },
            ],
        )
        items = {i.item_id: i for i in load_merge_items(str(tmp_path), "v1")}
        assert set(items) == {"merge:person:A|B", "merge:location:US|U.S. soil"}
        ab = items["merge:person:A|B"]
        assert ab.label is True
        assert ab.inputs["entity_type"] == "people"
        assert ab.inputs["new_profile_text"] == "ctx A"

    def test_relevance_falls_back_to_snippet_without_articles(self, tmp_path):
        _write_jsonl(
            tmp_path / "relevance_candidates.jsonl",
            [{"article_id": "a1", "title": "T", "content_snippet": "snippet"}],
        )
        _write_jsonl(
            tmp_path / "relevance_gold.jsonl",
            [{"article_id": "a1", "label": "irrelevant", "timestamp": "t"}],
        )
        [item] = load_relevance_items(str(tmp_path), articles_path=None)
        assert item.label is False
        assert item.inputs["text"] == "snippet"

    def test_merge_v2_reads_suffixed_files(self, tmp_path):
        row = {
            "entity_name": "Navy",
            "candidate_name": "U.S. Navy",
            "entity_type": "organization",
        }
        _write_jsonl(
            tmp_path / "merge_candidates_v2.jsonl",
            [{**row, "entity_context": "a", "candidate_context": "b"}],
        )
        _write_jsonl(
            tmp_path / "merge_gold_v2.jsonl",
            [{**row, "label": "yes", "timestamp": "t"}],
        )
        [item] = load_merge_items(str(tmp_path), "v2")
        assert item.label is True and item.inputs["entity_type"] == "organizations"


class TestRunner:
    def test_run_records_errors_and_appends_scoreboard(self, tmp_path):
        items = [
            EvalItem(item_id="ok", label=True, inputs={"x": 1}),
            EvalItem(item_id="wrong", label=False, inputs={"x": 1}),
            EvalItem(item_id="boom", label=True, inputs={"x": 0}),
        ]

        def predictor(inputs):
            if inputs["x"] == 0:
                raise RuntimeError("model down")
            return True

        records = run_eval(items, predictor, workers=2)
        assert [r["item_id"] for r in records] == ["ok", "wrong", "boom"]
        assert records[2]["prediction"] is None
        assert "model down" in records[2]["error"]

        summary = summarize(
            records, task="merge", model="ollama/m:1", split="all", approach="pipeline"
        )
        assert summary["errors"] == 1
        assert summary["accuracy"] == round(1 / 3, 4)

        save_run(str(tmp_path), summary, records)
        save_run(str(tmp_path), summary, records)
        board = load_scoreboard(str(tmp_path))
        assert len(board) == 2
        assert (tmp_path / "runs" / board[0]["predictions_file"]).exists()
