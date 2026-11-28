"""Tests for several answers per claim and paired comparisons
(``src/citecheck/votes.py``). Every id and reply here is made up."""

import json

from src.citecheck import votes as v
from src.citecheck.__main__ import main
from src.citecheck.export import reply_json
from src.citecheck.items import write_jsonl

ITEMS = [{"item_id": f"i{n}", "stratum": "random"} for n in range(4)]
LABELS = {
    "i0": {"label": "no"},
    "i1": {"label": "no"},
    "i2": {"label": "yes"},
    "i3": {"label": "unsure"},
}


def _reply(levels, sample=0, ids=("i0", "i1", "i2", "i3")):
    return {
        "item_ids": list(ids),
        "sample": sample,
        "reply": reply_json([(lv, "") for lv in levels]),
    }


def test_votes_majority_and_dial():
    rows = [
        _reply(["supported", "partial", "supported", "supported"], 0),
        _reply(["supported", "partial", "partial", "supported"], 1),
        _reply(["partial", "supported", "supported", "supported"], 2),
    ]
    samples = v.by_sample(rows)
    counts = v.vote_counts(samples)
    assert counts["i0"] == (2, 3) and counts["i1"] == (1, 3)
    assert v.majority(counts)["i0"] == "supported"
    assert v.majority(counts)["i1"] == "partial"
    assert v.at_least(counts, 3)["i0"] == "partial"
    report = v.vote_report(ITEMS, LABELS, samples)
    assert report["alone"][0] == {"wrong_passed": 1, "good_doubted": 0, "errors": 0}
    assert report["spread"]["wrong_passed"]["min"] == 1
    assert report["majority"]["wrong_passed"]["count"] == 1
    assert report["dial"][1]["wrong_passed"] == 2  # any yes passes
    assert report["dial"][3] == {"wrong_passed": 0, "good_doubted": 1, "errors": 0}
    assert "k=3" in v.describe_votes(report, "x")


def test_unreadable_answers_are_not_counted_and_tie_is_doubt():
    rows = [
        _reply(["supported"], 0, ids=("i0",)),
        _reply(["partial"], 1, ids=("i0",)),
        {"item_ids": ["i0"], "sample": 2, "reply": "no verdict here"},
    ]
    counts = v.vote_counts(v.by_sample(rows))
    assert counts["i0"] == (1, 2)
    assert v.majority(counts)["i0"] == "partial"
    assert v.majority({"x": (0, 0)})["x"] is None


def test_sign_test_and_paired():
    assert v.sign_test(0, 0) == 1.0
    assert abs(v.sign_test(5, 3) - 0.7266) < 1e-3
    assert v.sign_test(30, 10) < 0.01
    base = {"i0": "supported", "i1": "supported", "i2": "supported"}
    other = {"i0": "partial", "i1": "supported", "i2": "partial"}
    pr = v.paired(ITEMS, LABELS, base, other)
    assert pr["wrong"]["other_better"] == 1 and pr["wrong"]["other_worse"] == 0
    assert pr["good"]["other_better"] == 0 and pr["good"]["other_worse"] == 1


def test_core_and_audit():
    a = {"i0": "supported", "i1": "partial", "i2": "partial"}
    b = {"i0": "supported", "i1": "supported", "i2": "partial"}
    assert v.wrong_passed_by_all(ITEMS, LABELS, [a, b]) == ["i0"]
    # i0 (no) passed by both, i2 (yes) doubted by both; i3 unsure is skipped
    assert v.all_disagree(ITEMS, LABELS, [a, b]) == ["i0", "i2"]
    assert v.all_disagree(ITEMS, LABELS, [a, {"i0": None}]) == []


def test_votes_and_paired_commands(tmp_path, capsys):
    items, labels = tmp_path / "items.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(str(items), ITEMS)
    write_jsonl(str(labels), [{"item_id": k, **lab} for k, lab in LABELS.items()])
    votes = tmp_path / "votes.jsonl"
    write_jsonl(
        str(votes),
        [_reply(["supported", "partial", "supported", "partial"], n) for n in range(3)],
    )
    single = tmp_path / "single.jsonl"
    write_jsonl(
        str(single), [_reply(["supported", "supported", "supported", "partial"])]
    )
    saved = tmp_path / "saved.json"
    saved.write_text(
        json.dumps({"predicted": {"i0": "partial", "i1": "partial", "i2": "partial"}})
    )
    common = ["--items", str(items), "--labels", str(labels)]
    main(["votes", *common, "--replies", str(votes)])
    assert "3 answers per claim" in capsys.readouterr().out
    audit = tmp_path / "audit.txt"
    main(
        [
            "paired",
            *common,
            "--sources",
            f"one={single}",
            f"vote={votes}",
            f"judge={saved}",
            "--core-of",
            "one,vote",
            "--audit-of",
            "one,judge",
            "--audit-out",
            str(audit),
        ]
    )
    out = capsys.readouterr().out
    assert "majority of 3" in out and "| judge | saved |" in out
    assert audit.read_text() == ""  # i0: one passes (wrong), judge doubts (right)
