"""Thresholds and counts of the small-checker benchmark (no models loaded)."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "small_checkers",
    Path(__file__).parent.parent / "scripts" / "citecheck_small_checkers.py",
)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def test_equal_warnings_threshold_and_tally():
    scored = [(0.1, "no"), (0.2, "yes"), (0.6, "no"), (0.7, "yes"), (0.9, "yes")]
    cut = sc.threshold_for(scored, 2)
    assert cut == 0.6
    assert sc.tally(scored, cut) == {
        "wrong_passed": 1,
        "wrong": 2,
        "good_doubted": 1,
        "good": 3,
        "warnings": 2,
    }
    assert sc.tally(scored, sc.threshold_for(scored, 0))["warnings"] == 0
    assert sc.tally(scored, sc.threshold_for(scored, 9))["warnings"] == 5


def test_chunks_overlap():
    words = " ".join(str(i) for i in range(500))
    parts = sc.chunks(words, 350)
    assert parts[0].split()[0] == "0" and parts[-1].split()[-1] == "499"
    assert sc.chunks("a b c", 350) == ["a b c"]


def test_run_resumes(tmp_path):
    out = tmp_path / "x.jsonl"
    out.write_text(json.dumps({"item_id": "a", "score": 0.4}) + "\n")
    groups = {"art": [{"item_id": "a", "claim": "A"}, {"item_id": "b", "claim": "B"}]}
    seen = []

    def scorer(doc, claims):
        seen.extend(claims)
        return [0.9] * len(claims)

    sc.run("x", scorer, groups, {"art": "doc"}, str(out))
    assert seen == ["B"]
    assert [json.loads(line)["item_id"] for line in out.read_text().splitlines()] == [
        "a",
        "b",
    ]


def test_timing_survives_a_resume(tmp_path):
    out = tmp_path / "x.jsonl"
    groups = {"a1": [{"item_id": "a", "claim": "A"}, {"item_id": "b", "claim": "B"}]}
    sc.run("x", lambda doc, claims: [0.5] * len(claims), groups, {"a1": "d"}, str(out))
    rows = [
        {"item_id": "a", "article_id": "a1", "score": 1, "seconds": 2.0},
        {"item_id": "b", "article_id": "a1", "score": 1, "seconds": 2.0},
        {"item_id": "c", "article_id": "a2", "score": 1, "seconds": 1.0},
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sc.run("x", lambda d, c: [], groups, {}, str(out)) == 2.5
