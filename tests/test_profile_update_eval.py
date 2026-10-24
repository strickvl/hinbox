"""Offline tests for the profile-update scoring checks (no model calls)."""

import json

from src.evals import profile_update as pu

OLD = (
    "## Background\n"
    "Nashiri was captured in Dubai in 2002 by local security forces.^[a1] "
    "He was transferred to Guantanamo Bay detention camp in 2006 with others.^[a2] "
    "His lawyer Nancy Hollander represented him before the commission.^[a3]"
)


def _case(kind="new_fact", article_id="a9", expected=None):
    return {
        "case_id": f"{kind}:x",
        "kind": kind,
        "split": "dev",
        "entity_name": "Nashiri",
        "entity_type": "person",
        "profile": {"text": OLD, "sources": ["a1", "a2", "a3"]},
        "article_id": article_id,
        "article_text": "…",
        "expected": expected or {"must_cite": article_id},
    }


def test_good_new_fact_update_passes():
    new = OLD + " A military judge set his trial date for early 2014.^[a9]"
    scores = pu.score_update(_case(), new)
    assert scores["fact_retention"] == 1.0
    assert scores["unchanged_share"] == 1.0
    assert scores["added_sentences"] == 1
    assert scores["cites_article"] and scores["passed"]


def test_collapse_and_lost_facts_fail_even_when_new_fact_is_cited():
    new = (
        "## Background\nNashiri was captured in Dubai in 2002 by local security"
        " forces.^[a1] A military judge set his trial date for early 2014.^[a9]"
    )
    scores = pu.score_update(_case(), new)
    assert scores["facts_lost"] == 2
    assert scores["length_ratio"] < pu.COLLAPSE_RATIO
    assert not scores["passed"]


def test_light_rewording_still_counts_as_kept():
    new = OLD.replace("by local security forces", "by local forces") + (
        " A military judge set his trial date for early 2014.^[a9]"
    )
    scores = pu.score_update(_case(), new)
    assert scores["fact_retention"] == 1.0
    assert scores["unchanged_share"] < 1.0


def test_invented_citation_fails():
    new = OLD + " He was moved to Camp 7 later that year.^[zz]"
    scores = pu.score_update(_case(kind="irrelevant", expected={}), new)
    assert scores["invented_citations"] == ["zz"]
    assert not scores["passed"]


def test_irrelevant_case_allows_a_small_addition_only():
    case = _case(kind="irrelevant", expected={"keep_all": True, "max_new_sentences": 2})
    assert pu.score_update(case, OLD)["passed"]
    many = OLD + "".join(
        f" Brand new statement number {w} about hearings.^[a9]"
        for w in ("alpha", "bravo", "charlie")
    )
    assert pu.score_update(case, many)["added_sentences"] == 3
    assert not pu.score_update(case, many)["passed"]


def _disputed():
    return _case(
        kind="disputed",
        article_id="a9-disputed",
        expected={
            "keep_both": True,
            "original_value": "2006",
            "disputed_value": "2007",
            "original_sources": ["a2"],
            "profile_sentence": "He was transferred to Guantanamo Bay detention"
            " camp in 2006 with others.^[a2]",
            "article_sentence": "Nashiri was transferred to Guantanamo in 2007.",
        },
    )


def test_disputed_passes_only_when_both_years_keep_their_sources():
    both = OLD.replace(
        "with others.^[a2]",
        "with others.^[a2] Another report says he was transferred to Guantanamo"
        " in 2007.^[a9-disputed]",
    )
    assert pu.score_update(_disputed(), both)["keeps_both"]
    swapped = OLD.replace(
        "in 2006 with others.^[a2]", "in 2007 with others.^[a9-disputed]"
    )
    scores = pu.score_update(_disputed(), swapped)
    assert not scores["keeps_both"] and not scores["passed"]


def test_disputed_ignores_an_unrelated_sentence_with_the_same_year_and_source():
    # The transfer year was silently swapped; a different 2006 claim citing
    # the same source must not count as keeping the original
    swapped = (
        OLD.replace("in 2006 with others.^[a2]", "in 2007 with others.^[a9-disputed]")
        + " His lawyer Nancy Hollander joined the defence team in 2006.^[a2]"
    )
    assert not pu.score_update(_disputed(), swapped)["keeps_both"]


def test_pipeline_updater_reports_a_fallback_to_the_old_profile(monkeypatch):
    import src.engine.profiles as profiles

    monkeypatch.setattr(profiles, "update_profile", lambda *a, **kw: (a[2], a[3], []))
    update = pu.make_pipeline_updater("openai/x", "guantanamo")
    record = pu.run_updates([_case()], update)[0]
    assert "old profile came back" in record["error"] and "scores" not in record


def test_pipeline_updater_accepts_a_word_for_word_no_op_update(monkeypatch):
    import src.engine.profiles as profiles

    def no_op(*a, **kw):
        old, article_id = a[2], a[5]
        return {**old, "sources": old["sources"] + [article_id]}, a[3], []

    monkeypatch.setattr(profiles, "update_profile", no_op)
    update = pu.make_pipeline_updater("openai/x", "guantanamo")
    case = _case(kind="irrelevant", expected={"max_new_sentences": 2})
    record = pu.run_updates([case], update)[0]
    assert record["scores"]["passed"]


def test_load_cases_filters_split_and_unreviewed(tmp_path):
    rows = [
        {**_case(), "case_id": "1", "split": "dev"},
        {**_case(), "case_id": "2", "split": "test"},
        {**_disputed(), "case_id": "3", "needs_review": True},
    ]
    (tmp_path / "profile_update_cases.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    assert [c["case_id"] for c in pu.load_cases(str(tmp_path), "dev")] == ["1"]
    assert len(pu.load_cases(str(tmp_path), "all", include_unreviewed=True)) == 3


def test_run_updates_counts_errors_as_failures():
    def updater(case):
        if case["case_id"] == "boom":
            raise RuntimeError("API down")
        return OLD + " A military judge set his trial date for early 2014.^[a9]"

    cases = [_case(), {**_case(), "case_id": "boom"}]
    records = pu.run_updates(cases, updater, workers=2)
    assert records[1]["error"] == "RuntimeError: API down"
    summary = pu.summarize_updates(records)
    assert summary["errors"] == 1 and summary["pass_rate"] == 0.5
    assert summary["pass_new_fact"] == "1/2"


def test_collapse_is_tested_before_rounding():
    case = {**_case(), "profile": {"text": "x" * 10000, "sources": []}}
    scores = pu.score_update(case, "x" * 7999)
    assert scores["length_ratio"] == 0.8 and scores["collapsed"]


def test_retention_is_tested_before_rounding():
    kept = [
        f"Distinct claim number {i} about hearings happened.^[a1]" for i in range(188)
    ]
    lost = [
        f"Separate statement {i} regarding detainees occurred.^[a2]" for i in range(21)
    ]
    case = _case(kind="irrelevant", expected={"max_new_sentences": 2})
    case["profile"] = {"text": "\n".join(kept + lost), "sources": ["a1", "a2"]}
    scores = pu.score_update(case, "\n".join(kept) + "\n" + "x" * 3000)
    assert scores["fact_retention"] == 0.9 and scores["low_retention"]
    assert not scores["passed"]


def test_disputed_accepts_a_grouped_original_citation():
    case = _disputed()
    case["expected"]["original_sources"] = ["a2, a3"]  # as older case files stored it
    both = OLD.replace(
        "with others.^[a2]",
        "with others.^[a2, a3] Another report says he was transferred to"
        " Guantanamo in 2007.^[a9-disputed]",
    )
    assert pu.score_update(case, both)["keeps_both"]
