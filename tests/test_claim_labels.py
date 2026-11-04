"""Offline tests for claim labels and the stricter new_fact check (no model calls)."""

import json

from src.evals import claim_labels as cl
from src.evals import profile_update as pu

OLD = (
    "## Background\n"
    "Camp 7 holds former CIA captives.^[a1] "
    "It is run by a unit called Task Force Platinum.^[a2]"
)
LABEL = {
    "entity_name": "Camp 7",
    "article_id": "a9",
    "status": "reviewed",
    "facts": [
        {"id": "routine", "core": True, "match": [["routine"], ["search"]]},
        {"id": "custodian", "core": False, "match": [["custodian"]]},
    ],
    "must_not": [
        {"kind": "wrong_actor", "note": "", "match": [["prison lawyer walter ruiz"]]}
    ],
}


def _case(label=LABEL):
    case = {
        "case_id": "new_fact:Camp 7:10:a9",
        "kind": "new_fact",
        "split": "test",
        "entity_name": "Camp 7",
        "entity_type": "location",
        "profile": {"text": OLD, "sources": ["a1", "a2"]},
        "article_id": "a9",
        "article_text": "…",
        "expected": {"must_cite": "a9"},
    }
    if label:
        case["claim_label"] = label
    return case


def test_matches_needs_every_group_and_word_starts():
    groups = [["routine"], ["search"]]
    assert cl.matches("The prison said searches were routine.^[a9]", groups)
    assert not cl.matches("Searches happened daily.", groups)
    # "search" must start a word: "research" is not a search
    assert not cl.matches("Routine research continued.", groups)


def test_numbers_match_whole():
    assert cl.matches("It left 55 captives.", [["55"]])
    assert not cl.matches("He was Detainee 552.", [["55"]])


def test_fact_counts_only_in_a_sentence_citing_the_article():
    uncited = OLD + " Searches at Camp 7 are routine.^[a1]"
    assert cl.score_claims(LABEL, OLD, uncited, "a9")["core_missing"] == ["routine"]
    cited = OLD + " Searches at Camp 7 are routine.^[a1, a9]"
    scores = cl.score_claims(LABEL, OLD, cited, "a9")
    assert scores["claims_found"] == ["routine"] and scores["claims_ok"]


def test_vague_citation_no_longer_passes_a_labelled_new_fact_case():
    vague = OLD + " Camp 7 is described as a secretive prison camp.^[a9]"
    assert pu.score_update(_case(label=None), vague)["passed"]  # old rule
    scores = pu.score_update(_case(), vague)
    assert scores["cites_article"] and scores["core_missing"] == ["routine"]
    assert not scores["passed"]


def test_good_update_passes_and_minor_fact_is_optional():
    good = OLD + " The prison lawyer said searches at Camp 7 are routine.^[a9]"
    scores = pu.score_update(_case(), good)
    assert scores["passed"] and scores["claims_found"] == ["routine"]


def test_wrong_reading_fails_only_in_new_sentences():
    wrong = (
        OLD + " Defense attorneys questioned prison lawyer Walter Ruiz; he said"
        " searches are routine.^[a9]"
    )
    scores = pu.score_update(_case(), wrong)
    assert scores["must_not_hits"] and not scores["passed"]
    # The same words already in the old profile are not this update's mistake
    old = OLD + " Prison lawyer Walter Ruiz spoke.^[a1]"
    case = _case()
    case["profile"] = {"text": old, "sources": ["a1", "a2"]}
    new = old + " Searches at Camp 7 are routine.^[a9]"
    assert not pu.score_update(case, new)["must_not_hits"]


def test_invented_dispute_fails_unless_the_article_reports_one():
    disputed = (
        OLD + " Searches at Camp 7 are routine.^[a9] The sources disagree on"
        " who runs the camp.^[a9]"
    )
    assert not pu.score_update(_case(), disputed)["passed"]
    assert pu.score_update(_case({**LABEL, "disputes_ok": True}), disputed)["passed"]


def test_load_cases_attaches_reviewed_labels_only_by_default(tmp_path):
    case = _case(label=None)
    (tmp_path / "profile_update_cases.jsonl").write_text(json.dumps(case) + "\n")
    rows = [{**LABEL, "status": "draft"}]
    (tmp_path / cl.LABELS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "claim_label" not in pu.load_cases(str(tmp_path))[0]
    assert "claim_label" in pu.load_cases(str(tmp_path), claim_labels="draft")[0]
    assert "claim_label" not in pu.load_cases(str(tmp_path), claim_labels="off")[0]


def test_rescore_run_uses_todays_checks():
    vague = OLD + " Camp 7 is described as a secretive prison camp.^[a9]"
    case = _case()
    saved = [
        {
            "item_id": case["case_id"],
            "kind": "new_fact",
            "seconds": 1.0,
            "new_text": vague,
            "scores": {"passed": True},
        },
        {"item_id": "gone", "kind": "new_fact", "seconds": 1.0},
    ]
    out = pu.rescore_run(saved, [case])
    assert len(out) == 1 and not out[0]["scores"]["passed"]
