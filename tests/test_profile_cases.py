"""Offline tests for building profile-update test cases (no model calls)."""

import random

from src.evals import profile_cases as pc

PROFILE = (
    "### Background\nHe was captured in Dubai by Emirati agents in 2002.^[a1] "
    "He was moved in 2006.^[a2]^[a3]\n\nHe has a lawyer named Reyes."
)


def test_cited_years_maps_years_to_their_cited_sentences():
    years = pc.cited_years(PROFILE)
    assert [ids for _, ids in years["2002"]] == [["a1"]]
    assert [ids for _, ids in years["2006"]] == [["a2", "a3"]]


def test_disputed_article_changes_one_shared_year_that_appears_once():
    article = (
        "Emirati agents captured him in Dubai in 2002. In 2011 he was charged. "
        "2011 was busy."
    )
    out = pc.make_disputed_article(article, PROFILE, random.Random(0))
    assert out["original_value"] == "2002"
    assert out["disputed_value"] in {"2001", "2003"}
    assert out["original_sources"] == ["a1"]
    assert "2002" not in out["text"] and "2011 was busy" in out["text"]
    assert (
        "captured" in out["article_sentence"] and "2002" not in out["article_sentence"]
    )


def test_same_year_for_a_different_claim_is_not_a_dispute():
    # The article's 2002 is about charging, the profile's 2002 about the capture
    article = "Prosecutors charged the suspect in 2002, officials said."
    assert pc.make_disputed_article(article, PROFILE, random.Random(0)) is None


def test_no_disputed_article_without_a_usable_year():
    # 2002 appears twice, 2011 isn't in the profile
    article = "2002 and again 2002, then 2011."
    assert pc.make_disputed_article(article, PROFILE, random.Random(0)) is None


def test_build_cases_makes_each_kind_from_snapshots():
    manifest = {
        "entity": {"name": "X", "entity_type": "person", "terms": ["X"]},
        "model": "m",
        "articles": [{"id": "a1"}, {"id": "a2"}],
        "passing_mentions": [{"id": "p1"}],
    }
    steps = {1: {"step": 1, "profile": {"text": PROFILE}}}
    texts = {
        "a1": "",
        "a2": "Emirati agents captured X in Dubai in 2002.",
        "p1": "X, briefly.",
    }
    cases = list(pc.build_cases(manifest, steps, texts, at_steps=(1, 2)))
    kinds = {c["kind"]: c for c in cases}
    assert set(kinds) == {"irrelevant", "new_fact", "disputed"}
    assert kinds["irrelevant"]["article_id"] == "p1"
    assert kinds["irrelevant"]["expected"]["keep_all"] is True
    assert kinds["new_fact"]["expected"] == {"must_cite": "a2"}
    disputed = kinds["disputed"]
    assert disputed["article_id"] == "a2-disputed"
    assert disputed["expected"]["keep_both"] is True
    assert "2002" not in disputed["article_text"]
    assert len({c["split"] for c in cases}) == 1  # split follows the entity
    assert {k: c["needs_review"] for k, c in kinds.items()} == {
        "irrelevant": False,
        "new_fact": False,
        "disputed": True,
    }


def test_irrelevant_cases_prefer_articles_from_the_profiles_timeline():
    manifest = {
        "entity": {"name": "X", "entity_type": "person", "terms": ["X"]},
        "model": "m",
        "articles": [{"id": "a1", "published_date": 100}],
        "passing_mentions": [
            {"id": "old", "published_date": 50},
            {"id": "future", "published_date": 900},
        ],
    }
    steps = {1: {"step": 1, "profile": {"text": PROFILE}}}
    texts = {"a1": "", "old": "X.", "future": "X."}
    for seed in range(5):
        cases = list(pc.build_cases(manifest, steps, texts, at_steps=(1,), seed=seed))
        assert [c["article_id"] for c in cases] == ["old"]
