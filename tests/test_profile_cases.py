"""Offline tests for building profile-update test cases (no model calls)."""

import random

from src.evals import profile_cases as pc

PROFILE = (
    "### Background\nHe was captured in 2002 in Dubai.^[a1] "
    "He was moved in 2006.^[a2]^[a3]\n\nHe has a lawyer named Reyes."
)


def test_cited_years_maps_years_to_their_sentence_citations():
    assert pc.cited_years(PROFILE) == {"2002": ["a1"], "2006": ["a2", "a3"]}


def test_disputed_article_changes_one_shared_year_that_appears_once():
    article = "In 2002 he was captured. In 2011 he was charged. 2011 was busy."
    out = pc.make_disputed_article(article, PROFILE, random.Random(0))
    assert out["original_value"] == "2002"
    assert out["disputed_value"] in {"2001", "2003"}
    assert out["original_sources"] == ["a1"]
    assert "2002" not in out["text"] and "2011 was busy" in out["text"]


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
    texts = {"a1": "", "a2": "Captured in 2002, says X.", "p1": "X, briefly."}
    cases = list(pc.build_cases(manifest, steps, texts, at_steps=(1, 2)))
    kinds = {c["kind"]: c for c in cases}
    assert set(kinds) == {"irrelevant", "new_fact", "disputed"}
    assert kinds["irrelevant"]["article_id"] == "p1"
    assert kinds["new_fact"]["expected"] == {"must_cite": "a2"}
    disputed = kinds["disputed"]
    assert disputed["article_id"] == "a2-disputed"
    assert disputed["expected"]["keep_both"] is True
    assert "2002" not in disputed["article_text"]
    assert len({c["split"] for c in cases}) == 1  # split follows the entity
