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


def test_replay_entities_have_fixed_splits_covering_every_type_in_train():
    from src.evals.profile_replay import DEFAULT_ENTITIES

    splits = {e.name: pc.ENTITY_SPLITS[e.name] for e in DEFAULT_ENTITIES}
    assert sorted(splits.values()).count("train") == 6
    train_types = {e.entity_type for e in DEFAULT_ENTITIES if splits[e.name] == "train"}
    assert train_types == {"person", "organization", "location", "event"}


def test_sentences_do_not_split_at_abbreviations():
    text = (
        "Rear Adm. Harris said the U.S. Navy agreed.^[a1] George W. Bush spoke"
        " on Oct. 12, 2000.^[a2]\n## Header"
    )
    assert pc._sentences(text) == [
        "Rear Adm. Harris said the U.S. Navy agreed.^[a1]",
        "George W. Bush spoke on Oct. 12, 2000.^[a2]",
        "## Header",
    ]


def test_disputed_year_needs_matching_month_and_day():
    profile = (
        "Al Qaeda attacked the destroyer USS The Sullivans in January 2000"
        " without success.^[s1]"
    )
    article = "Al Qaeda attacked the destroyer USS Cole on Oct. 12, 2000 in Aden."
    assert pc.make_disputed_article(article, profile, random.Random(0)) is None
    same = "Al Qaeda attacked the destroyer USS Cole in October 2000 in Aden."
    profile_cole = "Al Qaeda attacked the destroyer USS Cole on Oct. 12, 2000.^[s1]"
    assert pc.make_disputed_article(same, profile_cole, random.Random(0))
    byline = "The report on prison detainees was published April 29, 2013."
    profile_report = "A prison detainees report was published April 13, 2013.^[s1]"
    assert pc.make_disputed_article(byline, profile_report, random.Random(0)) is None


def test_cited_years_splits_grouped_citations():
    years = pc.cited_years("He was moved to Guantanamo in 2006.^[a1, a2]")
    assert years["2006"][0][1] == ["a1", "a2"]


def test_irrelevant_draws_do_not_depend_on_disputed_cases(monkeypatch):
    manifest = {
        "entity": {"name": "X", "entity_type": "person", "terms": ["X"]},
        "model": "m",
        "articles": [{"id": f"a{i}"} for i in range(1, 6)],
        "passing_mentions": [{"id": f"p{i}"} for i in range(8)],
    }
    steps = {k: {"step": k, "profile": {"text": PROFILE}} for k in (1, 2, 3, 4)}
    texts = {
        f"a{i}": "Emirati agents captured X in Dubai in 2002." for i in range(1, 6)
    }
    texts.update({f"p{i}": "X, briefly." for i in range(8)})

    def irrelevant_ids():
        cases = pc.build_cases(manifest, steps, texts, at_steps=(1, 2, 3, 4))
        return [c["article_id"] for c in cases if c["kind"] == "irrelevant"]

    before = irrelevant_ids()
    monkeypatch.setattr(pc, "make_disputed_article", lambda *a, **kw: None)
    assert irrelevant_ids() == before
