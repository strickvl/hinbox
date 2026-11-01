"""Spelling-variant candidates: accents, "County", surnames, retry after a rejection.

Pairs come from the hand-labelled merge gold set (merge_gold_v2.jsonl).
"""

from unittest.mock import patch

import pytest

from src.engine import EntityMerger, MatchCheckResult
from src.engine.merge_candidates import name_variant_candidate
from src.utils.name_variants import names_are_variants, normalize_for_match
from tests.test_entity_merger_merge_smoke import (
    StubEmbeddingManager,
    create_profile_stub,
    make_empty_entities,
    update_profile_stub,
)

GTMO_GROUPS = [["Guantanamo Bay", "Guantanamo", "GTMO", "Gitmo"]]


def test_normalize_for_match_drops_accents_and_curly_quotes():
    assert normalize_for_match("Guantánamo Bay") == "guantanamo bay"
    assert normalize_for_match("Sheriff’s Office") == normalize_for_match(
        "Sheriff's Office"
    )


@pytest.mark.parametrize(
    "a, b, entity_type",
    [
        ("Guantánamo Bay", "Guantanamo Bay", "locations"),
        ("Broward", "Broward County", "locations"),
        ("Miami-Dade", "Miami-Dade County", "locations"),
        ("New York", "New York City", "locations"),
        ("Navy", "U.S. Navy", "organizations"),
        ("Gitmo", "Guantánamo Bay", "locations"),
        ("Trump", "Donald Trump", "people"),
        ("President Trump", "Donald J. Trump", "people"),
        ("Donald Trump", "Donald J. Trump", "people"),
        ("Khalid Sheik Mohammed", "Khalid Shaikh Mohammed", "people"),
        ("Duro", "Marcos Duro", "people"),
    ],
)
def test_labelled_same_pairs_are_variants(a, b, entity_type):
    assert names_are_variants(
        a, b, entity_type=entity_type, equivalence_groups=GTMO_GROUPS
    )


@pytest.mark.parametrize(
    "a, b, entity_type",
    [
        ("Guantanamo Bay Navy Base", "Guantanamo Bay", "locations"),
        ("Cuba", "southeast Cuba", "locations"),
        ("Florida", "South Florida", "locations"),
        ("Brevard County", "Leon County", "locations"),
        ("Leon County Sheriff's Office", "Leon County", "locations"),
        ("Palm Beach County", "West Palm Beach", "locations"),
        ("U.S. military", "U.S. Navy", "organizations"),
        ("David Rosenberg", "Mark Rosenberg", "people"),
        ("Leon", "Lisa Leonard", "people"),
        ("Daniel", "Daniel Harding", "people"),  # a lone first name
    ],
)
def test_labelled_different_pairs_are_not_variants(a, b, entity_type):
    assert not names_are_variants(
        a, b, entity_type=entity_type, equivalence_groups=GTMO_GROUPS
    )


def _loc(vec, **extra):
    return {"profile_embedding": vec, "profile": {"text": "p"}, **extra}


class TestNameVariantCandidate:
    def test_accent_variant_below_threshold_reaches_match_check(self):
        entities = make_empty_entities()
        entities["locations"] = {
            ("Guantanamo Bay", "military_base"): _loc([0.5, 0.866, 0])
        }
        key, score = EntityMerger("locations").find_similar_entity(
            ("Guantánamo Bay", "other"), [1.0, 0.0, 0.0], entities, 0.8
        )
        assert key == ("Guantanamo Bay", "military_base")
        assert score == pytest.approx(0.5, rel=1e-3)

    def test_surname_reaches_match_check(self):
        entities = make_empty_entities()
        entities["people"] = {"Donald Trump": _loc([0.3, 0.954, 0])}
        key, _ = EntityMerger("people").find_similar_entity(
            "Trump", [1.0, 0.0, 0.0], entities, 0.82
        )
        assert key == "Donald Trump"

    def test_variant_outside_the_capped_lexical_shortlist_is_found(self):
        """The shortlist keeps the first N in dict order; the variant may be cut."""
        entities = make_empty_entities()
        entities["locations"] = {
            (f"Guantanamo Camp {i}", "other"): _loc([0, 1.0, 0]) for i in range(5)
        }
        entities["locations"][("Guantanamo Bay County", "region")] = _loc(
            [0, 0.9, 0.436]
        )
        key, _ = EntityMerger("locations").find_similar_entity(
            ("Guantanamo Bay", "region"),
            [1.0, 0.0, 0.0],
            entities,
            0.8,
            lexical_blocking_config={
                "enabled": True,
                "threshold": 30,
                "max_candidates": 2,
            },
        )
        assert key == ("Guantanamo Bay County", "region")

    def test_exclude_skips_rejected_candidates(self):
        entities = make_empty_entities()
        entities["people"] = {
            "Donald Trump": _loc([1.0, 0, 0]),
            "Ivanka Trump": _loc([0.8, 0.6, 0]),
        }
        key, _ = name_variant_candidate(
            "people",
            "alternative_names",
            "Trump",
            [1.0, 0, 0],
            entities,
            exclude=("Donald Trump",),
        )
        assert key == "Ivanka Trump"


def test_rejected_best_candidate_falls_back_to_name_variant():
    """'Guantánamo Bay' is closest to the Navy Base on cosine; the match check
    says no, so the name-variant 'Guantanamo Bay' is tried and merged."""
    entities = make_empty_entities()
    base = ("Guantanamo Bay Navy Base", "military_base")
    bay = ("Guantanamo Bay", "military_base")
    entities["locations"] = {
        base: {
            "name": base[0],
            "type": base[1],
            "profile_embedding": [1.0, 0, 0],
            "profile": {"text": "The base."},
            "articles": [],
        },
        bay: {
            "name": bay[0],
            "type": bay[1],
            "profile_embedding": [0.5, 0.866, 0],
            "profile": {"text": "The bay."},
            "articles": [],
        },
    }
    asked = []

    def check(new_name, existing_name, *args, **kwargs):
        asked.append(existing_name)
        is_bay = existing_name == "Guantanamo Bay"
        return MatchCheckResult(is_match=is_bay, confidence=0.95, reason="stub")

    with (
        patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
        patch("src.engine.mergers.update_profile", side_effect=update_profile_stub),
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(vec=[1.0, 0, 0]),
        ),
        patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
    ):
        stats = EntityMerger("locations").merge_entities(
            [{"name": "Guantánamo Bay", "type": "military_base"}],
            entities,
            article_id="a1",
            article_title="t",
            article_url="u",
            article_published_date="2025-01-01",
            article_content="Guantánamo Bay.",
            extraction_timestamp="x",
            similarity_threshold=0.8,
        )
    assert asked == ["Guantanamo Bay Navy Base", "Guantanamo Bay"]
    assert stats.merged == 1 and stats.new == 0
    # One entity for the bay (whichever spelling wins the canonical name),
    # and the base untouched
    assert len(entities["locations"]) == 2
    assert entities["locations"][base]["articles"] == []
