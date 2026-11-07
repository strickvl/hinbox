"""After the match check rejects the closest candidate, the next-closest above
the threshold is checked (up to MERGE_MAX_SIMILAR_CHECKS), then a name variant.

Scenario from the 7 Nov 150-article run: an incoming event was closest to a
sibling event from the same story; the match check rightly said "different",
and the real event, second-closest, was never checked.
"""

from typing import List
from unittest.mock import patch

from src.engine import EntityMerger, MatchCheckResult
from src.engine.merge_candidates import next_candidate, rejected_note
from tests.test_entity_merger_merge_smoke import (
    StubEmbeddingManager,
    create_profile_stub,
    make_empty_entities,
    update_profile_stub,
)


def _event(name: str, vec: List[float]) -> dict:
    return {
        "title": name,
        "event_type": "other",
        "profile_embedding": vec,
        "profile": {"text": f"{name}."},
        "articles": [],
    }


SIBLING = ("Nashiri's defense lawyers' resignation accepted", "other")
REAL = ("Resignation of Nashiri's defense lawyers", "other")
FAR = ("Nashiri's defense lawyers' hearing postponed", "other")


def _entities() -> dict:
    entities = make_empty_entities()
    entities["events"] = {
        SIBLING: _event(SIBLING[0], [0.95, 0.312, 0]),
        REAL: _event(REAL[0], [0.9, 0.436, 0]),
        FAR: _event(FAR[0], [0.85, 0, 0.527]),
    }
    return entities


def _merge(entities: dict, accept: str, asked: List[str]):
    def check(new_name, existing_name, *args, **kwargs):
        asked.append(existing_name)
        return MatchCheckResult(
            is_match=existing_name == accept, confidence=0.95, reason="stub"
        )

    with (
        patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
        patch(
            "src.engine.profile_stubs.update_profile", side_effect=update_profile_stub
        ),
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(vec=[1.0, 0, 0]),
        ),
        patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
    ):
        return EntityMerger("events").merge_entities(
            [{"title": "Nashiri's defense lawyers resign", "event_type": "other"}],
            entities,
            article_id="a1",
            article_title="t",
            article_url="u",
            article_published_date="2025-01-01",
            article_content="Nashiri's defense lawyers resign.",
            extraction_timestamp="x",
            similarity_threshold=0.8,
        )


def test_second_closest_is_checked_after_sibling_is_rejected():
    entities, asked = _entities(), []
    stats = _merge(entities, accept=REAL[0], asked=asked)
    assert asked == [SIBLING[0], REAL[0]]
    assert stats.merged == 1 and stats.new == 0
    assert len(entities["events"]) == 3


def test_every_candidate_rejected_creates_new_entity():
    entities, asked = _entities(), []
    stats = _merge(entities, accept="nobody", asked=asked)
    assert asked == [SIBLING[0], REAL[0], FAR[0]]
    assert stats.new == 1 and len(entities["events"]) == 4


def test_one_similar_check_restores_old_behaviour():
    entities, asked = _entities(), []
    with (
        patch("src.engine.mergers.MERGE_MAX_SIMILAR_CHECKS", 1),
        patch("src.engine.merge_candidates.MERGE_MAX_SIMILAR_CHECKS", 1),
    ):
        stats = _merge(entities, accept=REAL[0], asked=asked)
    assert asked == [SIBLING[0]]
    assert stats.new == 1


def test_below_threshold_candidates_are_not_tried_as_similar():
    entities = _entities()
    key, score = next_candidate(
        "events",
        "alternative_titles",
        ("Nashiri's defense lawyers resign", "other"),
        [1.0, 0, 0],
        entities,
        0.92,
        [SIBLING],
    )
    # REAL (0.90) and FAR (0.85) are under 0.92 and no name is a variant
    assert key is None and score is None


def test_later_candidates_are_told_a_closer_one_was_rejected():
    entities, notes = _entities(), []

    def check(new_name, existing_name, new_text, *args, **kwargs):
        notes.append("already judged a different event" in new_text)
        return MatchCheckResult(is_match=False, confidence=0.95, reason="stub")

    with (
        patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
        patch(
            "src.engine.mergers.get_embedding_manager",
            return_value=StubEmbeddingManager(vec=[1.0, 0, 0]),
        ),
        patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
    ):
        EntityMerger("events").merge_entities(
            [{"title": "Nashiri's defense lawyers resign", "event_type": "other"}],
            entities,
            article_id="a1",
            article_title="t",
            article_url="u",
            article_published_date="2025-01-01",
            article_content="Nashiri's defense lawyers resign.",
            extraction_timestamp="x",
            similarity_threshold=0.8,
        )
    assert notes == [False, True, True]


def test_rejected_note_names_the_rejected_candidates():
    assert rejected_note("events", []) == ""
    note = rejected_note("events", [SIBLING])
    assert f'"{SIBLING[0]}"' in note and "not just a related one" in note
