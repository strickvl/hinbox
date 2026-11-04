"""Spelling variants and ALL CAPS names through the real ``EntityMerger`` flow."""

from unittest.mock import patch

from src.engine import EntityMerger, MatchCheckResult
from tests.test_entity_merger_merge_smoke import (
    StubEmbeddingManager,
    create_profile_stub,
    make_empty_entities,
    update_profile_stub,
)


def _entity(name, vec, **extra):
    return {
        "name": name,
        "profile_embedding": vec,
        "profile": {"text": "p"},
        "articles": [],
        **extra,
    }


def _merge(entity_type, incoming, entities, is_match):
    asked = []

    def check(new_name, existing_name, *args, **kwargs):
        asked.append(existing_name)
        return MatchCheckResult(is_match=is_match, confidence=0.95, reason="stub")

    with (
        patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
        patch("src.engine.mergers.update_profile", side_effect=update_profile_stub),
        patch(
            "src.engine.mergers.get_embedding_manager",
            # Far from the stored vectors: only the name-variant route finds them
            return_value=StubEmbeddingManager(vec=[1.0, 0, 0]),
        ),
        patch("src.engine.mergers.cloud_model_check_match", side_effect=check),
    ):
        stats = EntityMerger(entity_type).merge_entities(
            [incoming],
            entities,
            article_id="a1",
            article_title="t",
            article_url="u",
            article_published_date="2025-01-01",
            article_content="text",
            extraction_timestamp="x",
            similarity_threshold=0.8,
        )
    return stats, asked


def test_respelled_first_name_reaches_the_match_check_and_merges():
    entities = make_empty_entities()
    entities["people"] = {"Clayton Trivett": _entity("Clayton Trivett", [0, 1.0, 0])}
    stats, asked = _merge("people", {"name": "Clay Trivett"}, entities, True)
    assert asked == ["Clayton Trivett"]
    assert stats.merged == 1 and stats.new == 0
    assert list(entities["people"]) == ["Clayton Trivett"]


def test_rejected_spelling_variant_becomes_its_own_entity():
    entities = make_empty_entities()
    entities["people"] = {"Mark Martins": _entity("Mark Martins", [0, 1.0, 0])}
    stats, asked = _merge("people", {"name": "Mary Martins"}, entities, False)
    assert asked == ["Mark Martins"]
    assert stats.merged == 0 and stats.new == 1
    assert set(entities["people"]) == {"Mark Martins", "Mary Martins"}


def test_normal_case_name_replaces_an_all_caps_key_on_merge():
    entities = make_empty_entities()
    caps = ("GUANTANAMO BAY NAVY BASE", "military_base")
    entities["locations"] = {caps: _entity(caps[0], [0, 1.0, 0], type="military_base")}
    incoming = {"name": "Guantanamo Bay Navy Base", "type": "military_base"}
    stats, _ = _merge("locations", incoming, entities, True)
    assert stats.merged == 1
    assert list(entities["locations"]) == [
        ("Guantanamo Bay Navy Base", "military_base")
    ]
    merged = entities["locations"][("Guantanamo Bay Navy Base", "military_base")]
    assert merged["name"] == "Guantanamo Bay Navy Base"


def test_lone_first_name_merges_into_the_only_person_with_it():
    """'Lara' is far from 'Lara Tur' on cosine and isn't a surname; she is the
    only Lara, so the match check is asked and its yes merges them."""
    entities = make_empty_entities()
    entities["people"] = {"Lara Tur": _entity("Lara Tur", [0, 1.0, 0])}
    stats, asked = _merge("people", {"name": "Lara"}, entities, True)
    assert asked == ["Lara Tur"]
    assert stats.merged == 1 and stats.new == 0
    assert list(entities["people"]) == ["Lara Tur"]


def test_rejected_lone_first_name_becomes_its_own_entity():
    entities = make_empty_entities()
    entities["people"] = {"Lara Tur": _entity("Lara Tur", [0, 1.0, 0])}
    stats, asked = _merge("people", {"name": "Lara"}, entities, False)
    assert asked == ["Lara Tur"]  # asked once, not retried after the no
    assert stats.merged == 0 and stats.new == 1
    assert set(entities["people"]) == {"Lara Tur", "Lara"}


def test_lone_first_name_shared_by_two_people_is_not_checked():
    entities = make_empty_entities()
    entities["people"] = {
        n: _entity(n, [0, 1.0, 0]) for n in ("John Kelly", "John Kerry")
    }
    stats, asked = _merge("people", {"name": "John"}, entities, True)
    assert asked == []
    assert stats.new == 1
