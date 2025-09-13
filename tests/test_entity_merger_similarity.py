"""Tests for EntityMerger.find_similar_entity similarity search logic."""

import pytest

from src.mergers import EntityMerger


def make_empty_entities():
    """Helper to create an entities dict with all types initialized to empty dicts."""
    return {"people": {}, "organizations": {}, "locations": {}, "events": {}}


class TestEntityMergerFindSimilarEntity:
    """Focused tests for the find_similar_entity method."""

    def test_people_exact_key_similarity_match(self):
        """people (string key) -> exact-key similarity path should return the exact key when above threshold."""
        merger = EntityMerger("people")

        # New entity
        new_key = "Alice Smith"
        new_embedding = [0.6, 0.8, 0.0]  # normalized direction

        # Existing entities with exact key present and identical embedding (cosine = 1.0)
        entities = make_empty_entities()
        entities["people"] = {
            "Alice Smith": {
                "profile_embedding": [0.6, 0.8, 0.0],
                "profile": {"text": "Profile for Alice Smith"},
            },
            # Another person to ensure scan doesn't interfere when exact match is above threshold
            "Bob Jones": {
                "profile_embedding": [1.0, 0.0, 0.0],
                "profile": {"text": "Profile for Bob Jones"},
            },
        }

        match_key, score = merger.find_similar_entity(
            entity_key=new_key,
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.9,
        )

        assert match_key == "Alice Smith"
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_organizations_best_match_scan(self):
        """organizations (tuple key) -> best-match scan path should return the most similar different key."""
        merger = EntityMerger("organizations")

        new_key = ("ACME Corp", "Company")
        # New embedding is most similar to candidate "ACME Incorporated", not to the exact key
        new_embedding = [1.0, 0.0, 0.0]

        entities = make_empty_entities()
        entities["organizations"] = {
            # Exact key exists but is dissimilar to new_embedding (similarity ~ 0.0)
            ("ACME Corp", "Company"): {
                "profile_embedding": [0.0, 1.0, 0.0],
                "profile": {"text": "Exact key org"},
            },
            # Best match candidate with high similarity (cosine = 1.0)
            ("ACME Incorporated", "Company"): {
                "profile_embedding": [1.0, 0.0, 0.0],
                "profile": {"text": "Best match org"},
            },
            # Another candidate with lower similarity
            ("Globex LLC", "NGO"): {
                "profile_embedding": [0.0, 0.0, 1.0],
                "profile": {"text": "Other org"},
            },
        }

        match_key, score = merger.find_similar_entity(
            entity_key=new_key,
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.8,
        )

        assert match_key == ("ACME Incorporated", "Company")
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_events_no_match_below_threshold(self):
        """events (tuple key) -> when all similarities are below threshold, return (None, None)."""
        merger = EntityMerger("events")

        new_key = ("Hearing of evidence", "2025-03-01")
        new_embedding = [0.0, 1.0, 0.0]

        entities = make_empty_entities()
        entities["events"] = {
            ("Detention hearing", "2025-01-01"): {
                "profile_embedding": [1.0, 0.0, 0.0],  # cosine = 0.0 vs new_embedding
                "profile": {"text": "Detention hearing event"},
            },
            ("Status conference", "2025-02-01"): {
                "profile_embedding": [
                    0.5,
                    0.5,
                    0.0,
                ],  # cosine ~= 0.7071 vs new_embedding
                "profile": {"text": "Status conference event"},
            },
        }

        match_key, score = merger.find_similar_entity(
            entity_key=new_key,
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.75,  # Above the best available (~0.7071)
        )

        assert match_key is None
        assert score is None

    def test_guard_empty_embedding_returns_none(self):
        """Guard: empty embedding should return (None, None) regardless of entities content."""
        merger = EntityMerger("people")
        entities = make_empty_entities()
        # Add some entity to ensure the early return is due to empty embedding, not empty store
        entities["people"] = {
            "Bob": {"profile_embedding": [1.0, 0.0, 0.0], "profile": {"text": "Bob"}}
        }

        match_key, score = merger.find_similar_entity(
            entity_key="Bob",
            entity_embedding=[],  # Empty embedding triggers guard
            entities=entities,
            similarity_threshold=0.5,
        )

        assert match_key is None
        assert score is None

    def test_guard_empty_entities_returns_none(self):
        """Guard: empty entity store for the type should return (None, None)."""
        merger = EntityMerger("organizations")
        entities = make_empty_entities()  # organizations dict is empty

        match_key, score = merger.find_similar_entity(
            entity_key=("Any Org", "Type"),
            entity_embedding=[1.0, 0.0, 0.0],
            entities=entities,
            similarity_threshold=0.5,
        )

        assert match_key is None
        assert score is None
