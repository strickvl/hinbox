"""Tests for EntityMerger.find_similar_entity similarity search logic."""

import pytest

from src.engine import EntityMerger


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

    def test_dimension_mismatch_scan_skips_incompatible(self):
        """Scan should skip entities whose embedding dimension doesn't match."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        # Existing entity has 384-dim embedding (like MiniLM)
        entities["people"]["Alice Smith"] = {
            "profile_embedding": [0.1] * 384,
            "profile_embedding_dim": 384,
            "profile_embedding_model": "local-model",
            "profile": {"text": "Profile for Alice"},
        }

        # New embedding is 1024-dim (like Jina v3) — incompatible
        new_embedding = [0.1] * 1024

        match_key, score = merger.find_similar_entity(
            entity_key="Bob Jones",
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.5,
            embedding_model="cloud-model",
            embedding_dim=1024,
        )

        # Should find no match because dimensions differ
        assert match_key is None
        assert score is None

    def test_dimension_mismatch_exact_key_defers_to_match_check(self):
        """Exact-key match with incompatible dims should return forced score=1.0."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        # Existing entity with 384-dim embedding
        entities["people"]["Alice Smith"] = {
            "profile_embedding": [0.1] * 384,
            "profile_embedding_dim": 384,
            "profile_embedding_model": "local-model",
            "profile": {"text": "Profile for Alice"},
        }

        # Same key but different dimension — should defer to match-check
        new_embedding = [0.2] * 1024

        match_key, score = merger.find_similar_entity(
            entity_key="Alice Smith",
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.5,
            embedding_model="cloud-model",
            embedding_dim=1024,
        )

        assert match_key == "Alice Smith"
        assert score == 1.0  # forced score for exact-key dim mismatch

    def test_model_mismatch_skips_scan(self):
        """Scan should skip entities whose embedding model name differs."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        # Same dimension but different model names
        entities["people"]["Alice Smith"] = {
            "profile_embedding": [0.6, 0.8, 0.0],
            "profile_embedding_dim": 3,
            "profile_embedding_model": "model-A",
            "profile": {"text": "Profile for Alice"},
        }

        new_embedding = [0.6, 0.8, 0.0]  # identical vector

        match_key, score = merger.find_similar_entity(
            entity_key="Bob Jones",
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.5,
            embedding_model="model-B",
            embedding_dim=3,
        )

        # Should skip because model names differ
        assert match_key is None
        assert score is None

    def test_lexical_blocking_excludes_dissimilar_names(self):
        """Lexical blocking should exclude candidates with dissimilar names, even if embeddings match."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        # Candidate 1: lexically similar name, high embedding similarity
        entities["people"]["Alice Smithson"] = {
            "profile_embedding": [0.6, 0.8, 0.0],
            "profile": {"text": "Profile for Alice Smithson"},
        }
        # Candidate 2: lexically dissimilar, but identical embedding (cosine=1.0)
        entities["people"]["Zzyxvut Qqqppp"] = {
            "profile_embedding": [0.6, 0.8, 0.0],
            "profile": {"text": "Profile for Zzyxvut"},
        }

        # With lexical blocking ON at threshold 60, "Zzyxvut" should be excluded
        match_key, score = merger.find_similar_entity(
            entity_key="Alice Smith",
            entity_embedding=[0.6, 0.8, 0.0],
            entities=entities,
            similarity_threshold=0.5,
            lexical_blocking_config={
                "enabled": True,
                "threshold": 60,
                "max_candidates": 50,
            },
        )

        assert match_key == "Alice Smithson"
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_lexical_blocking_disabled_returns_best_embedding_match(self):
        """When lexical blocking is disabled, the best embedding match wins regardless of name."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        entities["people"]["Alice Smithson"] = {
            "profile_embedding": [0.5, 0.5, 0.0],  # similarity ~0.98
            "profile": {"text": "Profile for Alice Smithson"},
        }
        entities["people"]["Zzyxvut Qqqppp"] = {
            "profile_embedding": [0.6, 0.8, 0.0],  # similarity = 1.0
            "profile": {"text": "Profile for Zzyxvut"},
        }

        # With lexical blocking OFF, best embedding match wins
        match_key, score = merger.find_similar_entity(
            entity_key="Alice Smith",
            entity_embedding=[0.6, 0.8, 0.0],
            entities=entities,
            similarity_threshold=0.5,
            lexical_blocking_config={"enabled": False},
        )

        assert match_key == "Zzyxvut Qqqppp"
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_lexical_blocking_events_uses_title_only(self):
        """For events, lexical blocking should match on title only, ignoring date."""
        merger = EntityMerger("events")
        entities = make_empty_entities()

        # Same title, different date — should still pass lexical blocking
        entities["events"][("Detention hearing", "2025-01-01")] = {
            "profile_embedding": [0.6, 0.8, 0.0],
            "profile": {"text": "Detention hearing event"},
        }
        # Different title — should be excluded by lexical blocking
        entities["events"][("Budget committee review", "2025-03-01")] = {
            "profile_embedding": [0.6, 0.8, 0.0],
            "profile": {"text": "Budget review event"},
        }

        match_key, score = merger.find_similar_entity(
            entity_key=("Detention hearing", "2025-03-01"),
            entity_embedding=[0.6, 0.8, 0.0],
            entities=entities,
            similarity_threshold=0.5,
            lexical_blocking_config={
                "enabled": True,
                "threshold": 60,
                "max_candidates": 50,
            },
        )

        assert match_key == ("Detention hearing", "2025-01-01")
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_lexical_blocking_max_candidates(self):
        """Lexical blocking should respect max_candidates limit."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        # Create 10 candidates with similar names
        for i in range(10):
            entities["people"][f"Alice Smith {i}"] = {
                "profile_embedding": [0.6, 0.8, 0.0],
                "profile": {"text": f"Profile {i}"},
            }

        match_key, score = merger.find_similar_entity(
            entity_key="Alice Smith",
            entity_embedding=[0.6, 0.8, 0.0],
            entities=entities,
            similarity_threshold=0.5,
            lexical_blocking_config={
                "enabled": True,
                "threshold": 30,
                "max_candidates": 3,
            },
        )

        # Should still find a match (one of the 3 shortlisted candidates)
        assert match_key is not None
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_backward_compat_no_metadata(self):
        """Entities without embedding metadata should still be comparable."""
        merger = EntityMerger("people")
        entities = make_empty_entities()

        # Old-style entity without metadata fields
        entities["people"]["Alice Smith"] = {
            "profile_embedding": [0.6, 0.8, 0.0],
            "profile": {"text": "Profile for Alice"},
            # No profile_embedding_model or profile_embedding_dim
        }

        new_embedding = [0.6, 0.8, 0.0]

        match_key, score = merger.find_similar_entity(
            entity_key="Bob Jones",
            entity_embedding=new_embedding,
            entities=entities,
            similarity_threshold=0.5,
            # No model/dim passed — backward compat
        )

        assert match_key == "Alice Smith"
        assert score == pytest.approx(1.0, rel=1e-6)
