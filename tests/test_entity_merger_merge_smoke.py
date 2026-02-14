"""Smoke tests for EntityMerger.merge_entities covering creation and update paths.

These tests:
- Patch profile creation/update to deterministic stubs
- Patch embedding manager to a simple stub with embed_text_sync
- Patch cloud_model_check_match to always match in the update path
- Patch write_entity_to_file to no-op to avoid filesystem writes
- Verify key fields updated: articles, profile text, profile_versions, profile_embedding, alternative names
"""

from typing import Any, Dict, List
from unittest.mock import patch

from src.engine import EntityMerger, MatchCheckResult, VersionedProfile


class StubEmbeddingResult:
    """Minimal stub for EmbeddingResult returned by embed_text_result_sync."""

    def __init__(self, vec: List[float], model: str = "stub-model"):
        self.embeddings = [list(vec)]
        self.model = model
        self.dimension = len(vec) if vec else None


class _StubMode:
    """Minimal stub for EmbeddingMode enum value."""

    def __init__(self, value: str = "local"):
        self.value = value


class StubEmbeddingManager:
    """Minimal stub that mimics the EmbeddingManager interface needed by mergers.

    The embed_text_sync returns a fixed vector to make similarity deterministic.
    embed_text_result_sync returns a StubEmbeddingResult with model metadata.
    """

    def __init__(self, vec: List[float] = None, model: str = "stub-model"):
        self._vec = vec or [0.1, 0.2, 0.3]
        self._model = model
        self.mode = _StubMode("local")

    def embed_text_sync(self, text: str) -> List[float]:
        return list(self._vec)

    def embed_text_result_sync(self, text: str) -> StubEmbeddingResult:
        return StubEmbeddingResult(self._vec, self._model)

    def get_active_model_name(self) -> str:
        return self._model


def make_empty_entities() -> Dict[str, Dict]:
    """Helper to create an entities dict with all types initialized to empty dicts."""
    return {"people": {}, "organizations": {}, "locations": {}, "events": {}}


def create_profile_stub(
    entity_type: str,
    entity_name: str,
    article_text: str,
    article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
):
    """Deterministic create_profile stub returning a profile dict, a VersionedProfile with one version, and empty history."""
    profile_dict = {
        "text": f"Generated profile for {entity_name} from {article_id}",
        "confidence": 0.85,
        "tags": ["stub"],
        "sources": [article_id],
    }
    vp = VersionedProfile()
    vp.add_version(profile_dict, trigger_article_id=article_id)
    history: List[Dict[str, Any]] = []
    return profile_dict, vp, history


def update_profile_stub(
    entity_type: str,
    entity_name: str,
    existing_profile: Dict[str, Any],
    versioned_profile: VersionedProfile,
    new_article_text: str,
    new_article_id: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
):
    """Deterministic update_profile stub that appends a new version to the provided versioned_profile."""
    updated_profile = {
        "text": f"Updated profile for {entity_name} from {new_article_id}",
        "confidence": 0.9,
        "tags": ["updated"],
        "sources": sorted(
            list(set(existing_profile.get("sources", []) + [new_article_id]))
        ),
    }
    # Add a new version to the existing version history
    versioned_profile.add_version(updated_profile, trigger_article_id=new_article_id)
    history = [{"iteration": 1, "valid": True}]
    return updated_profile, versioned_profile, history


class TestEntityMergerMergeSmoke:
    """Smoke tests verifying creation and update behaviors for people entities."""

    def test_creation_path_people_versioning(self):
        """Creating a new person should populate core fields and write to the in-memory database."""
        entities = make_empty_entities()

        extracted_people = [
            {"name": "Alice Smith"}  # key field for people
        ]

        merger = EntityMerger("people")

        stub_manager = StubEmbeddingManager(
            vec=[0.25, 0.5, 0.75]
        )  # arbitrary but consistent

        with (
            patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
            patch(
                "src.engine.mergers.get_embedding_manager", return_value=stub_manager
            ),
            patch("src.engine.mergers.write_entity_to_file", return_value=None),
        ):
            merger.merge_entities(
                extracted_people,
                entities,
                article_id="art-001",
                article_title="First Article",
                article_url="http://example.com/1",
                article_published_date="2025-01-01",
                article_content="Article content about Alice Smith",
                extraction_timestamp="2025-01-02T00:00:00Z",
                model_type="gemini",
                similarity_threshold=0.8,
                domain="guantanamo",
            )

        # Assertions: new entity exists with expected fields
        assert "Alice Smith" in entities["people"]
        person = entities["people"]["Alice Smith"]

        # Articles
        assert isinstance(person.get("articles"), list)
        assert len(person["articles"]) == 1
        assert person["articles"][0]["article_id"] == "art-001"

        # Profile text and embedding
        assert (
            person["profile"]["text"]
            == "Generated profile for Alice Smith from art-001"
        )
        assert person["profile_embedding"] == [0.25, 0.5, 0.75]

        # Embedding metadata
        assert person["profile_embedding_model"] == "stub-model"
        assert person["profile_embedding_dim"] == 3
        assert person["profile_embedding_fingerprint"] == "stub-model:3"

        # Versioning
        assert person["profile_versions"] is not None
        assert person["profile_versions"]["current_version"] == 1
        assert len(person["profile_versions"]["versions"]) == 1
        assert (
            person["profile_versions"]["versions"][0]["trigger_article_id"] == "art-001"
        )

        # Alternative names should start empty for a brand new entity
        assert person.get("alternative_names") == []

        # Basic shape checks
        assert person["name"] == "Alice Smith"
        assert person["extraction_timestamp"] == "2025-01-02T00:00:00Z"

    def test_update_path_people_with_alternative_name_and_versioning(self):
        """Merging a similar person should update profile, add article, and record alternative name."""
        entities = make_empty_entities()

        # Create existing entity with one version
        original_profile = {
            "text": "Original profile for Alice Smith",
            "confidence": 0.7,
            "tags": ["orig"],
            "sources": ["art-000"],
        }
        versioned = VersionedProfile()
        versioned.add_version(original_profile, trigger_article_id="art-000")

        entities["people"]["Alice Smith"] = {
            "name": "Alice Smith",
            "articles": [
                {
                    "article_id": "art-000",
                    "article_title": "Previous Article",
                    "article_url": "http://example.com/0",
                    "article_published_date": "2024-12-31",
                }
            ],
            "profile": original_profile,
            "profile_versions": versioned.model_dump(),
            "profile_embedding": [0.42, 0.58, 0.0],  # baseline embedding
            "extraction_timestamp": "2025-01-01T00:00:00Z",
            "alternative_names": [],
            "reflection_history": [],
        }

        # New extracted person with a slightly different name, expected to match and update
        extracted_people = [
            {"name": "Alice B. Smith"}  # different key to trigger alternative_names
        ]

        merger = EntityMerger("people")

        # Use a stub manager that returns the SAME embedding as existing to guarantee similarity=1.0
        stub_manager = StubEmbeddingManager(vec=[0.42, 0.58, 0.0])

        def always_match(*args, **kwargs):
            return MatchCheckResult(
                is_match=True, confidence=0.95, reason="forced-match"
            )

        with (
            patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
            patch("src.engine.mergers.update_profile", side_effect=update_profile_stub),
            patch(
                "src.engine.mergers.get_embedding_manager", return_value=stub_manager
            ),
            patch(
                "src.engine.mergers.cloud_model_check_match", side_effect=always_match
            ),
            patch(
                "src.engine.match_checker.cloud_model_check_match",
                side_effect=always_match,
            ),
            patch("src.engine.mergers.write_entity_to_file", return_value=None),
        ):
            merger.merge_entities(
                extracted_people,
                entities,
                article_id="art-002",
                article_title="Update Article",
                article_url="http://example.com/2",
                article_published_date="2025-02-01",
                article_content="New article content mentioning Alice Smith",
                extraction_timestamp="2025-02-02T00:00:00Z",
                model_type="gemini",  # ensure cloud_model_check_match path
                similarity_threshold=0.5,
                domain="guantanamo",
            )

        # Assertions: existing entity updated in place
        assert "Alice Smith" in entities["people"]
        person = entities["people"]["Alice Smith"]

        # Articles updated: should include new art-002
        article_ids = [a["article_id"] for a in person["articles"]]
        assert set(article_ids) == {"art-000", "art-002"}

        # Profile updated via stub
        assert (
            person["profile"]["text"] == "Updated profile for Alice Smith from art-002"
        )

        # Versioning updated: now two versions
        assert person["profile_versions"]["current_version"] == 2
        assert len(person["profile_versions"]["versions"]) == 2
        assert (
            person["profile_versions"]["versions"][-1]["trigger_article_id"]
            == "art-002"
        )

        # Embedding updated to stub vector (from updated profile text)
        assert person["profile_embedding"] == [0.42, 0.58, 0.0]

        # Embedding metadata persisted on update
        assert person["profile_embedding_model"] == "stub-model"
        assert person["profile_embedding_dim"] == 3
        assert person["profile_embedding_fingerprint"] == "stub-model:3"

        # Alternative names added since the incoming key was different
        assert "Alice B. Smith" in person.get("alternative_names", [])


class TestMatchCheckResultSchema:
    """Ensure MatchCheckResult schema changes are backward compatible."""

    def test_default_confidence(self):
        """Constructing without confidence should default to 0.5."""
        result = MatchCheckResult(is_match=False, reason="test")
        assert result.confidence == 0.5

    def test_explicit_confidence(self):
        """Explicit confidence should be preserved."""
        result = MatchCheckResult(is_match=True, confidence=0.9, reason="test")
        assert result.confidence == 0.9

    def test_confidence_clamped_by_validator(self):
        """Confidence outside [0, 1] should raise validation error."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MatchCheckResult(is_match=True, confidence=1.5, reason="test")
        with pytest.raises(ValidationError):
            MatchCheckResult(is_match=True, confidence=-0.1, reason="test")
