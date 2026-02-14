"""Tests for merge dispute agent and gray-band routing in EntityMerger.

These tests verify:
- Dispute agent is NOT called when similarity is far from threshold or confidence is high
- Dispute agent CAN override a match-check "no match" to merge
- Dispute agent CAN override a match-check "match" to skip
- Dispute agent DEFER action results in a skip
- Review queue is written for DEFER decisions
- MergeDisputeDecision schema basics
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.engine import (
    EntityMerger,
    MatchCheckResult,
    VersionedProfile,
)
from src.engine.merge_dispute_agent import (
    MergeDisputeAction,
    MergeDisputeDecision,
    append_merge_dispute_review_queue,
    run_merge_dispute_agent,
)

# ── Test helpers (mirrors test_entity_merger_merge_smoke.py) ──


class StubEmbeddingResult:
    def __init__(self, vec: List[float], model: str = "stub-model"):
        self.embeddings = [list(vec)]
        self.model = model
        self.dimension = len(vec) if vec else None


class _StubMode:
    def __init__(self, value: str = "local"):
        self.value = value


class StubEmbeddingManager:
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
    return {"people": {}, "organizations": {}, "locations": {}, "events": {}}


def create_profile_stub(
    entity_type,
    entity_name,
    article_text,
    article_id,
    model_type="gemini",
    domain="guantanamo",
):
    profile_dict = {
        "text": f"Generated profile for {entity_name} from {article_id}",
        "confidence": 0.85,
        "tags": ["stub"],
        "sources": [article_id],
    }
    vp = VersionedProfile()
    vp.add_version(profile_dict, trigger_article_id=article_id)
    return profile_dict, vp, []


def update_profile_stub(
    entity_type,
    entity_name,
    existing_profile,
    versioned_profile,
    new_article_text,
    new_article_id,
    model_type="gemini",
    domain="guantanamo",
):
    updated_profile = {
        "text": f"Updated profile for {entity_name} from {new_article_id}",
        "confidence": 0.9,
        "tags": ["updated"],
        "sources": sorted(
            list(set(existing_profile.get("sources", []) + [new_article_id]))
        ),
    }
    versioned_profile.add_version(updated_profile, trigger_article_id=new_article_id)
    return updated_profile, versioned_profile, [{"iteration": 1, "valid": True}]


def _make_existing_entity(name: str, article_id: str = "art-000") -> Dict[str, Any]:
    """Create a pre-populated entity dict with one article and one profile version."""
    profile = {
        "text": f"Original profile for {name}",
        "confidence": 0.7,
        "tags": ["orig"],
        "sources": [article_id],
    }
    versioned = VersionedProfile()
    versioned.add_version(profile, trigger_article_id=article_id)
    return {
        "name": name,
        "articles": [
            {
                "article_id": article_id,
                "article_title": "Previous Article",
                "article_url": "http://example.com/0",
                "article_published_date": "2024-12-31",
            }
        ],
        "profile": profile,
        "profile_versions": versioned.model_dump(),
        "profile_embedding": [0.5, 0.5, 0.0],
        "profile_embedding_model": "stub-model",
        "profile_embedding_dim": 3,
        "profile_embedding_fingerprint": "stub-model:3",
        "extraction_timestamp": "2025-01-01T00:00:00Z",
        "alternative_names": [],
        "reflection_history": [],
    }


# ── Schema tests ──


class TestMergeDisputeDecisionSchema:
    def test_merge_action(self):
        d = MergeDisputeDecision(action="merge", confidence=0.9, reason="same person")
        assert d.action == MergeDisputeAction.MERGE

    def test_skip_action(self):
        d = MergeDisputeDecision(action="skip", confidence=0.8, reason="different")
        assert d.action == MergeDisputeAction.SKIP

    def test_defer_action(self):
        d = MergeDisputeDecision(action="defer", confidence=0.3, reason="ambiguous")
        assert d.action == MergeDisputeAction.DEFER

    def test_confidence_validation(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MergeDisputeDecision(action="merge", confidence=1.5, reason="test")
        with pytest.raises(ValidationError):
            MergeDisputeDecision(action="skip", confidence=-0.1, reason="test")

    def test_invalid_action(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MergeDisputeDecision(action="invalid", confidence=0.5, reason="test")


# ── Routing tests ──


class TestDisputeAgentRouting:
    """Verify that the dispute agent is called only in gray-band + uncertain cases."""

    def _run_merge_with_mocks(
        self,
        entities,
        match_result: MatchCheckResult,
        dispute_decision: MergeDisputeDecision = None,
        similarity_score: float = 0.77,
        similarity_threshold: float = 0.75,
    ):
        """Run merge_entities with full patching, returning the dispute mock for assertions."""
        merger = EntityMerger("people")
        stub_manager = StubEmbeddingManager(vec=[0.5, 0.5, 0.0])

        dispute_mock = (
            MagicMock(return_value=dispute_decision)
            if dispute_decision
            else MagicMock()
        )

        def always_return_match_result(*args, **kwargs):
            return match_result

        with (
            patch("src.engine.mergers.create_profile", side_effect=create_profile_stub),
            patch("src.engine.mergers.update_profile", side_effect=update_profile_stub),
            patch(
                "src.engine.mergers.get_embedding_manager", return_value=stub_manager
            ),
            patch(
                "src.engine.mergers.cloud_model_check_match",
                side_effect=always_return_match_result,
            ),
            patch(
                "src.engine.mergers.local_model_check_match",
                side_effect=always_return_match_result,
            ),
            patch("src.engine.mergers.run_merge_dispute_agent", dispute_mock),
            patch.object(
                merger,
                "find_similar_entity",
                return_value=("Alice Smith", similarity_score),
            ),
        ):
            merger.merge_entities(
                [{"name": "Alice B. Smith"}],
                entities,
                article_id="art-002",
                article_title="Test Article",
                article_url="http://example.com/2",
                article_published_date="2025-02-01",
                article_content="Article about Alice B. Smith",
                extraction_timestamp="2025-02-02T00:00:00Z",
                model_type="gemini",
                similarity_threshold=similarity_threshold,
                domain="guantanamo",
            )

        return dispute_mock

    def test_dispute_not_called_when_similarity_far_from_threshold(self):
        """High similarity, high confidence → no dispute routing."""
        entities = make_empty_entities()
        entities["people"]["Alice Smith"] = _make_existing_entity("Alice Smith")

        match_result = MatchCheckResult(
            is_match=True, confidence=0.95, reason="clear match"
        )
        dispute_mock = self._run_merge_with_mocks(
            entities,
            match_result,
            similarity_score=0.92,  # far above threshold
            similarity_threshold=0.75,
        )
        dispute_mock.assert_not_called()

    def test_dispute_not_called_when_confidence_high(self):
        """Similarity in gray band but confidence above cutoff → no dispute routing."""
        entities = make_empty_entities()
        entities["people"]["Alice Smith"] = _make_existing_entity("Alice Smith")

        match_result = MatchCheckResult(
            is_match=True, confidence=0.85, reason="confident match"
        )
        dispute_mock = self._run_merge_with_mocks(
            entities,
            match_result,
            similarity_score=0.77,  # in gray band (0.75 ± 0.05)
            similarity_threshold=0.75,
        )
        dispute_mock.assert_not_called()

    def test_dispute_called_when_gray_band_and_uncertain(self):
        """Similarity in gray band AND low confidence → dispute agent should be called."""
        entities = make_empty_entities()
        entities["people"]["Alice Smith"] = _make_existing_entity("Alice Smith")

        match_result = MatchCheckResult(
            is_match=True, confidence=0.55, reason="uncertain"
        )
        dispute_decision = MergeDisputeDecision(
            action=MergeDisputeAction.MERGE,
            confidence=0.85,
            reason="same person after review",
        )
        dispute_mock = self._run_merge_with_mocks(
            entities,
            match_result,
            dispute_decision=dispute_decision,
            similarity_score=0.77,
            similarity_threshold=0.75,
        )
        dispute_mock.assert_called_once()

    def test_dispute_overrides_no_match_to_merge(self):
        """Match checker says no, dispute agent says merge → entity should be updated."""
        entities = make_empty_entities()
        entities["people"]["Alice Smith"] = _make_existing_entity("Alice Smith")

        match_result = MatchCheckResult(
            is_match=False, confidence=0.55, reason="uncertain no-match"
        )
        dispute_decision = MergeDisputeDecision(
            action=MergeDisputeAction.MERGE,
            confidence=0.88,
            reason="profiles clearly overlap",
        )
        self._run_merge_with_mocks(
            entities,
            match_result,
            dispute_decision=dispute_decision,
            similarity_score=0.74,  # just below threshold (gray band)
            similarity_threshold=0.75,
        )

        person = entities["people"]["Alice Smith"]
        article_ids = [a["article_id"] for a in person["articles"]]
        assert "art-002" in article_ids, (
            "Dispute agent override to merge should add the new article"
        )

    def test_dispute_overrides_match_to_skip(self):
        """Match checker says yes, dispute agent says skip → entity should NOT be updated."""
        entities = make_empty_entities()
        entities["people"]["Alice Smith"] = _make_existing_entity("Alice Smith")

        match_result = MatchCheckResult(
            is_match=True, confidence=0.55, reason="uncertain match"
        )
        dispute_decision = MergeDisputeDecision(
            action=MergeDisputeAction.SKIP,
            confidence=0.80,
            reason="contradictory details",
        )
        self._run_merge_with_mocks(
            entities,
            match_result,
            dispute_decision=dispute_decision,
            similarity_score=0.77,
            similarity_threshold=0.75,
        )

        person = entities["people"]["Alice Smith"]
        article_ids = [a["article_id"] for a in person["articles"]]
        assert "art-002" not in article_ids, "Dispute agent skip should prevent merge"

    def test_dispute_defer_results_in_skip(self):
        """Defer action should be treated as skip (don't merge when unsure)."""
        entities = make_empty_entities()
        entities["people"]["Alice Smith"] = _make_existing_entity("Alice Smith")

        match_result = MatchCheckResult(
            is_match=True, confidence=0.55, reason="uncertain"
        )
        dispute_decision = MergeDisputeDecision(
            action=MergeDisputeAction.DEFER, confidence=0.3, reason="too ambiguous"
        )
        self._run_merge_with_mocks(
            entities,
            match_result,
            dispute_decision=dispute_decision,
            similarity_score=0.77,
            similarity_threshold=0.75,
        )

        person = entities["people"]["Alice Smith"]
        article_ids = [a["article_id"] for a in person["articles"]]
        assert "art-002" not in article_ids, "Defer should prevent merge"


# ── Review queue tests ──


class TestReviewQueuePersistence:
    def test_append_creates_file_and_writes_record(self, tmp_path):
        path = str(tmp_path / "review_queue" / "disputes.jsonl")
        record = {"test": "data", "value": 42}
        append_merge_dispute_review_queue(path, record)

        import json

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["test"] == "data"

    def test_append_is_additive(self, tmp_path):
        path = str(tmp_path / "disputes.jsonl")
        append_merge_dispute_review_queue(path, {"record": 1})
        append_merge_dispute_review_queue(path, {"record": 2})

        import json

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["record"] == 1
        assert json.loads(lines[1])["record"] == 2


# ── run_merge_dispute_agent unit tests ──


class TestRunMergeDisputeAgentUnit:
    def test_returns_decision_from_cloud_generation(self):
        """Cloud path should call cloud_generation and return the structured result."""
        expected = MergeDisputeDecision(
            action=MergeDisputeAction.MERGE, confidence=0.9, reason="same entity"
        )
        with patch(
            "src.engine.merge_dispute_agent.cloud_generation", return_value=expected
        ):
            result = run_merge_dispute_agent(
                entity_type="people",
                new_name="Alice Smith",
                existing_name="A. Smith",
                new_profile_text="Profile A",
                existing_profile_text="Profile B",
                similarity_score=0.76,
                similarity_threshold=0.75,
                match_is_match=True,
                match_confidence=0.55,
                match_reason="uncertain",
                model_type="gemini",
            )
        assert result.action == MergeDisputeAction.MERGE
        assert result.confidence == 0.9

    def test_returns_decision_from_local_generation(self):
        """Local path should call local_generation for ollama model_type."""
        expected = MergeDisputeDecision(
            action=MergeDisputeAction.SKIP, confidence=0.8, reason="different people"
        )
        with patch(
            "src.engine.merge_dispute_agent.local_generation", return_value=expected
        ):
            result = run_merge_dispute_agent(
                entity_type="people",
                new_name="Alice Smith",
                existing_name="Bob Jones",
                new_profile_text="Profile A",
                existing_profile_text="Profile B",
                similarity_score=0.76,
                similarity_threshold=0.75,
                match_is_match=False,
                match_confidence=0.4,
                match_reason="uncertain",
                model_type="ollama",
            )
        assert result.action == MergeDisputeAction.SKIP

    def test_api_error_returns_defer_fallback(self):
        """On LLM API error, should return DEFER with confidence=0.0."""
        with patch(
            "src.engine.merge_dispute_agent.cloud_generation",
            side_effect=RuntimeError("API down"),
        ):
            result = run_merge_dispute_agent(
                entity_type="people",
                new_name="Alice",
                existing_name="Alice S.",
                new_profile_text="P1",
                existing_profile_text="P2",
                similarity_score=0.76,
                similarity_threshold=0.75,
                match_is_match=True,
                match_confidence=0.5,
                match_reason="test",
            )
        assert result.action == MergeDisputeAction.DEFER
        assert result.confidence == 0.0
        assert "API error" in result.reason

    def test_defer_writes_to_review_queue(self, tmp_path):
        """When decision is DEFER and review_queue_path is provided, write to queue."""
        queue_path = str(tmp_path / "queue.jsonl")
        defer_result = MergeDisputeDecision(
            action=MergeDisputeAction.DEFER, confidence=0.3, reason="ambiguous"
        )
        with patch(
            "src.engine.merge_dispute_agent.cloud_generation", return_value=defer_result
        ):
            run_merge_dispute_agent(
                entity_type="people",
                new_name="Alice",
                existing_name="Alice S.",
                new_profile_text="P1",
                existing_profile_text="P2",
                similarity_score=0.76,
                similarity_threshold=0.75,
                match_is_match=True,
                match_confidence=0.5,
                match_reason="test",
                review_queue_path=queue_path,
                article_id="art-99",
            )

        import json

        with open(queue_path) as f:
            record = json.loads(f.readline())
        assert record["new_key"] == "Alice"
        assert record["candidate_key"] == "Alice S."
        assert record["article_id"] == "art-99"
        assert record["dispute_decision"]["action"] == "defer"
