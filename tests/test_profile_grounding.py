"""Tests for profile grounding verification (Item 15).

These tests verify:
- Citation extraction from profile text (deterministic, no LLM)
- Grounding report structure and scoring
- Missing source handling
- Cloud vs local LLM routing
- Batch post-processing integration (hash-based skip, entity mutation)
"""

from typing import Dict, List
from unittest.mock import patch

from src.utils.quality_controls import (
    ClaimVerification,
    GroundingReport,
    SupportLevel,
    _extract_cited_claims,
    verify_profile_grounding,
)

# ── Citation extraction tests (deterministic) ──


class TestExtractCitedClaims:
    def test_single_citation(self):
        text = "Alice was detained at Guantánamo.^[art-001]"
        claims = _extract_cited_claims(text)
        assert len(claims) == 1
        assert claims[0]["article_id"] == "art-001"
        assert claims[0]["citation"] == "^[art-001]"
        assert "detained" in claims[0]["claim"]

    def test_multiple_citations(self):
        text = "Alice was detained.^[art-001] She was released in 2015.^[art-002]"
        claims = _extract_cited_claims(text)
        assert len(claims) == 2
        assert claims[0]["article_id"] == "art-001"
        assert claims[1]["article_id"] == "art-002"
        assert "released" in claims[1]["claim"]

    def test_adjacent_citations_reuse_claim(self):
        text = "Alice was detained.^[art-001]^[art-002]"
        claims = _extract_cited_claims(text)
        assert len(claims) == 2
        # Second citation should reuse the claim from the first
        assert claims[0]["claim"] == claims[1]["claim"]

    def test_no_citations(self):
        text = "Just plain text without any citations."
        claims = _extract_cited_claims(text)
        assert len(claims) == 0

    def test_citation_at_start(self):
        text = "^[art-001] Some text follows."
        claims = _extract_cited_claims(text)
        assert len(claims) == 1
        # No text before the first citation
        assert claims[0]["claim"] == "(no claim text)"

    def test_complex_article_ids(self):
        text = "Claim text.^[abc-123-xyz]"
        claims = _extract_cited_claims(text)
        assert claims[0]["article_id"] == "abc-123-xyz"


# ── GroundingReport schema tests ──


class TestGroundingReportSchema:
    def test_default_values(self):
        report = GroundingReport()
        assert report.total_citations == 0
        assert report.verified == 0
        assert report.passed is True
        assert report.flags == []

    def test_grounding_score_computation(self):
        report = GroundingReport(
            total_citations=4,
            verified=3,
            unverified=1,
            grounding_score=0.75,
        )
        assert report.grounding_score == 0.75

    def test_support_level_enum(self):
        v = ClaimVerification(
            article_id="a1",
            citation="^[a1]",
            claim="test",
            support_level="supported",
        )
        assert v.support_level == SupportLevel.SUPPORTED


# ── verify_profile_grounding tests ──


class TestVerifyProfileGrounding:
    def test_no_citations_returns_early(self):
        """Profile without citations should not trigger LLM calls."""
        with patch("src.utils.llm.cloud_generation") as mock_cloud:
            report = verify_profile_grounding(
                profile_text="Just plain text, no citations here.",
                article_texts={},
            )
        mock_cloud.assert_not_called()
        assert report.total_citations == 0
        assert "no_citations" in report.flags
        assert report.grounding_score is None

    def test_missing_source_handling(self):
        """Citations referencing unavailable articles should be flagged."""
        supported_result = [
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="known claim",
                support_level=SupportLevel.SUPPORTED,
                reasoning="found in text",
            )
        ]
        with patch(
            "src.utils.llm.cloud_generation",
            return_value=supported_result,
        ):
            report = verify_profile_grounding(
                profile_text="Known claim.^[a1] Missing ref.^[missing-id]",
                article_texts={"a1": "Source text about the known claim."},
            )

        assert report.total_citations == 2
        assert report.missing_source == 1
        assert report.verified == 1
        assert "missing_sources" in report.flags

    def test_cloud_generation_called_for_gemini(self):
        """model_type='gemini' should use cloud_generation."""
        result = [
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="test",
                support_level=SupportLevel.SUPPORTED,
            )
        ]
        with (
            patch(
                "src.utils.llm.cloud_generation",
                return_value=result,
            ) as mock_cloud,
            patch(
                "src.utils.llm.local_generation",
            ) as mock_local,
        ):
            verify_profile_grounding(
                profile_text="Claim.^[a1]",
                article_texts={"a1": "Source text"},
                model_type="gemini",
            )
        mock_cloud.assert_called_once()
        mock_local.assert_not_called()

    def test_local_generation_called_for_ollama(self):
        """model_type='ollama' should use local_generation."""
        result = [
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="test",
                support_level=SupportLevel.SUPPORTED,
            )
        ]
        with (
            patch(
                "src.utils.llm.cloud_generation",
            ) as mock_cloud,
            patch(
                "src.utils.llm.local_generation",
                return_value=result,
            ) as mock_local,
        ):
            verify_profile_grounding(
                profile_text="Claim.^[a1]",
                article_texts={"a1": "Source text"},
                model_type="ollama",
            )
        mock_local.assert_called_once()
        mock_cloud.assert_not_called()

    def test_all_supported_gives_perfect_score(self):
        """When all claims are supported, grounding_score should be 1.0."""
        result = [
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="first",
                support_level=SupportLevel.SUPPORTED,
            ),
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="second",
                support_level=SupportLevel.PARTIAL,
            ),
        ]
        with patch(
            "src.utils.llm.cloud_generation",
            return_value=result,
        ):
            report = verify_profile_grounding(
                profile_text="First claim.^[a1] Second claim.^[a1]",
                article_texts={"a1": "Source text"},
            )
        assert report.grounding_score == 1.0
        assert report.passed is True

    def test_low_score_flags_and_fails(self):
        """Score below threshold should flag low_grounding_score and set passed=False."""
        result = [
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="unsupported",
                support_level=SupportLevel.NOT_SUPPORTED,
            ),
        ]
        with patch(
            "src.utils.llm.cloud_generation",
            return_value=result,
        ):
            report = verify_profile_grounding(
                profile_text="Unsupported claim.^[a1]",
                article_texts={"a1": "Unrelated source text"},
                min_grounding_score=0.7,
            )
        assert report.grounding_score == 0.0
        assert report.passed is False
        assert "low_grounding_score" in report.flags
        assert "unsupported_claims" in report.flags

    def test_api_error_returns_report_with_flags(self):
        """LLM errors should produce unclear verifications, not raise."""
        with patch(
            "src.utils.llm.cloud_generation",
            side_effect=RuntimeError("API down"),
        ):
            report = verify_profile_grounding(
                profile_text="Some claim.^[a1]",
                article_texts={"a1": "Source text"},
            )
        assert report.total_citations == 1
        assert report.unverified == 1
        assert "verification_error" in report.flags

    def test_profile_text_hash_included(self):
        """Report should include a SHA-256 hash of the profile text."""
        import hashlib

        text = "Claim.^[a1]"
        expected_hash = hashlib.sha256(text.encode()).hexdigest()
        result = [
            ClaimVerification(
                article_id="a1",
                citation="^[a1]",
                claim="Claim",
                support_level=SupportLevel.SUPPORTED,
            )
        ]
        with patch(
            "src.utils.llm.cloud_generation",
            return_value=result,
        ):
            report = verify_profile_grounding(
                profile_text=text,
                article_texts={"a1": "Source"},
            )
        assert report.profile_text_hash == expected_hash


# ── Batch post-processing integration tests ──


class TestGroundingPostprocess:
    def test_adds_grounding_to_entity(self):
        """Post-processing should add profile_grounding to entities with citations."""
        from src.engine.article_processor import ArticleProcessor
        from src.process_and_extract import run_profile_grounding_postprocess

        entities = {
            "people": {
                "Alice Smith": {
                    "name": "Alice Smith",
                    "profile": {"text": "Alice was detained.^[art-1]"},
                }
            },
            "organizations": {},
            "locations": {},
            "events": {},
        }
        rows = [
            {"id": "art-1", "clean_text": "Source article about Alice being detained."}
        ]

        mock_report = GroundingReport(
            profile_text_hash="abc123",
            total_citations=1,
            verified=1,
            grounding_score=1.0,
        )

        with (
            patch.object(ArticleProcessor, "__init__", lambda self, **kw: None),
            patch.object(
                ArticleProcessor,
                "prepare_article_info",
                return_value={
                    "id": "art-1",
                    "title": "T",
                    "url": "#",
                    "published_date": "",
                    "content": "Source article about Alice.",
                },
            ),
            patch(
                "src.process_and_extract.verify_profile_grounding",
                return_value=mock_report,
            ) as mock_verify,
        ):
            processor = ArticleProcessor.__new__(ArticleProcessor)
            counts = run_profile_grounding_postprocess(
                entities=entities,
                rows=rows,
                processor=processor,
                model_type="gemini",
            )

        assert counts["verified"] == 1
        assert "profile_grounding" in entities["people"]["Alice Smith"]
        mock_verify.assert_called_once()

    def test_skips_unchanged_profile(self):
        """Entities with matching profile_text_hash should be skipped."""
        import hashlib

        from src.engine.article_processor import ArticleProcessor
        from src.process_and_extract import run_profile_grounding_postprocess

        profile_text = "Alice was detained.^[art-1]"
        text_hash = hashlib.sha256(profile_text.encode()).hexdigest()

        entities = {
            "people": {
                "Alice Smith": {
                    "name": "Alice Smith",
                    "profile": {"text": profile_text},
                    "profile_grounding": {
                        "profile_text_hash": text_hash,
                        "grounding_score": 1.0,
                    },
                }
            },
            "organizations": {},
            "locations": {},
            "events": {},
        }
        rows = [{"id": "art-1", "clean_text": "Source."}]

        with (
            patch.object(ArticleProcessor, "__init__", lambda self, **kw: None),
            patch.object(
                ArticleProcessor,
                "prepare_article_info",
                return_value={
                    "id": "art-1",
                    "title": "T",
                    "url": "#",
                    "published_date": "",
                    "content": "Source.",
                },
            ),
            patch(
                "src.process_and_extract.verify_profile_grounding",
            ) as mock_verify,
        ):
            processor = ArticleProcessor.__new__(ArticleProcessor)
            counts = run_profile_grounding_postprocess(
                entities=entities,
                rows=rows,
                processor=processor,
                model_type="gemini",
            )

        assert counts["skipped_unchanged"] == 1
        assert counts["verified"] == 0
        mock_verify.assert_not_called()

    def test_skips_entities_without_citations(self):
        """Entities whose profile has no citations should be skipped."""
        from src.engine.article_processor import ArticleProcessor
        from src.process_and_extract import run_profile_grounding_postprocess

        entities = {
            "people": {
                "Alice Smith": {
                    "name": "Alice Smith",
                    "profile": {"text": "No citations here."},
                }
            },
            "organizations": {},
            "locations": {},
            "events": {},
        }
        rows: List[Dict] = []

        with (
            patch.object(ArticleProcessor, "__init__", lambda self, **kw: None),
            patch.object(
                ArticleProcessor,
                "prepare_article_info",
                return_value={
                    "id": "",
                    "title": "",
                    "url": "",
                    "published_date": "",
                    "content": "",
                },
            ),
            patch(
                "src.process_and_extract.verify_profile_grounding",
            ) as mock_verify,
        ):
            processor = ArticleProcessor.__new__(ArticleProcessor)
            counts = run_profile_grounding_postprocess(
                entities=entities,
                rows=rows,
                processor=processor,
                model_type="gemini",
            )

        assert counts["skipped_no_citations"] == 1
        mock_verify.assert_not_called()
