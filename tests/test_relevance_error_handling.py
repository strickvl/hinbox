"""Tests that a failed relevance check is kept separate from "not relevant".

These tests verify:
- check_relevance returns a failed PhaseOutcome when the LLM call raises
- A failed check skips extraction with reason "relevance_error", not "not_relevant"
- A real "not relevant" verdict still skips with reason "not_relevant"
- The merge actor counts and records the two skip reasons separately
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.engine import ArticleProcessor
from src.process_and_extract import (
    ArticleExtractionResult,
    extract_single_article_only,
    merge_and_finalize,
)
from src.utils.outcomes import PhaseOutcome

ROW = {"id": "art-001", "title": "Hearing", "content": "Guantanamo hearing text"}
ARGS = SimpleNamespace(force_reprocess=False, relevance_check=True, verbose=False)


def _extract_with_relevance(outcome: PhaseOutcome):
    """Run extract_single_article_only with a forced relevance outcome."""
    processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
    extract_mock = MagicMock(return_value={})
    with (
        patch.object(processor, "check_relevance", return_value=outcome),
        patch.object(processor, "extract_all_entities", extract_mock),
    ):
        result = extract_single_article_only(
            dict(ROW), 1, processor, ARGS, status_snapshot={}
        )
    return result, extract_mock


def _counters():
    return {
        "processed_count": 0,
        "skipped_relevance_count": 0,
        "skipped_relevance_error": 0,
        "skipped_already_processed": 0,
        "skipped_no_content": 0,
    }


class TestCheckRelevanceOutcome:
    def test_llm_error_returns_failed_outcome(self):
        processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
        with patch(
            "src.engine.article_processor.gemini_check_relevance",
            side_effect=RuntimeError("API down"),
        ):
            outcome = processor.check_relevance("some text", "art-001")

        assert outcome.success is False
        assert outcome.value is False
        assert "API down" in outcome.error.message


class TestExtractSkipReasons:
    def test_failed_check_skips_as_relevance_error(self):
        failed = PhaseOutcome.fail(
            "relevance", error=RuntimeError("API down"), fallback=False
        )
        result, extract_mock = _extract_with_relevance(failed)

        assert result.was_extracted is False
        assert result.skip_reason == "relevance_error"
        extract_mock.assert_not_called()
        assert "API down" in result.processing_metadata["reason"]
        assert result.processing_metadata["processed"] is False

    def test_not_relevant_verdict_still_skips_as_not_relevant(self):
        result, extract_mock = _extract_with_relevance(
            PhaseOutcome.ok("relevance", value=False, meta={"reason": "off-topic"})
        )

        assert result.skip_reason == "not_relevant"
        assert result.processing_metadata["reason"] == "Not relevant"
        extract_mock.assert_not_called()

    def test_relevant_verdict_extracts(self):
        result, extract_mock = _extract_with_relevance(
            PhaseOutcome.ok("relevance", value=True)
        )

        assert result.was_extracted is True
        extract_mock.assert_called_once()


class TestMergeActorSkipAccounting:
    def _skipped(self, reason: str) -> ArticleExtractionResult:
        processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
        return ArticleExtractionResult(
            row_index=1,
            row=dict(ROW),
            article_info=processor.prepare_article_info(dict(ROW), 1),
            processing_metadata={},
            extraction_timestamp="2025-01-01T00:00:00",
            extracted_entities={},
            was_extracted=False,
            skip_reason=reason,
        )

    def _finalize(self, reason: str):
        counters = _counters()
        tracker = MagicMock()
        merge_and_finalize(
            self._skipped(reason),
            entities={},
            processor=MagicMock(),
            args=ARGS,
            domain_config=MagicMock(),
            status_tracker=tracker,
            counters=counters,
        )
        return counters, tracker

    def test_relevance_error_counted_and_recorded_separately(self):
        counters, tracker = self._finalize("relevance_error")

        assert counters["skipped_relevance_error"] == 1
        assert counters["skipped_relevance_count"] == 0
        tracker.mark_skipped.assert_called_once_with("art-001", "relevance_error")

    def test_not_relevant_counted_as_before(self):
        counters, tracker = self._finalize("not_relevant")

        assert counters["skipped_relevance_count"] == 1
        assert counters["skipped_relevance_error"] == 0
        tracker.mark_skipped.assert_called_once_with("art-001", "not_relevant")
