"""Tests for reusing a stored "not relevant" verdict instead of re-checking.

These tests verify:
- An unchanged article with a stored "not relevant" verdict skips the LLM call
- The verdict is not reused when the text, model or relevance prompt changed,
  with --force-reprocess, with skip_if_unchanged off, or for old entries
  that have no stored hashes
- A "not relevant" skip stores the hashes needed for reuse, end to end
  through a real ProcessingStatus file
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.engine import ArticleProcessor
from src.process_and_extract import (
    ArticleExtractionResult,
    extract_single_article_only,
    merge_and_finalize,
)
from src.utils.cache_utils import sha256_text
from src.utils.outcomes import PhaseOutcome
from src.utils.processing_status import ProcessingStatus

CONTENT = "Article about a bake sale in Ohio"
ROW = {"id": "art-001", "title": "Bake sale", "content": CONTENT}


def _args(**overrides):
    base = {"force_reprocess": False, "relevance_check": True, "verbose": False}
    return SimpleNamespace(**{**base, **overrides})


def _processor() -> ArticleProcessor:
    return ArticleProcessor(domain="guantanamo", model_type="gemini")


def _stored_verdict(processor: ArticleProcessor, **overrides):
    entry = {
        "processed": False,
        "reason": "not_relevant",
        "content_hash": sha256_text(CONTENT),
        "relevance_fingerprint": processor.relevance_fingerprint(),
    }
    return {"art-001": {**entry, **overrides}}


def _extract(processor, snapshot, *, args=None, skip_if_unchanged=True):
    """Run extraction with a relevance check that always says 'not relevant'."""
    check = MagicMock(return_value=PhaseOutcome.ok("relevance", value=False))
    with (
        patch.object(processor, "check_relevance", check),
        patch.object(processor, "extract_all_entities", MagicMock(return_value={})),
    ):
        result = extract_single_article_only(
            dict(ROW),
            1,
            processor,
            args or _args(),
            status_snapshot=snapshot,
            skip_if_unchanged=skip_if_unchanged,
        )
    return result, check


def _counters():
    return {
        "processed_count": 0,
        "skipped_relevance_count": 0,
        "skipped_relevance_error": 0,
        "skipped_relevance_cached": 0,
        "skipped_already_processed": 0,
        "skipped_no_content": 0,
    }


def _finalize(result, tracker):
    counters = _counters()
    merge_and_finalize(
        result,
        entities={},
        processor=MagicMock(),
        args=_args(),
        domain_config=MagicMock(),
        status_tracker=tracker,
        counters=counters,
    )
    return counters


class TestVerdictReuse:
    def test_unchanged_article_reuses_verdict_without_llm_call(self):
        processor = _processor()
        result, check = _extract(processor, _stored_verdict(processor))

        check.assert_not_called()
        assert result.skip_reason == "not_relevant_cached"
        assert result.was_extracted is False

    def test_changed_text_is_checked_again(self):
        processor = _processor()
        snapshot = _stored_verdict(processor, content_hash=sha256_text("old text"))
        result, check = _extract(processor, snapshot)

        check.assert_called_once()
        assert result.skip_reason == "not_relevant"

    def test_changed_model_or_prompt_is_checked_again(self):
        processor = _processor()
        snapshot = _stored_verdict(processor, relevance_fingerprint="stale")
        _, check = _extract(processor, snapshot)

        check.assert_called_once()

    def test_force_reprocess_checks_again(self):
        processor = _processor()
        _, check = _extract(
            processor, _stored_verdict(processor), args=_args(force_reprocess=True)
        )

        check.assert_called_once()

    def test_skip_if_unchanged_off_checks_again(self):
        processor = _processor()
        _, check = _extract(
            processor, _stored_verdict(processor), skip_if_unchanged=False
        )

        check.assert_called_once()

    def test_old_entry_without_hashes_checks_again(self):
        _, check = _extract(
            _processor(), {"art-001": {"processed": False, "reason": "not_relevant"}}
        )

        check.assert_called_once()

    def test_relevance_error_is_never_reused(self):
        processor = _processor()
        snapshot = _stored_verdict(processor, reason="relevance_error")
        _, check = _extract(processor, snapshot)

        check.assert_called_once()


class TestFingerprint:
    def test_fingerprint_changes_with_prompt(self):
        with patch(
            "src.engine.article_processor.get_system_prompt", return_value="prompt A"
        ):
            fp_a = _processor().relevance_fingerprint()
        with patch(
            "src.engine.article_processor.get_system_prompt", return_value="prompt B"
        ):
            fp_b = _processor().relevance_fingerprint()

        assert fp_a != fp_b

    def test_fingerprint_changes_with_model(self):
        cloud = ArticleProcessor(domain="guantanamo", model_type="gemini")
        local = ArticleProcessor(domain="guantanamo", model_type="ollama")

        assert cloud.relevance_fingerprint() != local.relevance_fingerprint()


class TestRoundTripThroughStatusFile:
    def test_second_run_skips_llm_call(self, tmp_path):
        """Run 1 checks and stores the verdict; run 2 reads it back and reuses it."""
        # Run 1: no stored status, the LLM says "not relevant"
        tracker = ProcessingStatus(str(tmp_path))
        result, check = _extract(_processor(), tracker.snapshot())
        check.assert_called_once()
        assert result.skip_reason == "not_relevant"
        _finalize(result, tracker)
        tracker.flush()

        stored = tracker.snapshot()["art-001"]
        assert stored["reason"] == "not_relevant"
        assert stored["content_hash"] == sha256_text(CONTENT)
        assert stored["relevance_fingerprint"]

        # Run 2: a fresh tracker loads the file; no LLM call this time
        tracker2 = ProcessingStatus(str(tmp_path))
        result2, check2 = _extract(_processor(), tracker2.snapshot())
        check2.assert_not_called()
        assert result2.skip_reason == "not_relevant_cached"

        counters = _finalize(result2, tracker2)
        assert counters["skipped_relevance_cached"] == 1
        assert counters["skipped_relevance_count"] == 0
        # The stored verdict survives for run 3
        assert tracker2.snapshot()["art-001"] == stored

    def test_cached_skip_does_not_rewrite_status(self):
        processor = _processor()
        result = ArticleExtractionResult(
            row_index=1,
            row=dict(ROW),
            article_info=processor.prepare_article_info(dict(ROW), 1),
            processing_metadata={},
            extraction_timestamp="2025-01-01T00:00:00",
            extracted_entities={},
            was_extracted=False,
            skip_reason="not_relevant_cached",
        )
        tracker = MagicMock()
        _finalize(result, tracker)

        tracker.mark_skipped.assert_not_called()
