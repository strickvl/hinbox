"""Tests for conditional extraction retry on QC failure (Item 13)."""

from unittest.mock import patch

import pytest

from src.engine.article_processor import (
    ArticleProcessor,
    _build_repair_hint,
    _should_retry_extraction,
)
from src.utils.quality_controls import run_extraction_qc


class TestShouldRetryExtraction:
    """Test the retry decision helper."""

    def test_zero_entities_triggers_retry(self):
        assert _should_retry_extraction(["zero_entities"]) is True

    def test_high_drop_rate_triggers_retry(self):
        assert _should_retry_extraction(["high_drop_rate"]) is True

    def test_many_duplicates_triggers_retry(self):
        assert _should_retry_extraction(["many_duplicates"]) is True

    def test_many_low_quality_names_triggers_retry(self):
        assert _should_retry_extraction(["many_low_quality_names"]) is True

    def test_non_trigger_flag_does_not_retry(self):
        assert _should_retry_extraction(["short_name:A"]) is False
        assert _should_retry_extraction(["missing_required:name"]) is False

    def test_empty_flags_does_not_retry(self):
        assert _should_retry_extraction([]) is False

    def test_mixed_flags_triggers_if_any_match(self):
        assert _should_retry_extraction(["short_name:X", "zero_entities"]) is True


class TestBuildRepairHint:
    """Test the repair hint builder."""

    def test_includes_entity_type(self):
        hint = _build_repair_hint("people", ["zero_entities"])
        assert "people" in hint

    def test_includes_relevant_flags_only(self):
        hint = _build_repair_hint(
            "events", ["zero_entities", "short_name:X", "high_drop_rate"]
        )
        assert "zero_entities" in hint
        assert "high_drop_rate" in hint
        assert "short_name" not in hint

    def test_low_quality_names_hint_includes_naming_guidance(self):
        hint = _build_repair_hint("organizations", ["many_low_quality_names"])
        assert "proper nouns" in hint
        assert "generic plurals" in hint

    def test_non_low_quality_flags_omit_naming_guidance(self):
        hint = _build_repair_hint("organizations", ["zero_entities"])
        assert "proper nouns" not in hint


# Valid entity dicts that pass QC required-field checks.
# Person requires: name, type (from Pydantic schema)
_PERSON_ALICE = {"name": "Alice Smith", "type": "detainee"}
_PERSON_BOB = {"name": "Bob Jones", "type": "military_personnel"}


class TestExtractSingleEntityTypeRetry:
    """Test the retry loop inside extract_single_entity_type."""

    @pytest.fixture
    def processor(self):
        return ArticleProcessor(domain="guantanamo", model_type="gemini")

    def test_retry_on_zero_entities_picks_v2(self, processor):
        """When v1 returns nothing and v2 returns something, v2 should be used."""
        call_count = 0

        def mock_run_extraction(extractor, content, repair_hint=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # trigger zero_entities
            return [_PERSON_ALICE.copy()]

        with patch.object(
            processor, "_run_extraction", side_effect=mock_run_extraction
        ):
            outcome = processor.extract_single_entity_type(
                "people", "Some article text", "art-1"
            )

        assert outcome.success is True
        assert outcome.meta["retry_attempted"] is True
        assert outcome.meta["retry_used"] is True
        assert outcome.counts["final_count"] == 1
        assert call_count == 2

    def test_no_retry_when_qc_passes(self, processor):
        """When v1 QC is clean, no retry should be attempted."""
        call_count = 0

        def mock_run_extraction(extractor, content, repair_hint=None):
            nonlocal call_count
            call_count += 1
            return [_PERSON_BOB.copy()]

        with patch.object(
            processor, "_run_extraction", side_effect=mock_run_extraction
        ):
            outcome = processor.extract_single_entity_type(
                "people", "Some article text", "art-2"
            )

        assert outcome.success is True
        assert outcome.meta["retry_attempted"] is False
        assert call_count == 1

    def test_retry_keeps_v1_when_v2_is_worse(self, processor):
        """When retry produces worse results, v1 should be kept."""
        call_count = 0

        def mock_run_extraction(extractor, content, repair_hint=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 3 entities, 2 missing required field 'type' → high_drop_rate
                return [
                    _PERSON_ALICE.copy(),
                    {"name": "Missing Type 1"},  # dropped
                    {"name": "Missing Type 2"},  # dropped
                ]
            else:
                # Retry: worse — 0 entities
                return []

        with patch.object(
            processor, "_run_extraction", side_effect=mock_run_extraction
        ):
            outcome = processor.extract_single_entity_type(
                "people", "Some article text", "art-3"
            )

        assert outcome.success is True
        assert outcome.meta["retry_attempted"] is True
        assert outcome.meta["retry_used"] is False
        # v1 result kept: 1 valid entity survived QC
        assert outcome.counts["final_count"] == 1

    def test_retry_passes_repair_hint_to_extractor(self, processor):
        """The retry call must include a repair_hint argument."""
        hints_received = []

        def mock_run_extraction(extractor, content, repair_hint=None):
            hints_received.append(repair_hint)
            if repair_hint is None:
                return []  # trigger retry
            return [_PERSON_ALICE.copy()]

        with patch.object(
            processor, "_run_extraction", side_effect=mock_run_extraction
        ):
            processor.extract_single_entity_type("people", "Some article text", "art-4")

        assert len(hints_received) == 2
        assert hints_received[0] is None  # first attempt: no hint
        assert hints_received[1] is not None  # retry: has hint
        assert "zero_entities" in hints_received[1]

    def test_retry_metadata_includes_trigger_flags(self, processor):
        """Retry metadata should record which flags triggered the retry."""

        def mock_run_extraction(extractor, content, repair_hint=None):
            if repair_hint is None:
                return []  # trigger zero_entities
            return [_PERSON_ALICE.copy()]

        with patch.object(
            processor, "_run_extraction", side_effect=mock_run_extraction
        ):
            outcome = processor.extract_single_entity_type(
                "people", "Some article text", "art-5"
            )

        assert "zero_entities" in outcome.meta["retry_trigger_flags"]
        assert outcome.meta["retry_output_count"] == 1

    def test_many_duplicates_triggers_retry(self, processor):
        """When more than half of entities are duplicates, retry should trigger."""
        call_count = 0

        def mock_run_extraction(extractor, content, repair_hint=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # 4 entities, 3 are dupes of the first → many_duplicates
                return [
                    _PERSON_ALICE.copy(),
                    _PERSON_ALICE.copy(),
                    _PERSON_ALICE.copy(),
                    _PERSON_ALICE.copy(),
                ]
            else:
                return [
                    _PERSON_ALICE.copy(),
                    _PERSON_BOB.copy(),
                ]

        with patch.object(
            processor, "_run_extraction", side_effect=mock_run_extraction
        ):
            outcome = processor.extract_single_entity_type(
                "people", "Some article text", "art-6"
            )

        assert outcome.meta["retry_attempted"] is True
        assert outcome.meta["retry_used"] is True
        assert outcome.counts["final_count"] == 2


class TestLowQualityNameQCFlag:
    """Test that QC flags many_low_quality_names when appropriate."""

    def test_flags_many_low_quality_org_names(self):
        """Two or more low-quality names should trigger the flag."""
        entities = [
            {"name": "Department of Defense", "type": "government"},
            {"name": "Defense departments", "type": "government"},
            {"name": "intelligence agencies", "type": "government"},
        ]
        _, report = run_extraction_qc(entity_type="organizations", entities=entities)
        assert "many_low_quality_names" in report.flags

    def test_no_flag_when_names_are_good(self):
        """All proper names should not trigger the flag."""
        entities = [
            {"name": "Department of Defense", "type": "government"},
            {"name": "FBI", "type": "intelligence"},
            {"name": "Coast Guard", "type": "military"},
        ]
        _, report = run_extraction_qc(entity_type="organizations", entities=entities)
        assert "many_low_quality_names" not in report.flags

    def test_single_low_quality_name_not_flagged(self):
        """A single low-quality name among many good ones should not trigger."""
        entities = [
            {"name": "Department of Defense", "type": "government"},
            {"name": "FBI", "type": "intelligence"},
            {"name": "defense departments", "type": "government"},
        ]
        _, report = run_extraction_qc(entity_type="organizations", entities=entities)
        assert "many_low_quality_names" not in report.flags

    def test_flags_low_quality_location_names(self):
        """Descriptive location phrases should trigger the flag."""
        entities = [
            {"name": "military base in Cuba", "type": "military_base"},
            {"name": "prison near Havana", "type": "detention_facility"},
            {"name": "Guantanamo Bay", "type": "military_base"},
        ]
        _, report = run_extraction_qc(entity_type="locations", entities=entities)
        assert "many_low_quality_names" in report.flags
