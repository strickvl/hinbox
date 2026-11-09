"""Tests that an article is only marked done when nothing failed on the way.

These tests verify:
- A failed extraction, a crashed type merge, or a failed profile, match or
  dispute call leaves
  the article unprocessed with reason "incomplete", so the next run retries it
- merge_all_entities records crashed merges and failed profile calls
  (the merger's own failure tests live in test_entity_merger_merge_smoke.py)
- The entity tables are written before the status file
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.engine import ArticleProcessor, EntityMerger
from src.engine.mergers import MergeStats
from src.process_and_extract import (
    ArticleExtractionResult,
    merge_all_entities,
    merge_and_finalize,
    save_entities_then_status,
)
from src.utils.processing_status import ProcessingStatus
from tests.test_entity_merger_merge_smoke import make_empty_entities

TYPES = ["people", "organizations", "locations", "events"]
ARTICLE = {
    "id": "art-001",
    "title": "T",
    "url": "u",
    "published_date": "",
    "content": "c",
}


def _result(failed_type=None, **meta):
    extraction = {et: {"success": et != failed_type} for et in TYPES}
    return ArticleExtractionResult(
        row_index=1,
        row={},
        article_info=ARTICLE,
        processing_metadata={"phase_outcomes": {"extraction": extraction}, **meta},
        extraction_timestamp="2025-01-01T00:00:00",
        extracted_entities={et: [] for et in TYPES},
        was_extracted=True,
    )


def _finalize(result, tracker):
    counters = {"processed_count": 0, "skipped_incomplete": 0}
    with patch("src.process_and_extract.merge_all_entities"):
        merge_and_finalize(
            result,
            entities=make_empty_entities(),
            processor=ArticleProcessor(domain="guantanamo", model_type="ollama"),
            args=MagicMock(verbose=False),
            domain_config=MagicMock(),
            status_tracker=tracker,
            counters=counters,
        )
    return counters


class TestIncompleteArticlesAreRetried:
    @pytest.mark.parametrize(
        "result, problem",
        [
            (_result(failed_type="people"), "extract_people"),
            (_result(error_events="embedding down"), "error_events"),
            (_result(failed_profiles_people=1), "failed_profiles_people"),
            (_result(failed_checks_events=2), "failed_checks_events"),
        ],
    )
    def test_failure_leaves_article_unprocessed(self, tmp_path, result, problem):
        tracker = ProcessingStatus(str(tmp_path))
        counters = _finalize(result, tracker)

        entry = tracker.get_metadata("art-001") or {}
        assert entry["processed"] is False
        assert entry["reason"] == "incomplete"
        assert entry["problems"] == [problem]
        assert counters == {"processed_count": 0, "skipped_incomplete": 1}

    def test_clean_article_is_marked_done(self, tmp_path):
        tracker = ProcessingStatus(str(tmp_path))
        counters = _finalize(_result(), tracker)

        assert tracker.is_processed("art-001")
        assert counters["processed_count"] == 1


class TestMergeAllEntitiesRecordsFailures:
    def test_crashed_merge_and_failed_profiles_are_recorded(self):
        def fake_merge(self, extracted, *args, **kwargs):
            if self.entity_type == "events":
                raise RuntimeError("embedding down")
            if self.entity_type == "locations":
                return MergeStats(failed_checks=1)
            return MergeStats(failed_profiles=2 if self.entity_type == "people" else 0)

        processor = MagicMock(domain="guantanamo")
        processor.convert_pydantic_to_dict.side_effect = lambda items: items
        config = MagicMock()
        config.get_concurrency_config.return_value = {"merge_parallel_types": 4}
        metadata = {}
        with patch.object(EntityMerger, "merge_entities", fake_merge):
            merge_all_entities(
                {et: [] for et in TYPES},
                make_empty_entities(),
                ARTICLE,
                "2025-01-01T00:00:00",
                "ollama",
                metadata,
                processor,
                domain_config=config,
            )

        assert metadata == {
            "error_events": "embedding down",
            "failed_profiles_people": 2,
            "failed_checks_locations": 1,
        }


class TestSaveOrder:
    def test_failed_entity_write_leaves_status_file_unwritten(self, tmp_path):
        tracker = ProcessingStatus(str(tmp_path))
        tracker.mark_processed("art-001", {"processed": True})
        with (
            patch(
                "src.process_and_extract.write_entities_to_files",
                side_effect=OSError("disk full"),
            ),
            pytest.raises(OSError),
        ):
            save_entities_then_status(make_empty_entities(), str(tmp_path), tracker)

        assert not os.path.exists(tmp_path / "processing_status.json")
