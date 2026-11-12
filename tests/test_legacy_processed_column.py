"""An old in-file ``processing_metadata`` column must not mark an article done.

Only the processing-status sidecar decides whether an article was processed.
In the 12 Nov full run, 13 articles whose column still said ``processed: True``
from an old run were skipped and never extracted.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.engine import ArticleProcessor
from src.process_and_extract import extract_single_article_only
from src.utils.cache_utils import sha256_text

CONTENT = "Detainee lawyers filed a motion at the base on Tuesday."
LEGACY = {"processed": True, "extracted_counts": {"people": 3}}


def _run(row, snapshot):
    processor = ArticleProcessor(domain="guantanamo", model_type="gemini")
    args = SimpleNamespace(force_reprocess=False, relevance_check=False, verbose=False)
    extract = MagicMock(return_value={})
    with patch.object(processor, "extract_all_entities", extract):
        result = extract_single_article_only(
            row, 1, processor, args, status_snapshot=snapshot, skip_if_unchanged=True
        )
    return result, extract


def test_legacy_column_saying_processed_does_not_skip_the_article():
    row = {"id": "a1", "title": "t", "content": CONTENT, "processing_metadata": LEGACY}
    result, extract = _run(row, {})
    assert result.skip_reason != "already_processed"
    assert extract.called
    assert "extracted_counts" not in result.processing_metadata


def test_sidecar_entry_still_skips_a_processed_article():
    row = {"id": "a1", "title": "t", "content": CONTENT}
    done = {"a1": {"processed": True, "content_hash": sha256_text(CONTENT)}}
    result, extract = _run(row, done)
    assert result.skip_reason == "already_processed"
    assert not extract.called
