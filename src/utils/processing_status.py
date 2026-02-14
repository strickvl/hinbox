"""Sidecar processing status tracker.

Stores article processing status in a separate JSON file instead of rewriting
the articles parquet. The articles file stays read-only; this module handles
the "has this article been processed?" bookkeeping.
"""

import json
import os
from typing import Any, Dict, Optional, Set

from src.logging_config import get_logger

logger = get_logger("utils.processing_status")


class ProcessingStatus:
    """Manages a sidecar JSON file tracking which articles have been processed."""

    def __init__(self, base_dir: str):
        self.status_path = os.path.join(base_dir, "processing_status.json")
        self._status: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing status from disk."""
        if os.path.exists(self.status_path):
            try:
                with open(self.status_path) as f:
                    self._status = json.load(f)
                logger.info(
                    f"Loaded processing status for {len(self._status)} articles"
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not read processing status, starting fresh: {e}")
                self._status = {}

    def is_processed(self, article_id: str) -> bool:
        """Check if an article has been processed."""
        entry = self._status.get(article_id, {})
        return entry.get("processed", False)

    def get_metadata(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get processing metadata for an article, or None if not tracked."""
        return self._status.get(article_id)

    def mark_processed(self, article_id: str, metadata: Dict[str, Any]) -> None:
        """Mark an article as processed and store its metadata."""
        self._status[article_id] = metadata

    def mark_skipped(self, article_id: str, reason: str) -> None:
        """Mark an article as skipped (not relevant, no content, etc.)."""
        self._status[article_id] = {"processed": False, "reason": reason}

    def flush(self) -> None:
        """Write current status to disk atomically."""
        tmp_path = self.status_path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self._status, f, indent=2, default=str)
            os.replace(tmp_path, self.status_path)
            logger.info(f"Flushed processing status for {len(self._status)} articles")
        except OSError as e:
            logger.error(f"Failed to write processing status: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of the status dict for safe read-only use in workers.

        Unlike ``processed_ids()`` which only returns IDs, this includes
        the full metadata (e.g. ``content_hash``) needed for skip-if-unchanged.
        """
        return dict(self._status)

    def processed_ids(self) -> Set[str]:
        """Return a snapshot of article IDs currently marked as processed.

        Thread-safe when called before concurrent work begins (the returned
        set is an independent copy).
        """
        return {aid for aid, meta in self._status.items() if meta.get("processed")}

    @property
    def total_processed(self) -> int:
        """Count of articles marked as processed."""
        return sum(1 for v in self._status.values() if v.get("processed"))

    @property
    def total_tracked(self) -> int:
        """Count of all tracked articles (processed + skipped)."""
        return len(self._status)
