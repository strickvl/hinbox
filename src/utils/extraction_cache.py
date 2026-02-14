"""Persistent sidecar cache for entity extraction results.

Stores extraction outputs as JSON files keyed on all output-affecting inputs
(content hash, model, entity type, prompt hash, schema hash, temperature).
This avoids redundant LLM calls when re-processing unchanged articles.

Cache layout on disk::

    {base_dir}/{subdir}/v{version}/{key[0:2]}/{key[2:4]}/{key}.json

Version-based invalidation: bumping ``cache.extraction.version`` in the
domain config causes reads from the old ``vN/`` directory to stop matching,
effectively invalidating the entire cache without deleting files.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.logging_config import get_logger
from src.utils.cache_utils import sha256_jsonable, sha256_text

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema hashing
# ---------------------------------------------------------------------------


def _schema_hash(response_model: Any) -> str:
    """Compute a stable hash of a Pydantic response model's JSON schema.

    Handles both ``List[Entity]`` (the common case) and bare ``Entity``
    models.  Falls back to ``str(response_model)`` for anything else.
    """
    origin = getattr(response_model, "__origin__", None)
    if origin is list:
        args = getattr(response_model, "__args__", ())
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            item_schema = args[0].model_json_schema()
            return sha256_jsonable({"type": "array", "items": item_schema})
        return sha256_text(str(response_model))
    if isinstance(response_model, type) and issubclass(response_model, BaseModel):
        return sha256_jsonable(response_model.model_json_schema())
    return sha256_text(str(response_model))


# ---------------------------------------------------------------------------
# ExtractionSidecarCache
# ---------------------------------------------------------------------------


class ExtractionSidecarCache:
    """Persistent JSON sidecar cache for extraction results.

    Thread-safe: each write goes through a temp file + ``os.replace``
    (atomic on POSIX).  Concurrent reads of the same key are safe because
    ``os.replace`` is atomic — readers either see the old file or the new one,
    never a partial write.
    """

    def __init__(
        self,
        *,
        base_dir: str,
        subdir: str = "cache/extractions",
        version: int = 1,
        enabled: bool = True,
    ):
        self._enabled = enabled
        self._version = version
        self._root = os.path.join(base_dir, subdir, f"v{version}")
        self._hits = 0
        self._misses = 0

        if self._enabled:
            os.makedirs(self._root, exist_ok=True)

    # ------------------------------------------------------------------
    # Key building
    # ------------------------------------------------------------------

    def make_key(
        self,
        *,
        text: str,
        system_prompt: str,
        response_model: Any,
        model: str,
        entity_type: str,
        temperature: float,
    ) -> str:
        """Build a deterministic hex cache key from all output-affecting inputs."""
        content_hash = sha256_text(text)
        prompt_hash = sha256_text(system_prompt)
        schema_hash = _schema_hash(response_model)

        parts = (
            f"extraction|v{self._version}"
            f"|{entity_type}|{model}"
            f"|temp={temperature}"
            f"|content={content_hash}"
            f"|prompt={prompt_hash}"
            f"|schema={schema_hash}"
        )
        return sha256_text(parts)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def _path_for(self, key: str) -> str:
        """Return the file path for a cache key, sharded by 2-char prefix."""
        return os.path.join(self._root, key[:2], key[2:4], f"{key}.json")

    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the cached record for *key*, or ``None`` on miss."""
        if not self._enabled:
            return None

        path = self._path_for(key)
        try:
            with open(path, "r") as f:
                record = json.load(f)
            self._hits += 1
            return record
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._misses += 1
            return None

    def write(self, key: str, record: Dict[str, Any]) -> None:
        """Atomically write *record* as JSON for *key*."""
        if not self._enabled:
            return

        path = self._path_for(key)
        parent = os.path.dirname(path)
        os.makedirs(parent, exist_ok=True)

        # Write to a temp file in the same directory, then atomic rename.
        fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, separators=(",", ":"), default=str)
            os.replace(tmp, path)
        except Exception:
            # Clean up on failure; ignore errors during cleanup.
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total else 0.0,
            "version": self._version,
            "root": self._root,
        }


# ---------------------------------------------------------------------------
# Record helpers
# ---------------------------------------------------------------------------


def build_cache_record(
    *,
    output: Any,
    entity_type: str,
    model: str,
    temperature: float,
    content_hash: str,
    prompt_hash: str,
    schema_hash: str,
    cache_version: int,
) -> Dict[str, Any]:
    """Build the JSON-serializable record stored in the cache file."""
    # Serialize Pydantic models to dicts if needed
    serialized: List[Dict[str, Any]] = []
    for item in output or []:
        if isinstance(item, dict):
            serialized.append(item)
        elif hasattr(item, "model_dump"):
            serialized.append(item.model_dump())
        elif hasattr(item, "dict"):
            serialized.append(item.dict())
        else:
            serialized.append(item)

    return {
        "cache_version": cache_version,
        "entity_type": entity_type,
        "model": model,
        "temperature": temperature,
        "content_hash": content_hash,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": serialized,
    }
