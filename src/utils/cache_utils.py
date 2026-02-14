"""Shared caching utilities: thread-safe LRU cache and stable hashing helpers."""

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache backed by an OrderedDict.

    Evicts the least-recently-used entry when ``max_items`` is reached.
    All public methods are serialized with a lock so the cache is safe
    to use from concurrent threads (e.g. extraction workers).
    """

    def __init__(self, max_items: int):
        self._max = max(1, max_items)
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: K) -> Optional[V]:
        """Return cached value or ``None`` on miss. Promotes key on hit."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return self._data[key]
            self._misses += 1
            return None

    def set(self, key: K, value: V) -> None:
        """Insert or update *key*. Evicts LRU entry if over capacity."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                self._data[key] = value
                if len(self._data) > self._max:
                    self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def stats(self) -> dict:
        """Return hit/miss counters for diagnostics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
                "size": len(self._data),
            }


# ---------------------------------------------------------------------------
# Stable hashing helpers
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_jsonable(obj: object) -> str:
    """Return the hex SHA-256 of a JSON-serializable object.

    Keys are sorted for determinism. Non-serializable objects fall back
    to ``str(obj)`` so this never raises.
    """
    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = str(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
