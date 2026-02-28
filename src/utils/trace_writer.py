"""Append-only JSONL writer for pipeline trace events."""

import os
import threading
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

from src.utils.cache_utils import sha256_text
from src.utils.trace_schema import TraceEvent


class TraceWriter:
    """Run-scoped append-only JSONL trace writer.

    Uses one atomic os.write call per line and a short critical section guarded
    by a lock to avoid line interleaving across extraction worker threads.
    """

    def __init__(
        self,
        *,
        output_dir: str,
        run_id: Optional[str] = None,
        enabled: bool = True,
        traces_subdir: str = "traces",
        runs_subdir: str = "runs",
    ) -> None:
        self.enabled = bool(enabled)
        self.run_id = run_id or self._make_run_id()
        self._count = 0
        self._lock = threading.Lock()
        self._fd: Optional[int] = None

        self.traces_dir = os.path.join(output_dir, traces_subdir, runs_subdir)
        self.filepath = os.path.join(self.traces_dir, f"{self.run_id}.jsonl")

        if not self.enabled:
            return

        os.makedirs(self.traces_dir, exist_ok=True)
        self._fd = os.open(
            self.filepath,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )

    @staticmethod
    def _make_run_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
        entropy = f"{ts}|{os.getpid()}|{time.time_ns()}"
        short_hash = sha256_text(entropy)[:6]
        return f"{ts}-{short_hash}"

    def emit(self, event: TraceEvent) -> None:
        """Append one trace event to the run JSONL file."""
        if not self.enabled or self._fd is None:
            return

        if not event.run_id:
            event.run_id = self.run_id

        line = event.model_dump_json(exclude_none=True) + "\n"
        payload = line.encode("utf-8")
        with self._lock:
            os.write(self._fd, payload)
            self._count += 1

    def emit_batch(self, events: Iterable[TraceEvent]) -> None:
        """Append multiple events in order."""
        for event in events:
            self.emit(event)

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._count

    def close(self) -> None:
        """Close the underlying file descriptor (idempotent)."""
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
