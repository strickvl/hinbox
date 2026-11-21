"""SQLite lifecycle and transaction boundary for research records."""

import fcntl
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.research.schema import Migration, migrations


class UnsupportedSchemaVersion(RuntimeError):
    """Raised before writes when a database was created by newer code."""


class InvalidMigrationHistory(RuntimeError):
    """Raised when applied migrations are not an exact packaged prefix."""


class ResearchStore:
    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        write_retries: int = 3,
    ) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self.write_retries = write_retries

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            for attempt in range(self.write_retries + 1):
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as error:
                    if (
                        "locked" not in str(error).lower()
                        or attempt == self.write_retries
                    ):
                        raise
                    time.sleep(0.05 * (attempt + 1))
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._migration_lock():
            self._initialize_locked()

    @contextmanager
    def _migration_lock(self) -> Iterator[None]:
        """Serialize schema inspection and migration across local processes."""
        lock_path = self.path.with_name(f"{self.path.name}.migrate.lock")
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _initialize_locked(self) -> None:
        available = migrations()
        self._validate_migration_set(available)
        with self.connection() as connection:
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            applied = []
            if table:
                applied = connection.execute(
                    "SELECT version, name FROM schema_migrations ORDER BY version"
                ).fetchall()
            current = applied[-1]["version"] if applied else 0
            latest = available[-1].version if available else 0
            if current > latest:
                raise UnsupportedSchemaVersion(
                    f"database schema {current} is newer than supported schema {latest}"
                )
            expected_prefix = [
                (migration.version, migration.name)
                for migration in available[: len(applied)]
            ]
            actual_history = [(row["version"], row["name"]) for row in applied]
            if actual_history != expected_prefix:
                raise InvalidMigrationHistory(
                    "applied migration history is not an exact prefix: "
                    f"expected {expected_prefix}, got {actual_history}"
                )

        for migration in available:
            if migration.version <= current:
                continue
            self._apply_migration(migration.version, migration.name, migration.path)

        with self.connection() as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise sqlite3.DatabaseError("research database integrity check failed")
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise sqlite3.IntegrityError(
                    "research database has foreign-key violations"
                )

    @staticmethod
    def _validate_migration_set(available: list[Migration]) -> None:
        versions = [migration.version for migration in available]
        expected = list(range(1, len(available) + 1))
        if versions != expected:
            raise InvalidMigrationHistory(
                f"packaged migrations must be unique and contiguous: expected {expected}, got {versions}"
            )

    def _apply_migration(self, version: int, name: str, path: Path) -> None:
        sql = path.read_text()
        with self.connection() as connection:
            try:
                already_applied = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
                ).fetchone()
                if (
                    already_applied
                    and connection.execute(
                        "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
                    ).fetchone()
                ):
                    connection.rollback()
                    return
                quoted_name = name.replace("'", "''")
                # executescript commits a pending transaction first, so the
                # transaction must be part of the script to keep DDL atomic.
                connection.executescript(
                    "BEGIN EXCLUSIVE;\n"
                    f"{sql}\n"
                    "INSERT INTO schema_migrations(version, name, applied_at) "
                    f"VALUES ({version}, '{quoted_name}', CURRENT_TIMESTAMP);\n"
                    "COMMIT;"
                )
            except Exception:
                connection.rollback()
                raise
