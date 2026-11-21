import sqlite3
import threading

import pytest

from src.research.models import AnchorCreate, AnchorKind
from src.research.schema import Migration, migrations
from src.research.service import ResearchService
from src.research.store import (
    InvalidMigrationHistory,
    ResearchStore,
    UnsupportedSchemaVersion,
)


def test_fresh_database_migrates_once_and_reopens(tmp_path) -> None:
    path = tmp_path / "research.sqlite3"
    ResearchStore(path).initialize()
    ResearchStore(path).initialize()

    with ResearchStore(path).connection() as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row["version"] for row in versions] == [
            migration.version for migration in migrations()
        ]
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_service_keeps_same_name_anchors_distinct(tmp_path) -> None:
    service = ResearchService.open(tmp_path / "research.sqlite3")
    first = service.create_anchor(
        AnchorCreate(
            kind=AnchorKind.PERSON,
            display_name="Muhammad",
            aliases=["Muhammad A"],
        )
    )
    second = service.create_anchor(
        AnchorCreate(
            kind=AnchorKind.PERSON,
            display_name="Muhammad",
            aliases=["Muhammad B"],
        )
    )

    assert first.id != second.id
    assert service.get_anchor(first.id).aliases == ["Muhammad A"]
    assert service.get_anchor(second.id).aliases == ["Muhammad B"]


def test_supersession_preserves_history_and_selects_one_current_version(
    tmp_path,
) -> None:
    service = ResearchService.open(tmp_path / "research.sqlite3")
    original = service.create_record_version(
        kind="historian_assessment",
        payload={"text": "provisional reading"},
        actor_id="historian",
    )
    revised = service.supersede_record_version(
        original.id,
        payload={"text": "corrected reading"},
        actor_id="historian",
    )

    history = service.get_record_history(original.logical_record_id)
    assert [item.id for item in history] == [original.id, revised.id]
    assert history[0].is_current is False
    assert history[1].is_current is True
    assert service.get_current_record(original.logical_record_id).id == revised.id


def test_failed_multi_record_write_rolls_back(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()

    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO audit_actors(id, actor_type, display_name) VALUES (?, ?, ?)",
                ("actor_a", "human", "A"),
            )
            connection.execute(
                "INSERT INTO anchor_aliases(id, anchor_id, alias) VALUES (?, ?, ?)",
                ("alias_bad", "missing_anchor", "Bad"),
            )

    with store.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM audit_actors WHERE id = 'actor_a'"
            ).fetchone()[0]
            == 0
        )


def test_database_rejects_self_supersession_and_history_deletion(tmp_path) -> None:
    service = ResearchService.open(tmp_path / "research.sqlite3")
    record = service.create_record_version(
        kind="claim", payload={"text": "x"}, actor_id="historian"
    )
    store = service.store

    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE record_versions SET supersedes_id = id WHERE id = ?",
                (record.id,),
            )

    revised = service.supersede_record_version(
        record.id, payload={"text": "y"}, actor_id="historian"
    )
    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction() as connection:
            connection.execute("DELETE FROM record_versions WHERE id = ?", (record.id,))
    assert service.get_current_record(record.logical_record_id).id == revised.id


def test_newer_schema_is_rejected_without_writes(tmp_path) -> None:
    path = tmp_path / "research.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)"
    )
    connection.execute(
        "INSERT INTO schema_migrations VALUES (999, 'future', '2026-01-01')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(UnsupportedSchemaVersion):
        ResearchStore(path).initialize()

    connection = sqlite3.connect(path)
    assert (
        connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
    )
    connection.close()


def test_failed_migration_rolls_back_ddl(tmp_path, monkeypatch) -> None:
    migration_path = tmp_path / "001_broken.sql"
    migration_path.write_text(
        "CREATE TABLE should_rollback(id TEXT PRIMARY KEY);\nTHIS IS NOT SQL;\n"
    )
    monkeypatch.setattr(
        "src.research.store.migrations",
        lambda: [Migration(1, "broken", migration_path)],
    )
    path = tmp_path / "research.sqlite3"

    with pytest.raises(sqlite3.OperationalError):
        ResearchStore(path).initialize()

    connection = sqlite3.connect(path)
    assert (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'should_rollback'"
        ).fetchone()
        is None
    )
    connection.close()


def test_database_rejects_duplicate_current_version_and_cycle(tmp_path) -> None:
    service = ResearchService.open(tmp_path / "research.sqlite3")
    first = service.create_record_version(
        kind="claim", payload={"text": "first"}, actor_id="historian"
    )

    with pytest.raises(sqlite3.IntegrityError):
        with service.store.transaction() as connection:
            connection.execute(
                "INSERT INTO record_versions(id, logical_record_id, kind, payload_json, "
                "actor_id, status, is_current, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "version_duplicate",
                    first.logical_record_id,
                    "claim",
                    "{}",
                    "historian",
                    "accepted",
                    1,
                    "2026-01-01T00:00:00+00:00",
                ),
            )

    second = service.supersede_record_version(
        first.id, payload={"text": "second"}, actor_id="historian"
    )
    third = service.supersede_record_version(
        second.id, payload={"text": "third"}, actor_id="historian"
    )
    with pytest.raises(sqlite3.IntegrityError):
        with service.store.transaction() as connection:
            connection.execute(
                "UPDATE record_versions SET supersedes_id = ? WHERE id = ?",
                (third.id, first.id),
            )


def test_overlapping_connections_allow_reads_and_bound_write_retries(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "research.sqlite3"
    blocker = ResearchStore(path, busy_timeout_ms=5)
    blocker.initialize()
    contender = ResearchStore(path, busy_timeout_ms=5, write_retries=2)
    retry_delays = []
    monkeypatch.setattr(
        "src.research.store.time.sleep", lambda delay: retry_delays.append(delay)
    )

    # This open transaction represents an extraction coordinator write. A
    # separate request connection must still read the last committed state,
    # while a review write retries only the configured number of times.
    with blocker.transaction() as extraction_write:
        extraction_write.execute(
            "INSERT INTO audit_actors(id, actor_type, display_name) VALUES (?, ?, ?)",
            ("extractor", "model", "Extraction agent"),
        )

        with contender.connection() as reader:
            assert (
                reader.execute(
                    "SELECT COUNT(*) FROM audit_actors WHERE id = 'extractor'"
                ).fetchone()[0]
                == 0
            )

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with contender.transaction() as review_write:
                review_write.execute(
                    "INSERT INTO audit_actors(id, actor_type, display_name) "
                    "VALUES ('reviewer', 'human', 'Reviewer')"
                )

        assert retry_delays == [0.05, 0.1]

    with contender.transaction() as review_write:
        review_write.execute(
            "INSERT INTO audit_actors(id, actor_type, display_name) "
            "VALUES ('reviewer', 'human', 'Reviewer')"
        )

    with contender.connection() as reader:
        actors = reader.execute(
            "SELECT id FROM audit_actors WHERE id IN ('extractor', 'reviewer') "
            "ORDER BY id"
        ).fetchall()
        assert [row["id"] for row in actors] == ["extractor", "reviewer"]


def test_record_content_and_provenance_are_immutable(tmp_path) -> None:
    service = ResearchService.open(tmp_path / "research.sqlite3")
    record = service.create_record_version(
        kind="claim", payload={"text": "original"}, actor_id="historian"
    )

    immutable_updates = {
        "payload_json": '{"text":"mutated"}',
        "kind": "historian_assessment",
        "actor_id": "historian",
        "created_at": "2099-01-01T00:00:00+00:00",
    }
    for field, value in immutable_updates.items():
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with service.store.transaction() as connection:
                connection.execute(
                    f"UPDATE record_versions SET {field} = ? WHERE id = ?",
                    (value, record.id),
                )

    assert service.get_current_record(record.logical_record_id).payload == {
        "text": "original"
    }


def test_incoherent_current_status_is_rejected(tmp_path) -> None:
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()

    for status in ("provisional", "superseded", "rejected"):
        with pytest.raises(sqlite3.IntegrityError):
            with store.transaction() as connection:
                connection.execute(
                    "INSERT INTO record_versions(id, logical_record_id, kind, "
                    "payload_json, actor_id, status, is_current, created_at) "
                    "VALUES (?, ?, 'claim', '{}', 'historian', ?, 1, CURRENT_TIMESTAMP)",
                    (f"version_{status}", f"record_{status}", status),
                )


def test_concurrent_fresh_initialization_applies_migration_once(tmp_path) -> None:
    path = tmp_path / "research.sqlite3"
    start = threading.Barrier(3)
    errors = []

    def initialize() -> None:
        start.wait()
        try:
            ResearchStore(path).initialize()
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=initialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    with ResearchStore(path).connection() as connection:
        rows = connection.execute(
            "SELECT version, name FROM schema_migrations"
        ).fetchall()
        assert [(row["version"], row["name"]) for row in rows] == [
            (migration.version, migration.name) for migration in migrations()
        ]


def test_packaged_migration_versions_must_be_unique_and_contiguous(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "001_first.sql"
    duplicate = tmp_path / "001_duplicate.sql"
    first.write_text("SELECT 1;")
    duplicate.write_text("SELECT 1;")
    monkeypatch.setattr(
        "src.research.store.migrations",
        lambda: [Migration(1, "first", first), Migration(1, "duplicate", duplicate)],
    )

    with pytest.raises(InvalidMigrationHistory, match="unique and contiguous"):
        ResearchStore(tmp_path / "research.sqlite3").initialize()


def test_applied_migration_name_must_match_packaged_history(tmp_path) -> None:
    path = tmp_path / "research.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)"
    )
    connection.execute(
        "INSERT INTO schema_migrations VALUES (1, 'not_core', '2026-01-01')"
    )
    connection.commit()
    connection.close()

    with pytest.raises(InvalidMigrationHistory, match="exact prefix"):
        ResearchStore(path).initialize()


def test_applied_migration_gap_is_rejected(tmp_path, monkeypatch) -> None:
    migration_files = []
    for version in range(1, 4):
        path = tmp_path / f"00{version}_migration.sql"
        path.write_text("SELECT 1;")
        migration_files.append(Migration(version, f"migration_{version}", path))
    monkeypatch.setattr("src.research.store.migrations", lambda: migration_files)
    database = tmp_path / "research.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)"
    )
    connection.executemany(
        "INSERT INTO schema_migrations VALUES (?, ?, '2026-01-01')",
        [(1, "migration_1"), (3, "migration_3")],
    )
    connection.commit()
    connection.close()

    with pytest.raises(InvalidMigrationHistory, match="exact prefix"):
        ResearchStore(database).initialize()


def test_populated_schema_state_survives_reinitialization(tmp_path) -> None:
    path = tmp_path / "research.sqlite3"
    service = ResearchService.open(path)
    anchor = service.create_anchor(
        AnchorCreate(
            kind=AnchorKind.PERSON,
            display_name="Distinct person",
            aliases=["Recorded alias"],
        )
    )
    first = service.create_record_version(
        kind="claim", payload={"text": "first"}, actor_id="historian"
    )
    second = service.supersede_record_version(
        first.id, payload={"text": "second"}, actor_id="historian"
    )

    reopened = ResearchService.open(path)
    assert reopened.get_anchor(anchor.id).aliases == ["Recorded alias"]
    history = reopened.get_record_history(first.logical_record_id)
    assert [version.id for version in history] == [first.id, second.id]
    assert [version.status for version in history] == ["superseded", "accepted"]
