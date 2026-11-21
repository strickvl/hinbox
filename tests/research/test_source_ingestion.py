import hashlib
import os
import stat
from pathlib import Path

import pytest

from src.research.ingestion import CorpusIngestor
from src.research.service import ResearchService

FIXTURE = Path("tests/fixtures/research/release_zero/manifest.yaml")


def open_service(tmp_path: Path) -> ResearchService:
    return ResearchService.open(tmp_path / "research.sqlite")


def test_ingestion_is_idempotent_and_keeps_distinct_source_identity(tmp_path: Path):
    service = open_service(tmp_path)
    ingestor = CorpusIngestor(service, tmp_path / "managed")

    first = ingestor.ingest_manifest(FIXTURE)
    second = ingestor.ingest_manifest(FIXTURE)

    assert first.imported_representations == 2
    assert second.imported_representations == 0
    assert service.research_counts() == {
        "sources": 6,
        "blobs": 2,
        "representations": 2,
        "source_units": 2,
    }

    shared = tmp_path / "shared.txt"
    shared.write_text("identical source bytes")
    digest = hashlib.sha256(shared.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus: {{id: shared, title: Shared, version: '1'}}
sources:
  - import_key: a
    title: A
    source_type: text
    status: included
    representations: [{{kind: original_text, path: shared.txt, sha256: {digest}}}]
  - import_key: b
    title: B
    source_type: text
    status: included
    representations: [{{kind: original_text, path: shared.txt, sha256: {digest}}}]
"""
    )
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    counts = service.research_counts()
    assert counts["sources"] == 8
    assert counts["blobs"] == 3
    assert counts["representations"] == 4


def test_changed_bytes_create_unselected_successor(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("old words")
    manifest = tmp_path / "manifest.yaml"

    def write_manifest() -> None:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest.write_text(
            f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '1'}}
sources:
  - import_key: item
    title: Item
    source_type: text
    status: included
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
        )

    write_manifest()
    service = open_service(tmp_path)
    ingestor = CorpusIngestor(service, tmp_path / "managed")
    ingestor.ingest_manifest(manifest)
    original = service.list_representations("corpus", "item")[0]

    source.write_text("new words")
    write_manifest()
    ingestor.ingest_manifest(manifest)
    representations = service.list_representations("corpus", "item")

    assert len(representations) == 2
    assert [item.selected for item in representations] == [True, False]
    assert service.retrieve_quote(original.id, 0, 3) == "old"


def test_checksum_failure_creates_no_database_or_managed_file(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("bytes")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """schema_version: 1
corpus: {id: corpus, title: Corpus, version: '1'}
sources:
  - import_key: item
    title: Item
    source_type: text
    status: included
    representations: [{kind: original_text, path: source.txt, sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}]
"""
    )
    service = open_service(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    assert service.research_counts()["sources"] == 0
    assert service.research_counts()["representations"] == 0


def test_changed_manifest_metadata_is_not_destructively_overwritten(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"

    def write(title: str, version: str) -> None:
        manifest.write_text(
            f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '{version}'}}
sources:
  - import_key: item
    title: {title}
    source_type: text
    status: included
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
        )

    service = open_service(tmp_path)
    ingestor = CorpusIngestor(service, tmp_path / "managed")
    write("Original title", "1")
    ingestor.ingest_manifest(manifest)
    write("Rewritten title", "2")
    with pytest.raises(ValueError, match="metadata changed"):
        ingestor.ingest_manifest(manifest)
    assert service.get_source_metadata("corpus", "item")["title"] == "Original title"
    assert service.manifest_import_count("corpus") == 1


def test_final_orphan_blob_is_audited_and_removed(tmp_path: Path):
    service = open_service(tmp_path)
    managed = tmp_path / "managed"
    orphan = managed / "blobs" / "aa" / ("a" * 64)
    orphan.parent.mkdir(parents=True)
    orphan.write_text("uncommitted")

    result = CorpusIngestor(service, managed).reconcile_staging()

    assert result.removed_orphan_blobs == 1
    assert not orphan.exists()


def test_new_blob_fsyncs_staged_file_and_parent_directory(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("durable")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '1'}}
sources:
  - import_key: item
    title: Item
    source_type: text
    status: included
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    fsync_targets = []
    original_fsync = os.fsync

    def record_fsync(file_descriptor: int) -> None:
        fsync_targets.append(stat.S_ISDIR(os.fstat(file_descriptor).st_mode))
        original_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    CorpusIngestor(open_service(tmp_path), tmp_path / "managed").ingest_manifest(
        manifest
    )

    assert False in fsync_targets
    assert True in fsync_targets


def test_manifest_versions_are_preserved_as_append_only_snapshots(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"

    def write(version: str) -> None:
        manifest.write_text(
            f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '{version}'}}
sources:
  - import_key: item
    title: Item
    source_type: text
    status: included
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
        )

    service = open_service(tmp_path)
    ingestor = CorpusIngestor(service, tmp_path / "managed")
    write("1")
    ingestor.ingest_manifest(manifest)
    write("2")
    ingestor.ingest_manifest(manifest)

    assert service.manifest_import_count("corpus") == 2


def test_interrupted_stage_is_reconciled(tmp_path: Path):
    service = open_service(tmp_path)
    managed = tmp_path / "managed"
    stage = managed / ".staging" / "orphan.stage"
    stage.parent.mkdir(parents=True)
    stage.write_text("unfinished")

    result = CorpusIngestor(service, managed).reconcile_staging()

    assert result.removed_stages == 1
    assert not stage.exists()


def test_retry_after_blob_rename_and_database_failure_is_safe(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source.txt"
    source.write_text("durable evidence")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '1'}}
sources:
  - import_key: item
    title: Item
    source_type: text
    status: included
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    service = open_service(tmp_path)
    ingestor = CorpusIngestor(service, tmp_path / "managed")
    original_insert = service.register_representation

    def fail_once(**kwargs):
        monkeypatch.setattr(service, "register_representation", original_insert)
        raise RuntimeError("injected pre-commit failure")

    monkeypatch.setattr(service, "register_representation", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        ingestor.ingest_manifest(manifest)
    assert service.research_counts()["representations"] == 0

    ingestor.ingest_manifest(manifest)
    assert service.research_counts()["representations"] == 1
    assert (
        service.retrieve_quote(
            service.list_representations("corpus", "item")[0].id, 0, 7
        )
        == "durable"
    )
