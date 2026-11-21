"""Minimal, rerunnable born-digital corpus ingestion."""

import hashlib
import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from src.research.manifest import ManifestRepresentation, load_manifest
from src.research.service import ResearchService

TEXT_KINDS = {"original_text", "published_translation", "researcher_translation"}


class IngestionResult(BaseModel):
    imported_sources: int = 0
    imported_representations: int = 0


class ReconciliationResult(BaseModel):
    removed_stages: int = 0
    removed_orphan_blobs: int = 0


class CorpusIngestor:
    def __init__(self, service: ResearchService, managed_root: str | Path) -> None:
        self.service = service
        self.managed_root = Path(managed_root)

    def reconcile_staging(self) -> ReconciliationResult:
        staging = self.managed_root / ".staging"
        removed = 0
        if staging.exists():
            for path in staging.iterdir():
                if path.is_file():
                    path.unlink()
                    removed += 1
        referenced = self.service.referenced_blob_checksums()
        removed_orphans = 0
        blobs = self.managed_root / "blobs"
        if blobs.exists():
            for path in blobs.glob("*/*"):
                if (
                    path.is_file()
                    and len(path.name) == 64
                    and all(character in "0123456789abcdef" for character in path.name)
                    and path.name not in referenced
                ):
                    path.unlink()
                    removed_orphans += 1
        return ReconciliationResult(
            removed_stages=removed, removed_orphan_blobs=removed_orphans
        )

    def ingest_manifest(self, manifest_path: str | Path) -> IngestionResult:
        self.reconcile_staging()
        manifest = load_manifest(manifest_path)
        prepared = []
        for source in manifest.sources:
            if source.expected_source_form not in (None, "born_digital_text"):
                continue
            for representation in source.representations:
                if representation.kind in TEXT_KINDS:
                    raw, text = self._preflight_representation(
                        manifest.path.parent, representation
                    )
                    prepared.append((source.import_key, representation, raw, text))
        source_payloads = [
            source.model_dump(mode="json") for source in manifest.sources
        ]
        snapshot = manifest.model_dump(mode="json", exclude={"path"})
        manifest_checksum = hashlib.sha256(manifest.path.read_bytes()).hexdigest()
        source_ids = self.service.register_manifest_records(
            manifest.corpus.model_dump(mode="json"),
            manifest.path,
            manifest_checksum,
            snapshot,
            source_payloads,
        )
        imported = 0
        for import_key, representation, raw, text in prepared:
            imported += self._persist_representation(
                source_ids[import_key], representation, raw, text
            )
        return IngestionResult(
            imported_sources=len(source_ids), imported_representations=imported
        )

    @staticmethod
    def _preflight_representation(
        manifest_root: Path, representation: ManifestRepresentation
    ) -> tuple[bytes, str]:
        source_path = (manifest_root / representation.path).resolve()
        try:
            source_path.relative_to(manifest_root.resolve())
        except ValueError as error:
            raise ValueError(
                "representation path escapes manifest directory"
            ) from error
        raw = source_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"born-digital representation {representation.path} is not UTF-8"
            ) from error
        actual_checksum = hashlib.sha256(raw).hexdigest()
        if actual_checksum != representation.sha256:
            raise ValueError(
                f"checksum mismatch for {representation.path}: "
                f"expected {representation.sha256}, got {actual_checksum}"
            )
        return raw, text

    def _persist_representation(
        self,
        source_id: str,
        representation: ManifestRepresentation,
        raw: bytes,
        text: str,
    ) -> int:
        actual_checksum = hashlib.sha256(raw).hexdigest()
        if self.service.has_representation(
            source_id, representation.kind, actual_checksum
        ):
            return 0

        staging = self.managed_root / ".staging"
        blob_path = self.managed_root / "blobs" / actual_checksum[:2] / actual_checksum
        staging.mkdir(parents=True, exist_ok=True)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        stage_path = staging / f"{actual_checksum}.{uuid4().hex}.stage"
        with stage_path.open("wb") as stage_file:
            stage_file.write(raw)
            stage_file.flush()
            os.fsync(stage_file.fileno())
        if hashlib.sha256(stage_path.read_bytes()).hexdigest() != actual_checksum:
            stage_path.unlink(missing_ok=True)
            raise ValueError("staged representation checksum mismatch")
        if blob_path.exists():
            stage_path.unlink()
        else:
            os.replace(stage_path, blob_path)
            directory_fd = os.open(blob_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

        self.service.register_representation(
            source_id=source_id,
            kind=representation.kind,
            checksum=actual_checksum,
            byte_length=len(raw),
            character_length=len(text),
            managed_path=blob_path,
            review_state=representation.review_state,
            metadata=representation.model_dump(mode="json"),
        )
        return 1
