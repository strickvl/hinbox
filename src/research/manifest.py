"""Validated, inspectable corpus manifest parsing."""

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, model_validator


class ManifestRepresentation(BaseModel):
    kind: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_state: str = "unreviewed"
    derived_from: str | None = None
    page: int | None = None


class ManifestSource(BaseModel):
    import_key: str
    title: str
    creator: str | None = None
    source_type: str
    status: str
    expected_source_form: str | None = None
    language: str | None = None
    representations: List[ManifestRepresentation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def retain_bibliography(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = set(cls.model_fields)
            copied = dict(data)
            explicit = data.get("metadata") or {}
            copied["metadata"] = {
                **explicit,
                **{key: value for key, value in data.items() if key not in known},
            }
            return copied
        return data


class CorpusMetadata(BaseModel):
    id: str
    title: str
    version: str
    frozen_at: str | None = None
    evaluation_scope: str | None = None
    scope_note: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def retain_corpus_boundary(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = set(cls.model_fields)
            copied = dict(data)
            explicit = data.get("metadata") or {}
            copied["metadata"] = {
                **explicit,
                **{key: value for key, value in data.items() if key not in known},
            }
            return copied
        return data


class CorpusManifest(BaseModel):
    schema_version: int
    corpus: CorpusMetadata
    sources: List[ManifestSource]
    path: Path


def load_manifest(path: str | Path) -> CorpusManifest:
    manifest_path = Path(path).resolve()
    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("manifest must contain a mapping")
    raw["path"] = manifest_path
    return CorpusManifest.model_validate(raw)
