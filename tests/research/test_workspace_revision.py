from pathlib import Path

import pytest

from src.research.ingestion import CorpusIngestor
from src.research.service import ResearchService


def test_workspace_revision_changes_only_when_context_changes(tmp_path: Path):
    service = ResearchService.open(tmp_path / "research.sqlite")
    workspace = service.ensure_workspace("zaeef-release-zero", "Zaeef Release 0")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(
        Path("tests/fixtures/research/release_zero/manifest.yaml")
    )
    source_id = service.list_representations(
        "release-zero-rights-safe", "synthetic-account-a"
    )[0].source_id
    first = service.create_workspace_revision(workspace.id)
    again = service.create_workspace_revision(workspace.id)
    assert first.id == again.id

    changed = service.create_workspace_revision(
        workspace.id,
        memberships=[("source", source_id, "included")],
        reviewed_context={"scope_note": "bounded corpus"},
    )
    assert changed.id != first.id
    assert service.get_workspace_revision(changed.id).reviewed_context == {
        "scope_note": "bounded corpus"
    }


def test_workspace_rejects_invalid_missing_and_duplicate_membership(tmp_path: Path):
    service = ResearchService.open(tmp_path / "research.sqlite")
    workspace = service.ensure_workspace("workspace", "Workspace")
    with pytest.raises(ValueError, match="record kind"):
        service.create_workspace_revision(
            workspace.id, memberships=[("claim", "missing", "included")]
        )
    with pytest.raises(KeyError):
        service.create_workspace_revision(
            workspace.id, memberships=[("source", "missing", "included")]
        )
    with pytest.raises(ValueError, match="duplicate"):
        service.create_workspace_revision(
            workspace.id,
            memberships=[
                ("source", "same", "included"),
                ("source", "same", "excluded"),
            ],
        )
