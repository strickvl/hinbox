"""Contract tests for the rights-safe Release 0 research corpus."""

import hashlib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "research" / "release_zero"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.yaml"
EVALUATION_PATH = REPO_ROOT / "docs" / "evaluations" / "zaeef-release-0.md"


def _load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
        return yaml.safe_load(manifest_file)


def test_manifest_is_frozen_rights_safe_and_checksum_verified():
    manifest = _load_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["corpus"]["evaluation_scope"] == "infrastructure_fidelity"
    assert manifest["corpus"]["frozen_order"]

    entries = manifest["sources"]
    by_key = {entry["import_key"]: entry for entry in entries}
    assert len(by_key) == len(entries)
    assert manifest["corpus"]["frozen_order"] == [
        entry["import_key"] for entry in entries if entry["status"] == "included"
    ]

    allowed_statuses = {"included", "excluded", "unavailable", "unprocessed"}
    allowed_forms = {"born_digital_text", "image_backed_page", "local_corpus_slot"}
    for entry in entries:
        assert entry["status"] in allowed_statuses
        assert entry["expected_source_form"] in allowed_forms
        assert entry["rights"]["redistribution"]
        assert entry["rights"]["training_eligibility"]

        for representation in entry.get("representations", []):
            fixture_path = FIXTURE_ROOT / representation["path"]
            assert fixture_path.is_file()
            digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
            assert representation["sha256"] == digest


def test_fixture_covers_conflict_qualifier_and_text_correction_cases():
    manifest = _load_manifest()
    phenomena = {
        phenomenon
        for entry in manifest["sources"]
        for phenomenon in entry.get("evaluation_phenomena", [])
    }

    assert {
        "conflicting_date",
        "material_qualifier",
        "ocr_correction",
        "omission",
    } <= phenomena
    included_forms = {
        entry["expected_source_form"]
        for entry in manifest["sources"]
        if entry["status"] == "included"
    }
    assert {"born_digital_text", "image_backed_page"} <= included_forms


def test_evaluation_freezes_three_baselines_and_losable_thresholds():
    document = EVALUATION_PATH.read_text(encoding="utf-8")
    normalized_document = " ".join(document.split())

    for heading in (
        "## Corpus boundary and evidential limits",
        "## Frozen comparison protocol",
        "### A. Raw source folder",
        "### B. Current or prompt-improved prose pipeline",
        "### C. Minimal evidence ledger",
        "## Material-advantage decision rule",
        "## Review-workload ceiling",
        "## Reproducibility record",
    ):
        assert heading in document

    assert "workflow-redesign" in document
    assert "evidence-model-redesign" in document
    assert "does not evaluate narrative synthesis" in normalized_document
