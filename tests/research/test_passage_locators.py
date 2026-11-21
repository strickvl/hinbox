import hashlib
from pathlib import Path

import pytest

from src.research.ingestion import CorpusIngestor
from src.research.locators import TextLocator
from src.research.service import ResearchService


def test_quote_round_trips_from_checksum_verified_bytes(tmp_path: Path):
    text = "alpha beta gamma"
    source = tmp_path / "source.txt"
    source.write_text(text)
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
    language: en
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    service = ResearchService.open(tmp_path / "research.sqlite")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    representation = service.list_representations("corpus", "item")[0]

    locator = TextLocator(representation_id=representation.id, start=6, end=10)
    assert service.retrieve_passage(locator).quotation == "beta"
    assert service.retrieve_passage(locator).textual_form == "original_text"

    with pytest.raises(ValueError, match="bounds"):
        service.retrieve_quote(representation.id, 1, 100)

    Path(representation.managed_path).write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        service.retrieve_quote(representation.id, 0, 5)


def test_locator_offsets_are_unicode_codepoints_not_bytes(tmp_path: Path):
    text = "Pāštō passage"
    source = tmp_path / "source.txt"
    source.write_text(text)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '1'}}
sources:
  - import_key: unicode
    title: Unicode
    source_type: text
    status: included
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    service = ResearchService.open(tmp_path / "research.sqlite")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    representation = service.list_representations("corpus", "unicode")[0]
    locator = TextLocator(representation_id=representation.id, start=0, end=5)
    assert service.retrieve_passage(locator).quotation == "Pāštō"
