import pytest

from src.research.citations import project_citation
from src.research.ingestion import CorpusIngestor
from src.research.locators import TextLocator
from src.research.service import ResearchService


def test_citation_projection_preserves_known_and_labels_missing_fields():
    projected = project_citation(
        {
            "title": "A memoir",
            "creator": "An author",
            "edition": "Second edition",
            "language": "en",
            "translation_responsibility": "Translator Name",
            "stable_identifier": "isbn:123",
        },
        locator={"page": 36},
    )
    assert projected["edition"] == "Second edition"
    assert projected["translation_responsibility"] == "Translator Name"
    assert projected["locator"] == {"page": 36}
    assert projected["access_detail"] == "[unknown]"


def test_imported_source_projects_bibliography_without_inventing_fields(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source words")
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus: {{id: corpus, title: Corpus, version: '1'}}
sources:
  - import_key: memoir
    title: Memoir
    creator: Author
    source_type: book
    status: included
    expected_source_form: born_digital_text
    language: ps
    edition: 2nd
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    service = ResearchService.open(tmp_path / "research.sqlite")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    representation = service.list_representations("corpus", "memoir")[0]
    locator = TextLocator(representation_id=representation.id, start=0, end=6)
    citation = service.project_source_citation("corpus", "memoir", locator=locator)
    assert citation["title"] == "Memoir"
    assert citation["creator"] == "Author"
    assert citation["edition"] == "2nd"
    assert citation["translation_responsibility"] == "[unknown]"
    assert citation["locator"] == locator.model_dump()
    assert citation["quotation"] == "source"

    with pytest.raises(ValueError, match="does not belong"):
        service.project_source_citation("corpus", "another-source", locator=locator)
    with pytest.raises(KeyError):
        service.project_source_citation(
            "corpus",
            "memoir",
            locator=TextLocator(
                representation_id="representation_missing", start=0, end=1
            ),
        )


def test_manifest_nested_metadata_is_merged_with_top_level_fields(tmp_path):
    import hashlib

    source = tmp_path / "source.txt"
    source.write_text("text")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus:
  id: corpus
  title: Corpus
  version: '1'
  metadata: {{custodian: Archive A}}
  archival_series: Series 4
sources:
  - import_key: item
    title: Item
    source_type: archive
    status: included
    expected_source_form: born_digital_text
    metadata: {{intended_audience: internal}}
    institutional_setting: tribunal
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    service = ResearchService.open(tmp_path / "research.sqlite")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    metadata = service.get_source_metadata("corpus", "item")
    assert metadata["intended_audience"] == "internal"
    assert metadata["institutional_setting"] == "tribunal"
    assert service.get_corpus_metadata("corpus")["metadata"] == {
        "custodian": "Archive A",
        "archival_series": "Series 4",
    }
