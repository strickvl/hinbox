"""Lossless citation projection with visible unknowns."""

from typing import Any, Dict

FIELDS = (
    "title",
    "creator",
    "publication_date",
    "edition",
    "language",
    "translation_responsibility",
    "stable_identifier",
    "access_detail",
)


def project_citation(
    source_metadata: Dict[str, Any], *, locator: Dict[str, Any]
) -> Dict[str, Any]:
    projected = {field: source_metadata.get(field) or "[unknown]" for field in FIELDS}
    projected["locator"] = locator
    return projected
