"""Deterministic quality controls for extraction and profile outputs.

These checks run *after* LLM generation and apply rules that don't need
a model: required-field validation, name normalization, within-article
dedup, and suspicious-result flagging.

QC functions never raise — they only drop invalid items and report.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.constants import (
    PROFILE_QC_MIN_TAG_COUNT,
    PROFILE_QC_MIN_TEXT_LENGTH,
    QC_MIN_NAME_LENGTH,
)
from src.logging_config import get_logger

logger = get_logger("quality_controls")

# Citation pattern: ^[article_id] where article_id is non-empty
CITATION_RE = re.compile(r"\^\[([^\]\s]+)\]")

# ──────────────────────────────────────────────
# Extraction QC
# ──────────────────────────────────────────────

# Fallback required fields — used when Pydantic schema lookup fails
_FALLBACK_REQUIRED_FIELDS: Dict[str, Set[str]] = {
    "people": {"name"},
    "organizations": {"name"},
    "locations": {"name"},
    "events": {"title", "description", "event_type", "start_date"},
}


def _get_required_fields(entity_type: str, domain: str) -> Set[str]:
    """Derive required fields from the domain's Pydantic schema.

    Falls back to _FALLBACK_REQUIRED_FIELDS if schema loading fails.
    """
    try:
        from src.dynamic_models import (
            get_event_model,
            get_location_model,
            get_organization_model,
            get_person_model,
        )

        model_getters = {
            "people": get_person_model,
            "organizations": get_organization_model,
            "locations": get_location_model,
            "events": get_event_model,
        }
        getter = model_getters.get(entity_type)
        if getter:
            model = getter(domain)
            return {
                name
                for name, field in model.model_fields.items()
                if field.is_required()
            }
    except Exception:
        logger.debug(
            f"Could not load schema for {entity_type}/{domain}, using fallback required fields"
        )
    return _FALLBACK_REQUIRED_FIELDS.get(entity_type, set())


class ExtractionQCReport(BaseModel):
    """Summary of what extraction QC did to a batch of entities."""

    input_count: int = 0
    dropped_missing_required: int = 0
    deduped: int = 0
    output_count: int = 0
    flags: List[str] = Field(default_factory=list)
    fixes: Dict[str, Any] = Field(default_factory=dict)


def normalize_name(s: Any) -> str:
    """Normalize an entity name: strip/collapse whitespace, normalize unicode."""
    text = str(s or "").strip()
    text = " ".join(text.split())  # collapse whitespace runs
    return unicodedata.normalize("NFC", text)


def _entity_dedup_key(entity_type: str, e: Dict[str, Any]) -> Tuple:
    """Derive a dedup key for within-article deduplication."""
    if entity_type == "people":
        return (normalize_name(e.get("name")),)
    if entity_type in ("organizations", "locations"):
        return (normalize_name(e.get("name")), str(e.get("type") or "").strip())
    if entity_type == "events":
        return (
            normalize_name(e.get("title")),
            str(e.get("start_date") or "").strip(),
        )
    return (id(e),)


def _check_required_fields(
    entity_type: str, entity: Dict[str, Any], domain: str = "guantanamo"
) -> Optional[str]:
    """Return the name of the first missing required field, or None if all present."""
    required = _get_required_fields(entity_type, domain)
    for field in sorted(required):
        val = entity.get(field)
        if val is None or (isinstance(val, str) and not val.strip()):
            return field
    return None


def run_extraction_qc(
    *,
    entity_type: str,
    entities: List[Dict[str, Any]],
    domain: str = "guantanamo",
    min_name_len: int = QC_MIN_NAME_LENGTH,
) -> Tuple[List[Dict[str, Any]], ExtractionQCReport]:
    """Run deterministic QC on a batch of extracted entities.

    Returns (cleaned_entities, report). Never raises.
    """
    report = ExtractionQCReport(input_count=len(entities))
    flags: List[str] = []
    normalized_count = 0

    if not entities:
        flags.append("zero_entities")
        report.flags = flags
        return [], report

    cleaned: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for entity in entities:
        # 1. Required-field check
        missing = _check_required_fields(entity_type, entity, domain)
        if missing:
            report.dropped_missing_required += 1
            flags.append(f"missing_required:{missing}")
            logger.debug(
                f"Dropped {entity_type} entity missing required field '{missing}': "
                f"{entity}"
            )
            continue

        # 2. Normalize name/title
        name_field = "title" if entity_type == "events" else "name"
        raw_name = entity.get(name_field, "")
        normed = normalize_name(raw_name)

        if normed != str(raw_name or "").strip():
            normalized_count += 1
        entity[name_field] = normed

        # 3. Short-name flag
        if len(normed) < min_name_len:
            flags.append(f"short_name:{normed}")

        # 4. Within-article dedup
        key = _entity_dedup_key(entity_type, entity)
        if key in seen_keys:
            report.deduped += 1
            continue
        seen_keys.add(key)

        cleaned.append(entity)

    report.output_count = len(cleaned)
    report.fixes = {"normalized_names": normalized_count}

    if report.dropped_missing_required > len(entities) * 0.5 and len(entities) > 2:
        flags.append("high_drop_rate")
    if report.deduped > len(entities) * 0.5 and len(entities) > 2:
        flags.append("many_duplicates")

    report.flags = flags
    return cleaned, report


# ──────────────────────────────────────────────
# Profile QC
# ──────────────────────────────────────────────


class ProfileQCReport(BaseModel):
    """Summary of profile quality checks."""

    text_length: int = 0
    citation_count: int = 0
    tag_count: int = 0
    confidence: Optional[float] = None
    passed: bool = True
    flags: List[str] = Field(default_factory=list)
    fixes: Dict[str, Any] = Field(default_factory=dict)


def run_profile_qc(
    *,
    profile: Dict[str, Any],
    min_text_len: int = PROFILE_QC_MIN_TEXT_LENGTH,
    min_tags: int = PROFILE_QC_MIN_TAG_COUNT,
    require_citations: bool = True,
) -> Tuple[Dict[str, Any], ProfileQCReport]:
    """Run deterministic QC on a generated profile dict.

    Applies safe fixes (clamp confidence, default tags) and flags issues.
    Returns (possibly-fixed profile, report). Never raises.
    """
    report = ProfileQCReport()
    flags: List[str] = []
    fixes: Dict[str, Any] = {}

    text = profile.get("text", "")
    report.text_length = len(text)

    # Text length check
    if len(text) < min_text_len:
        flags.append("text_too_short")
        report.passed = False

    # Citation check
    citations = CITATION_RE.findall(text)
    report.citation_count = len(citations)
    if require_citations and not citations:
        flags.append("no_citations")

    # Tag check
    tags = profile.get("tags", [])
    report.tag_count = len(tags)
    if len(tags) < min_tags:
        flags.append("tags_below_minimum")
        profile["tags"] = tags if tags else ["needs-review"]
        fixes["tags_defaulted"] = True

    # Confidence check
    confidence = profile.get("confidence")
    if confidence is None or not isinstance(confidence, (int, float)):
        flags.append("confidence_missing_or_invalid")
        profile["confidence"] = 0.0
        fixes["confidence_set_default"] = True
        report.confidence = 0.0
    else:
        if confidence < 0.0 or confidence > 1.0:
            flags.append("confidence_clamped")
            profile["confidence"] = max(0.0, min(1.0, float(confidence)))
            fixes["confidence_clamped"] = True
        report.confidence = profile["confidence"]

    report.flags = flags
    report.fixes = fixes

    if flags and "text_too_short" not in flags:
        # Minor issues don't fail the QC overall
        report.passed = True

    return profile, report
