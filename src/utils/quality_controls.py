"""Deterministic quality controls for extraction and profile outputs.

These checks run *after* LLM generation and apply rules that don't need
a model: required-field validation, name normalization, within-article
dedup, and suspicious-result flagging.

QC functions never raise — they only drop invalid items and report.
"""

import hashlib
import re
import unicodedata
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.config_loader import get_domain_config
from src.constants import (
    PROFILE_QC_MIN_TAG_COUNT,
    PROFILE_QC_MIN_TEXT_LENGTH,
    QC_MIN_NAME_LENGTH,
)
from src.logging_config import get_logger
from src.utils.name_variants import (
    is_low_quality_name,
    names_likely_same,
    score_canonical_name,
)

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


def _collapse_within_article_variants(
    *,
    entity_type: str,
    entities: List[Dict[str, Any]],
    domain: str = "guantanamo",
) -> Tuple[List[Dict[str, Any]], int]:
    """Collapse name variants within a single article's extraction results.

    For organizations and locations, detects when two extracted entities
    likely refer to the same real-world entity (via acronym matching,
    substring containment, or configured equivalence groups) and merges
    them — keeping the more canonical name (proper nouns over descriptive
    phrases) and adding the other to an ``aliases`` list.

    For people and events, this is a no-op (returns entities unchanged).

    Returns (consolidated_entities, collapsed_count).
    """
    if entity_type not in ("organizations", "locations"):
        return entities, 0

    if len(entities) <= 1:
        return entities, 0

    # Load equivalence groups from domain config
    try:
        cfg = get_domain_config(domain)
        variants_cfg = cfg.get_name_variants_config(entity_type)
        equivalence_groups = variants_cfg.get("equivalence_groups", [])
    except Exception:
        equivalence_groups = []

    name_field = "name"
    collapsed_count = 0

    # Track which entities have been absorbed into another
    absorbed: Set[int] = set()

    for i in range(len(entities)):
        if i in absorbed:
            continue
        for j in range(i + 1, len(entities)):
            if j in absorbed:
                continue

            name_i = entities[i].get(name_field, "")
            name_j = entities[j].get(name_field, "")
            type_i = str(entities[i].get("type") or "").strip()
            type_j = str(entities[j].get("type") or "").strip()

            # Only collapse entities of the same type
            if type_i and type_j and type_i != type_j:
                continue

            if names_likely_same(
                name_i,
                name_j,
                entity_type=entity_type,
                equivalence_groups=equivalence_groups,
            ):
                # Keep the more canonical name (proper nouns over descriptions)
                if score_canonical_name(name_i) >= score_canonical_name(name_j):
                    keep_idx, drop_idx = i, j
                else:
                    keep_idx, drop_idx = j, i

                keep = entities[keep_idx]
                drop = entities[drop_idx]

                # Add dropped name to aliases
                keep.setdefault("aliases", [])
                drop_name = drop.get(name_field, "")
                if drop_name and drop_name not in keep["aliases"]:
                    keep["aliases"].append(drop_name)

                # Also absorb any aliases from the dropped entity
                for alias in drop.get("aliases", []):
                    if alias not in keep["aliases"]:
                        keep["aliases"].append(alias)

                absorbed.add(drop_idx)
                collapsed_count += 1
                logger.debug(
                    f"Collapsed '{drop_name}' into '{keep.get(name_field)}' "
                    f"(aliases: {keep['aliases']})"
                )

    result = [e for idx, e in enumerate(entities) if idx not in absorbed]
    return result, collapsed_count


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

        # 4. Within-article dedup (exact key match)
        key = _entity_dedup_key(entity_type, entity)
        if key in seen_keys:
            report.deduped += 1
            continue
        seen_keys.add(key)

        cleaned.append(entity)

    # 5. Variant consolidation (acronym/substring/equivalence-group dedup)
    cleaned, collapsed_count = _collapse_within_article_variants(
        entity_type=entity_type,
        entities=cleaned,
        domain=domain,
    )
    report.deduped += collapsed_count

    report.output_count = len(cleaned)
    report.fixes = {
        "normalized_names": normalized_count,
        "collapsed_variants": collapsed_count,
    }

    if report.dropped_missing_required > len(entities) * 0.5 and len(entities) > 2:
        flags.append("high_drop_rate")
    if report.deduped > len(entities) * 0.5 and len(entities) > 2:
        flags.append("many_duplicates")

    # 6. Flag if many entities have generic/descriptive names
    if cleaned:
        name_field = "title" if entity_type == "events" else "name"
        low_quality_count = sum(
            1
            for e in cleaned
            if is_low_quality_name(e.get(name_field, ""), entity_type=entity_type)
        )
        if low_quality_count >= 2:
            flags.append("many_low_quality_names")

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


# ──────────────────────────────────────────────
# Profile Grounding Verification
# ──────────────────────────────────────────────


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"
    UNCLEAR = "unclear"
    MISSING_SOURCE = "missing_source"


class ClaimVerification(BaseModel):
    """Result of verifying a single citation-anchored claim against its source."""

    article_id: str
    citation: str = Field(..., description="Original citation marker, e.g. ^[art-123]")
    claim: str = Field(..., description="The text span supported by the citation")
    support_level: SupportLevel
    reasoning: Optional[str] = None


class GroundingReport(BaseModel):
    """Summary of profile grounding verification."""

    profile_text_hash: str = ""
    total_citations: int = 0
    verified: int = 0
    unverified: int = 0
    missing_source: int = 0
    grounding_score: Optional[float] = None
    passed: bool = True
    flags: List[str] = Field(default_factory=list)
    verifications: List[ClaimVerification] = Field(default_factory=list)


def _extract_cited_claims(profile_text: str) -> List[Dict[str, str]]:
    """Extract citation-anchored claims from profile text.

    Uses CITATION_RE to find all ^[article_id] markers. For each citation,
    the claim is the text span between the previous citation end and this
    citation start.

    Returns list of {"article_id": str, "citation": str, "claim": str}.
    """
    claims: List[Dict[str, str]] = []
    last_end = 0
    last_claim = ""

    for match in CITATION_RE.finditer(profile_text):
        article_id = match.group(1)
        citation = match.group(0)  # e.g. "^[art-123]"

        # Claim is the text between the end of the last citation and this one
        claim_text = profile_text[last_end : match.start()].strip()
        if not claim_text:
            claim_text = last_claim  # reuse last non-empty claim for adjacent citations

        if claim_text:
            last_claim = claim_text

        claims.append(
            {
                "article_id": article_id,
                "citation": citation,
                "claim": claim_text or "(no claim text)",
            }
        )
        last_end = match.end()

    return claims


_GROUNDING_SYSTEM_PROMPT = """You are a fact-checking assistant verifying whether claims in an entity profile are supported by source article text.

For each claim-citation pair, determine:
- "supported": The source article clearly contains information that supports the claim
- "partial": The source partially supports the claim but some details are not confirmed
- "not_supported": The source does not contain information supporting this claim
- "unclear": The source text is ambiguous or insufficient to determine support

You MUST return a verification for every claim provided, in the same order."""

_GROUNDING_USER_TEMPLATE = """Verify whether the following claims are supported by the source article.

## SOURCE ARTICLE (ID: {article_id})
{article_text}

## CLAIMS TO VERIFY
{claims_text}

For each claim, provide: article_id, citation, claim, support_level, and brief reasoning."""


def verify_profile_grounding(
    *,
    profile_text: str,
    article_texts: Dict[str, str],
    model_type: str = "gemini",
    max_article_chars: int = 12000,
    max_claim_chars: int = 600,
    min_grounding_score: float = 0.7,
) -> GroundingReport:
    """Verify that profile claims are supported by their cited sources.

    Extracts citations from profile text, groups them by article, and uses
    an LLM call per article to verify each claim. Returns a GroundingReport
    with per-claim details and summary statistics.

    Never raises — returns a report with flags on any error.
    """
    from src.utils.llm import cloud_generation, local_generation

    text_hash = hashlib.sha256(profile_text.encode()).hexdigest()
    report = GroundingReport(profile_text_hash=text_hash)

    cited_claims = _extract_cited_claims(profile_text)
    report.total_citations = len(cited_claims)

    if not cited_claims:
        report.flags.append("no_citations")
        report.grounding_score = None
        return report

    # Separate missing sources from verifiable claims
    verifiable_by_article: Dict[str, List[Dict[str, str]]] = {}
    all_verifications: List[ClaimVerification] = []

    for claim_item in cited_claims:
        aid = claim_item["article_id"]
        if aid not in article_texts or not article_texts[aid].strip():
            all_verifications.append(
                ClaimVerification(
                    article_id=aid,
                    citation=claim_item["citation"],
                    claim=claim_item["claim"][:max_claim_chars],
                    support_level=SupportLevel.MISSING_SOURCE,
                    reasoning="Source article not available for verification",
                )
            )
        else:
            verifiable_by_article.setdefault(aid, []).append(claim_item)

    # Verify claims grouped by article (one LLM call per article)
    for aid, claims_for_article in verifiable_by_article.items():
        source_text = article_texts[aid][:max_article_chars]
        claims_text = "\n".join(
            f"{i + 1}. Citation: {c['citation']} | Claim: {c['claim'][:max_claim_chars]}"
            for i, c in enumerate(claims_for_article)
        )

        user_content = _GROUNDING_USER_TEMPLATE.format(
            article_id=aid,
            article_text=source_text,
            claims_text=claims_text,
        )

        messages = [
            {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            if model_type == "ollama":
                verifications_list = local_generation(
                    messages=messages,
                    response_model=List[ClaimVerification],
                    temperature=0,
                )
            else:
                verifications_list = cloud_generation(
                    messages=messages,
                    response_model=List[ClaimVerification],
                    temperature=0,
                )

            # Match returned verifications to input claims
            for i, claim_item in enumerate(claims_for_article):
                if i < len(verifications_list):
                    v = verifications_list[i]
                    all_verifications.append(v)
                else:
                    # LLM returned fewer results than expected
                    all_verifications.append(
                        ClaimVerification(
                            article_id=aid,
                            citation=claim_item["citation"],
                            claim=claim_item["claim"][:max_claim_chars],
                            support_level=SupportLevel.UNCLEAR,
                            reasoning="LLM did not return verification for this claim",
                        )
                    )

            if len(verifications_list) != len(claims_for_article):
                report.flags.append("llm_count_mismatch")

        except Exception as e:
            logger.error(f"Grounding verification failed for article {aid}: {e}")
            for claim_item in claims_for_article:
                all_verifications.append(
                    ClaimVerification(
                        article_id=aid,
                        citation=claim_item["citation"],
                        claim=claim_item["claim"][:max_claim_chars],
                        support_level=SupportLevel.UNCLEAR,
                        reasoning=f"Verification error: {str(e)}",
                    )
                )
            report.flags.append("verification_error")

    # Compute summary statistics
    report.verifications = all_verifications
    report.verified = sum(
        1
        for v in all_verifications
        if v.support_level in (SupportLevel.SUPPORTED, SupportLevel.PARTIAL)
    )
    report.unverified = sum(
        1
        for v in all_verifications
        if v.support_level in (SupportLevel.NOT_SUPPORTED, SupportLevel.UNCLEAR)
    )
    report.missing_source = sum(
        1 for v in all_verifications if v.support_level == SupportLevel.MISSING_SOURCE
    )

    if report.total_citations > 0:
        report.grounding_score = report.verified / report.total_citations
    else:
        report.grounding_score = None

    if report.missing_source > 0:
        report.flags.append("missing_sources")
    if any(v.support_level == SupportLevel.NOT_SUPPORTED for v in all_verifications):
        report.flags.append("unsupported_claims")
    if (
        report.grounding_score is not None
        and report.grounding_score < min_grounding_score
    ):
        report.flags.append("low_grounding_score")
        report.passed = False

    return report
