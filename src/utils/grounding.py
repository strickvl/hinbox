"""Profile grounding: check each cited claim in a profile against the article
it cites, one model call per article.

A profile sentence like "He was transferred in 2016.^[a1, a2]" gives two
claims, one for a1 and one for a2. Each article's claims go to the model with
that article's text. The report keeps, per article, whether the check worked
(``SourceCheck``), so a later run re-asks only for articles whose check
failed, whose text became available, or whose claims changed.
"""

import hashlib
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.logging_config import get_logger
from src.utils.quality_controls import CITATION_RE, citation_ids
from src.utils.step_timing import timed

logger = get_logger("grounding")


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


class SourceCheck(BaseModel):
    """How one cited article's claims were checked, so a later run can tell
    which results it may keep.

    ``status`` is "checked" (the model judged every claim), "error" (the call
    failed or returned too few results: retry next time) or "missing" (the
    article text was not available: retry once it is). ``text_hash`` is the
    hash of the article text the model saw ("" when missing) and
    ``claims_hash`` the hash of the claims sent for it.
    """

    article_id: str
    status: str
    text_hash: str = ""
    claims_hash: str = ""


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
    sources: List[SourceCheck] = Field(default_factory=list)
    checked_articles: int = 0  # model calls made for this report
    reused_articles: int = 0  # articles whose results came from the last report


_LEADING_PUNCT = re.compile(r"^[\s.,;:!?)\]]+")
_HAS_WORD = re.compile(r"\w")


def _extract_cited_claims(profile_text: str) -> List[Dict[str, str]]:
    """Extract citation-anchored claims from profile text.

    Uses CITATION_RE to find all ^[article_id] markers, including grouped
    ones (^[a1, a2] gives one claim per ID). For each marker, the claim is
    the text span between the previous marker's end and this marker's start.

    Returns list of {"article_id": str, "citation": str, "claim": str}.
    """
    claims: List[Dict[str, str]] = []
    last_end = 0
    last_claim = ""

    for match in CITATION_RE.finditer(profile_text):
        ids = citation_ids(match.group(1))
        if not ids:
            continue
        citation = match.group(0)  # e.g. "^[art-123]" or "^[a1, a2]"

        # Claim is the text between the end of the last citation and this one
        # The span starts right after the previous marker, so "fact^[a1]. Next
        # fact^[a2]" leaves ". Next fact" here: drop that leading punctuation.
        # A span with no letters or digits at all ("fact^[a1].^[a2]") is not
        # a claim, so it reuses the previous one.
        claim_text = _LEADING_PUNCT.sub("", profile_text[last_end : match.start()])
        claim_text = claim_text.strip()
        if not _HAS_WORD.search(claim_text):
            claim_text = last_claim  # reuse last non-empty claim for adjacent citations

        if claim_text:
            last_claim = claim_text

        for article_id in ids:
            claims.append(
                {
                    "article_id": article_id,
                    "citation": citation,
                    "claim": claim_text or "(no claim text)",
                }
            )
        last_end = match.end()

    return claims


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _claims_hash(claims: List[Dict[str, str]], max_claim_chars: int) -> str:
    return _sha(
        "\n".join(f"{c['citation']}|{c['claim'][:max_claim_chars]}" for c in claims)
    )


def _previous_results(
    previous: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, SourceCheck], Dict[str, List[ClaimVerification]]]:
    """Source checks and per-article verifications from an earlier report
    (a dict, as stored on the entity). Anything unreadable is ignored."""
    if not isinstance(previous, dict):
        return {}, {}
    try:
        sources = {
            s.article_id: s
            for s in (SourceCheck(**d) for d in previous.get("sources") or [])
        }
        by_article: Dict[str, List[ClaimVerification]] = {}
        for d in previous.get("verifications") or []:
            v = ClaimVerification(**d)
            by_article.setdefault(v.article_id, []).append(v)
        return sources, by_article
    except Exception:
        return {}, {}


# Article text sent to the grounding check (and hashed to spot changes)
GROUNDING_MAX_ARTICLE_CHARS = 12000
GROUNDING_MAX_CLAIM_CHARS = 600

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


@timed("grounding")
def verify_profile_grounding(
    *,
    profile_text: str,
    article_texts: Dict[str, str],
    model_type: str = "gemini",
    max_article_chars: int = GROUNDING_MAX_ARTICLE_CHARS,
    max_claim_chars: int = GROUNDING_MAX_CLAIM_CHARS,
    min_grounding_score: float = 0.7,
    previous: Optional[Dict[str, Any]] = None,
) -> GroundingReport:
    """Verify that profile claims are supported by their cited sources.

    Extracts citations from profile text, groups them by article, and uses
    an LLM call per article to verify each claim. Returns a GroundingReport
    with per-claim details and summary statistics.

    ``previous`` is the report stored from an earlier check of this entity.
    An article whose earlier check succeeded, with the same article text and
    the same claims sent, keeps its earlier results without a model call.
    Articles that failed, were missing, or whose claims changed are checked
    again, so the model is called at most once per such article.

    Never raises — returns a report with flags on any error.
    """
    from src.utils.llm import cloud_generation, local_generation

    report = GroundingReport(profile_text_hash=_sha(profile_text))

    cited_claims = _extract_cited_claims(profile_text)
    report.total_citations = len(cited_claims)

    if not cited_claims:
        report.flags.append("no_citations")
        report.grounding_score = None
        return report

    previous_sources, previous_by_article = _previous_results(previous)

    # Group claims by article, keeping first-seen article order
    by_article: Dict[str, List[Dict[str, str]]] = {}
    for claim_item in cited_claims:
        by_article.setdefault(claim_item["article_id"], []).append(claim_item)

    def fallback(aid: str, claim_item: Dict[str, str], level: SupportLevel, why: str):
        return ClaimVerification(
            article_id=aid,
            citation=claim_item["citation"],
            claim=claim_item["claim"][:max_claim_chars],
            support_level=level,
            reasoning=why,
        )

    all_verifications: List[ClaimVerification] = []

    for aid, claims_for_article in by_article.items():
        claims_hash = _claims_hash(claims_for_article, max_claim_chars)
        if aid not in article_texts or not article_texts[aid].strip():
            all_verifications.extend(
                fallback(
                    aid,
                    c,
                    SupportLevel.MISSING_SOURCE,
                    "Source article not available for verification",
                )
                for c in claims_for_article
            )
            report.sources.append(
                SourceCheck(article_id=aid, status="missing", claims_hash=claims_hash)
            )
            continue

        source_text = article_texts[aid][:max_article_chars]
        text_hash = _sha(source_text)
        earlier = previous_sources.get(aid)
        earlier_results = previous_by_article.get(aid, [])
        if (
            earlier is not None
            and earlier.status == "checked"
            and earlier.text_hash == text_hash
            and earlier.claims_hash == claims_hash
            and len(earlier_results) == len(claims_for_article)
        ):
            all_verifications.extend(earlier_results)
            report.sources.append(earlier)
            report.reused_articles += 1
            continue

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

        status = "checked"
        report.checked_articles += 1
        try:
            generate = local_generation if model_type == "ollama" else cloud_generation
            verifications_list = generate(
                messages=messages,
                response_model=List[ClaimVerification],
                temperature=0,
            )

            # Match returned verifications to input claims by position. The
            # model's own copy of the ID, citation and claim is replaced by
            # what was sent, so a later run can find these results again.
            for i, claim_item in enumerate(claims_for_article):
                if i < len(verifications_list):
                    v = verifications_list[i]
                    all_verifications.append(
                        fallback(aid, claim_item, v.support_level, v.reasoning)
                    )
                else:
                    all_verifications.append(
                        fallback(
                            aid,
                            claim_item,
                            SupportLevel.UNCLEAR,
                            "LLM did not return verification for this claim",
                        )
                    )

            if len(verifications_list) != len(claims_for_article):
                report.flags.append("llm_count_mismatch")
            if len(verifications_list) < len(claims_for_article):
                status = "error"  # some claims were never judged: retry them

        except Exception as e:
            logger.error(f"Grounding verification failed for article {aid}: {e}")
            all_verifications.extend(
                fallback(aid, c, SupportLevel.UNCLEAR, f"Verification error: {str(e)}")
                for c in claims_for_article
            )
            report.flags.append("verification_error")
            status = "error"

        report.sources.append(
            SourceCheck(
                article_id=aid,
                status=status,
                text_hash=text_hash,
                claims_hash=claims_hash,
            )
        )

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
    report.grounding_score = report.verified / report.total_citations

    if report.missing_source > 0:
        report.flags.append("missing_sources")
    if any(v.support_level == SupportLevel.NOT_SUPPORTED for v in all_verifications):
        report.flags.append("unsupported_claims")
    if report.grounding_score < min_grounding_score:
        report.flags.append("low_grounding_score")
        report.passed = False

    return report


_RETRY_FLAGS = ("verification_error", "llm_count_mismatch")


def grounding_is_current(
    report: Any,
    profile_text: str,
    article_texts: Dict[str, str],
    max_article_chars: int = GROUNDING_MAX_ARTICLE_CHARS,
    max_claim_chars: int = GROUNDING_MAX_CLAIM_CHARS,
) -> bool:
    """True when a stored report still answers for this profile text and
    these articles, so checking again would only repeat the same calls.

    It is not current when the text changed, the citation parser now finds a
    different number of citations, an article's check failed, a missing
    article has become available, a checked article's text changed, or the
    claims parsed from the text now differ from the ones that were checked
    (the claim parser changed). Reports saved before per-article ``sources``
    existed are judged by their flags, missing-source claims and claims with
    no words in them instead.
    """
    if not isinstance(report, dict):
        return False
    if report.get("profile_text_hash") != _sha(profile_text):
        return False
    claims = _extract_cited_claims(profile_text)
    if report.get("total_citations") != len(claims):
        return False
    by_article: Dict[str, List[Dict[str, str]]] = {}
    for c in claims:
        by_article.setdefault(c["article_id"], []).append(c)

    def available(aid: str) -> bool:
        return bool((article_texts.get(aid) or "").strip())

    sources = report.get("sources") or []
    if not sources:
        if any(f in (report.get("flags") or []) for f in _RETRY_FLAGS):
            return False
        old = [v for v in report.get("verifications") or [] if isinstance(v, dict)]
        return not any(
            (
                v.get("support_level") == SupportLevel.MISSING_SOURCE.value
                and available(v.get("article_id", ""))
            )
            or not _HAS_WORD.search(v.get("claim") or "")
            for v in old
        )
    if {s.get("article_id", "") for s in sources} != set(by_article):
        return False
    for s in sources:
        aid = s.get("article_id", "")
        status = s.get("status")
        if s.get("claims_hash") != _claims_hash(by_article[aid], max_claim_chars):
            return False
        if status == "missing":
            if available(aid):
                return False
        elif status == "checked":
            text = (article_texts.get(aid) or "")[:max_article_chars]
            if not available(aid) or s.get("text_hash") != _sha(text):
                return False
        else:
            return False
    return True
