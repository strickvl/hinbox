"""Profile grounding: check each cited claim in a profile against the article
it cites, one model call per article.

A profile sentence like "He was transferred in 2016.^[a1, a2]" gives two
claims, one for a1 and one for a2. Each article's claims go to the model with
that article's text. The report keeps, per article, whether the check worked
(``SourceCheck``), so a later run re-asks only for articles whose check
failed, whose text became available, or whose claims changed.
"""

import contextvars
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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
# Claims sent in one call. A call for one article can hold claims from many
# profiles; more than this are split over several calls, so a reply never
# grows past the output limit.
GROUNDING_MAX_CLAIMS_PER_CALL = int(os.getenv("HINBOX_GROUNDING_MAX_CLAIMS", "30"))
GROUNDING_MAX_TOKENS = int(os.getenv("HINBOX_GROUNDING_MAX_TOKENS", "4096"))

_GROUNDING_SYSTEM_PROMPT = """You are a fact-checking assistant verifying whether claims from entity profiles are supported by one source article.

Each claim is numbered and names the entity whose profile it comes from; "he", "she", "it" or "they" in a claim refer to that entity.

For each claim, decide:
- "supported": The source article clearly contains information that supports the claim
- "partial": The source partially supports the claim but some details are not confirmed
- "not_supported": The source does not contain information supporting this claim
- "unclear": The source text is ambiguous or insufficient to determine support

Return one verdict for every claim: its claim_number, support_level, and a reason of at most 15 words. Do not repeat the claim text."""

_GROUNDING_USER_TEMPLATE = """## SOURCE ARTICLE (ID: {article_id})
{article_text}

## CLAIMS TO VERIFY
{claims_text}"""


class ClaimVerdict(BaseModel):
    """The model's verdict on one numbered claim. It doesn't copy the claim
    back, so a reply for 30 claims stays a few hundred tokens."""

    claim_number: int
    support_level: SupportLevel
    reasoning: Optional[str] = None


@dataclass
class GroundingPlan:
    """One profile's claims, with the verdicts already known (reused from the
    last report, or "missing source") filled in and the rest left as None
    until ``judge_pending()`` asks the model."""

    profile_text: str
    entity_name: str
    claims: List[Dict[str, str]]
    by_article: Dict[str, List[Dict[str, str]]]
    results: Dict[str, List[Optional[ClaimVerification]]]
    sources: Dict[str, SourceCheck]
    failed: Set[str] = field(default_factory=set)  # article IDs with a failed call
    asked: Set[str] = field(default_factory=set)  # article IDs sent to the model
    flags: List[str] = field(default_factory=list)
    reused_articles: int = 0

    def pending(self) -> List[Tuple[str, int]]:
        """(article_id, claim index) of every claim still to judge."""
        return [
            (aid, i)
            for aid, results in self.results.items()
            for i, r in enumerate(results)
            if r is None
        ]


def _verification(
    aid: str, claim: Dict[str, str], level: SupportLevel, why: Optional[str], cap: int
) -> ClaimVerification:
    return ClaimVerification(
        article_id=aid,
        citation=claim["citation"],
        claim=claim["claim"][:cap],
        support_level=level,
        reasoning=why,
    )


def plan_grounding(
    *,
    profile_text: str,
    article_texts: Dict[str, str],
    previous: Optional[Dict[str, Any]] = None,
    entity_name: str = "",
    max_article_chars: int = GROUNDING_MAX_ARTICLE_CHARS,
    max_claim_chars: int = GROUNDING_MAX_CLAIM_CHARS,
) -> GroundingPlan:
    """Work out which of a profile's claims need the model.

    Say a profile citing a1 and a2 was checked last run, and an update has
    since added one sentence citing a2. a1's claims are unchanged, so all its
    verdicts are kept. a2 has one new claim: the old a2 claims keep their
    verdicts (same article text, same claim text) and only the new one is
    left pending. A claim is reused only from an article whose earlier check
    succeeded with the same article text.
    """
    claims = _extract_cited_claims(profile_text)
    by_article: Dict[str, List[Dict[str, str]]] = {}
    for c in claims:
        by_article.setdefault(c["article_id"], []).append(c)
    plan = GroundingPlan(
        profile_text=profile_text,
        entity_name=entity_name,
        claims=claims,
        by_article=by_article,
        results={},
        sources={},
    )
    previous_sources, previous_by_article = _previous_results(previous)

    for aid, items in by_article.items():
        claims_hash = _claims_hash(items, max_claim_chars)
        if not (article_texts.get(aid) or "").strip():
            plan.results[aid] = [
                _verification(
                    aid,
                    c,
                    SupportLevel.MISSING_SOURCE,
                    "Source article not available for verification",
                    max_claim_chars,
                )
                for c in items
            ]
            plan.sources[aid] = SourceCheck(
                article_id=aid, status="missing", claims_hash=claims_hash
            )
            continue

        text_hash = _sha(article_texts[aid][:max_article_chars])
        plan.sources[aid] = SourceCheck(
            article_id=aid,
            status="checked",
            text_hash=text_hash,
            claims_hash=claims_hash,
        )
        earlier = previous_sources.get(aid)
        earlier_results = previous_by_article.get(aid, [])
        usable = (
            earlier is not None
            and earlier.status == "checked"
            and earlier.text_hash == text_hash
        )
        if usable and earlier.claims_hash == claims_hash:
            if len(earlier_results) == len(items):
                plan.results[aid] = list(earlier_results)
                plan.reused_articles += 1
                continue
        known = (
            {v.claim[:max_claim_chars]: v for v in earlier_results} if usable else {}
        )
        article = _squash(article_texts[aid])
        plan.results[aid] = [
            _settled(aid, c, known, article, max_claim_chars) for c in items
        ]
    return plan


def _settled(
    aid: str,
    claim: Dict[str, str],
    known: Dict[str, ClaimVerification],
    squashed_article: str,
    max_claim_chars: int,
) -> Optional[ClaimVerification]:
    """A verdict that needs no model call: kept from the last report, or
    "supported" for a sentence quoted from the article. None otherwise."""
    text = claim["claim"][:max_claim_chars]
    if text in known:
        return known[text].model_copy(update={"citation": claim["citation"]})
    if _is_quote(claim["claim"], squashed_article):
        why = "Quoted word for word from the article"
        return _verification(aid, claim, SupportLevel.SUPPORTED, why, max_claim_chars)
    return None


_SPACE = re.compile(r"\s+")
_QUOTE_MIN_CHARS = 40


def _squash(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


def _is_quote(claim: str, squashed_article: str) -> bool:
    """True when the claim is a sentence copied word for word from the article
    (case and spacing aside), like the "Mentions" sentences of a stub
    profile. Such a claim needs no model call. Short claims ("He was held")
    don't count: a few common words can appear anywhere."""
    text = _squash(claim).rstrip(".;:, ")
    return len(text) >= _QUOTE_MIN_CHARS and text in squashed_article


def _claim_line(n: int, entity_name: str, claim: str) -> str:
    about = f"({entity_name}) " if entity_name else ""
    return f"{n}. {about}{claim}"


@timed("grounding")
def judge_claims(
    *,
    article_id: str,
    article_text: str,
    claims: List[Tuple[str, str]],
    model_type: str,
) -> List[Optional[Tuple[SupportLevel, Optional[str]]]]:
    """One model call: judge ``claims`` (entity name, claim text) against one
    article. Returns a verdict per claim, or None where the reply had none.
    Raises when the call itself fails."""
    from src.utils.llm import cloud_generation, local_generation

    claims_text = "\n".join(
        _claim_line(i + 1, name, text) for i, (name, text) in enumerate(claims)
    )
    messages = [
        {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _GROUNDING_USER_TEMPLATE.format(
                article_id=article_id,
                article_text=article_text,
                claims_text=claims_text,
            ),
        },
    ]
    generate = local_generation if model_type == "ollama" else cloud_generation
    reply = generate(
        messages=messages,
        response_model=List[ClaimVerdict],
        temperature=0,
        max_tokens=GROUNDING_MAX_TOKENS,
    )
    out: List[Optional[Tuple[SupportLevel, Optional[str]]]] = [None] * len(claims)
    numbers = [getattr(v, "claim_number", None) for v in reply]
    by_number = len(set(numbers)) == len(numbers) and all(
        isinstance(n, int) and 1 <= n <= len(claims) for n in numbers
    )
    for pos, v in enumerate(reply):
        i = v.claim_number - 1 if by_number else pos
        if i < len(claims):
            out[i] = (v.support_level, v.reasoning)
    return out


def _record_verdicts(
    batch: List[Tuple[GroundingPlan, str, int]],
    verdicts: List[Optional[Tuple[SupportLevel, Optional[str]]]],
    error: Optional[str],
    max_claim_chars: int,
) -> None:
    """Write one call's verdicts into the plans it came from. A claim with no
    verdict (failed call, or the reply skipped it) is "unclear" and its
    article is marked failed, so the next run asks again."""
    for (plan, aid, i), verdict in zip(batch, verdicts):
        plan.asked.add(aid)
        claim = plan.by_article[aid][i]
        if verdict is None:
            why = error or "LLM did not return verification for this claim"
            flag = "verification_error" if error else "llm_count_mismatch"
            level = SupportLevel.UNCLEAR
            plan.failed.add(aid)
            if flag not in plan.flags:
                plan.flags.append(flag)
        else:
            level, why = verdict
        plan.results[aid][i] = _verification(aid, claim, level, why, max_claim_chars)


_results_lock = threading.Lock()


def judge_pending(
    plans: List[GroundingPlan],
    article_texts: Dict[str, str],
    model_type: str = "gemini",
    workers: int = 1,
    max_article_chars: int = GROUNDING_MAX_ARTICLE_CHARS,
    max_claim_chars: int = GROUNDING_MAX_CLAIM_CHARS,
    max_claims_per_call: int = GROUNDING_MAX_CLAIMS_PER_CALL,
) -> int:
    """Ask the model about every pending claim in ``plans``, one call per
    article (split when an article has more than ``max_claims_per_call``
    claims), however many profiles cite it. Fills in the plans and returns
    the number of calls made.

    Before, 25 profiles citing the same article meant 25 calls, each sending
    the whole article again. Now they share one or two calls.
    """
    by_article: Dict[str, List[Tuple[GroundingPlan, str, int]]] = {}
    for plan in plans:
        for aid, i in plan.pending():
            by_article.setdefault(aid, []).append((plan, aid, i))
    batches = [
        (aid, items[start : start + max_claims_per_call])
        for aid, items in by_article.items()
        for start in range(0, len(items), max(1, max_claims_per_call))
    ]

    def run(aid: str, batch: List[Tuple[GroundingPlan, str, int]]) -> None:
        claims = [
            (plan.entity_name, plan.by_article[a][i]["claim"][:max_claim_chars])
            for plan, a, i in batch
        ]
        try:
            verdicts = judge_claims(
                article_id=aid,
                article_text=article_texts[aid][:max_article_chars],
                claims=claims,
                model_type=model_type,
            )
            error = None
        except Exception as e:
            logger.error(f"Grounding verification failed for article {aid}: {e}")
            verdicts = [None] * len(batch)
            error = f"Verification error: {e}"
        with _results_lock:
            _record_verdicts(batch, verdicts, error, max_claim_chars)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, run, aid, batch)
            for aid, batch in batches
        ]
        for f in futures:
            f.result()
    return len(batches)


def finish_report(
    plan: GroundingPlan, min_grounding_score: float = 0.7
) -> GroundingReport:
    """Build the stored report from a plan whose claims are all judged."""
    report = GroundingReport(profile_text_hash=_sha(plan.profile_text))
    report.total_citations = len(plan.claims)
    if not plan.claims:
        report.flags.append("no_citations")
        return report
    report.flags.extend(plan.flags)
    verifications: List[ClaimVerification] = []
    for aid, results in plan.results.items():
        verifications.extend(r for r in results if r is not None)
        source = plan.sources[aid]
        if aid in plan.failed:
            source = source.model_copy(update={"status": "error"})
        report.sources.append(source)
    report.reused_articles = plan.reused_articles
    report.checked_articles = len(plan.asked)
    report.verifications = verifications
    report.verified = sum(
        1
        for v in verifications
        if v.support_level in (SupportLevel.SUPPORTED, SupportLevel.PARTIAL)
    )
    report.unverified = sum(
        1
        for v in verifications
        if v.support_level in (SupportLevel.NOT_SUPPORTED, SupportLevel.UNCLEAR)
    )
    report.missing_source = sum(
        1 for v in verifications if v.support_level == SupportLevel.MISSING_SOURCE
    )
    report.grounding_score = report.verified / report.total_citations
    if report.missing_source > 0:
        report.flags.append("missing_sources")
    if any(v.support_level == SupportLevel.NOT_SUPPORTED for v in verifications):
        report.flags.append("unsupported_claims")
    if report.grounding_score < min_grounding_score:
        report.flags.append("low_grounding_score")
        report.passed = False
    return report


def verify_profile_grounding(
    *,
    profile_text: str,
    article_texts: Dict[str, str],
    model_type: str = "gemini",
    max_article_chars: int = GROUNDING_MAX_ARTICLE_CHARS,
    max_claim_chars: int = GROUNDING_MAX_CLAIM_CHARS,
    min_grounding_score: float = 0.7,
    previous: Optional[Dict[str, Any]] = None,
    entity_name: str = "",
) -> GroundingReport:
    """Verify that one profile's claims are supported by their cited sources.

    ``previous`` is the report stored from an earlier check of this entity:
    verdicts for claims it already judged against the same article text are
    kept, and only new or changed claims (and articles whose check failed or
    that were missing) go to the model, one call per article. The grounding
    pass checks many profiles at once with ``plan_grounding()`` and
    ``judge_pending()`` instead, so profiles citing the same article share
    calls.

    Never raises — returns a report with flags on any error.
    """
    plan = plan_grounding(
        profile_text=profile_text,
        article_texts=article_texts,
        previous=previous,
        entity_name=entity_name,
        max_article_chars=max_article_chars,
        max_claim_chars=max_claim_chars,
    )
    judge_pending(
        [plan],
        article_texts,
        model_type=model_type,
        max_article_chars=max_article_chars,
        max_claim_chars=max_claim_chars,
    )
    return finish_report(plan, min_grounding_score)


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
