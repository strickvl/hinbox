"""Merge dispute agent for resolving ambiguous entity matches.

When the primary match checker returns a result in the "gray band" (similarity
near the threshold) with low confidence, this module provides a second-stage
analysis that can override the initial merge/skip decision.

The agent uses a structured LLM call (mirroring match_checker.py patterns)
to produce a MergeDisputeDecision with action, confidence, and reasoning.
"""

import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.logging_config import log
from src.utils.llm import cloud_generation, local_generation


class MergeDisputeAction(str, Enum):
    MERGE = "merge"
    SKIP = "skip"
    DEFER = "defer"


class MergeDisputeDecision(BaseModel):
    action: MergeDisputeAction
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How confident the agent is in its decision (0=uncertain, 1=certain)",
    )
    reason: str


_SYSTEM_PROMPT = """You are an expert entity-resolution adjudicator for news articles about Guantánamo Bay.

A primary match checker has already compared evidence from a new article against an existing entity profile but returned an UNCERTAIN result.
Your job is to make a final determination: should these entities be merged, kept separate, or deferred for human review?

Consider carefully:
1. Name variations: Different spellings, nicknames, titles, or partial names
2. Contextual overlap: Shared roles, affiliations, actions, biographical details
3. Temporal consistency: Could both profiles describe the same entity at different times?
4. Contradictions: Any facts that are mutually exclusive (different birth dates, different nationalities, etc.)
5. Sub-location rule: A camp inside Guantánamo Bay is NOT the same as Guantánamo Bay itself

You MUST provide:
- action: "merge" (same entity), "skip" (different entities), or "defer" (too ambiguous for automated decision)
- confidence: a float from 0.0 to 1.0 indicating how certain you are
- reason: a detailed explanation citing specific evidence

Choose "defer" only when evidence is genuinely contradictory or insufficient to decide."""

_USER_TEMPLATE = """The primary match checker was uncertain about these two profiles.

## MATCH CHECKER RESULT
- Decision: {match_decision}
- Confidence: {match_confidence:.2f}
- Reason: {match_reason}

## SIMILARITY CONTEXT
- Embedding similarity: {similarity_score:.4f}
- Merge threshold: {similarity_threshold:.4f}
- This falls within the ambiguous "gray band" around the threshold.

## EVIDENCE FROM NEW ARTICLE
Name: {new_name}
{new_profile_text}

## EXISTING PROFILE IN DATABASE
Name: {existing_name}
Profile: {existing_profile_text}

Based on ALL available evidence, should these entities be merged, kept separate, or deferred for human review?"""


def run_merge_dispute_agent(
    *,
    entity_type: str,
    new_name: str,
    existing_name: str,
    new_profile_text: str,
    existing_profile_text: str,
    similarity_score: float,
    similarity_threshold: float,
    match_is_match: bool,
    match_confidence: float,
    match_reason: str,
    model_type: str = "gemini",
    domain: str = "guantanamo",
    article_id: Optional[str] = None,
    review_queue_path: Optional[str] = None,
) -> MergeDisputeDecision:
    """Run the dispute agent to resolve an ambiguous merge decision.

    This is invoked only when the match check result falls in the gray band
    (similarity near threshold AND confidence below cutoff). The agent makes
    a more deliberate assessment using richer prompt context.

    Returns a MergeDisputeDecision. On error, defaults to DEFER to prevent
    false merges.
    """
    user_content = _USER_TEMPLATE.format(
        match_decision="MATCH" if match_is_match else "NO MATCH",
        match_confidence=match_confidence,
        match_reason=match_reason,
        similarity_score=similarity_score,
        similarity_threshold=similarity_threshold,
        new_name=new_name,
        new_profile_text=new_profile_text,
        existing_name=existing_name,
        existing_profile_text=existing_profile_text,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        if model_type == "ollama":
            decision = local_generation(
                messages=messages,
                response_model=MergeDisputeDecision,
                model=OLLAMA_MODEL,
                temperature=0,
            )
        else:
            decision = cloud_generation(
                messages=messages,
                response_model=MergeDisputeDecision,
                model=CLOUD_MODEL,
                temperature=0,
            )
    except Exception as e:
        log(f"Dispute agent error: {e}", level="error")
        decision = MergeDisputeDecision(
            action=MergeDisputeAction.DEFER,
            confidence=0.0,
            reason=f"Dispute agent API error: {str(e)}",
        )

    log(
        f"Dispute agent decision for '{new_name}' vs '{existing_name}': "
        f"{decision.action.value} (confidence={decision.confidence:.2f})",
        level="info",
    )

    if decision.action == MergeDisputeAction.DEFER and review_queue_path:
        record = _build_review_record(
            entity_type=entity_type,
            new_name=new_name,
            existing_name=existing_name,
            similarity_score=similarity_score,
            similarity_threshold=similarity_threshold,
            match_is_match=match_is_match,
            match_confidence=match_confidence,
            match_reason=match_reason,
            decision=decision,
            domain=domain,
            article_id=article_id,
        )
        append_merge_dispute_review_queue(review_queue_path, record)

    return decision


def _build_review_record(
    *,
    entity_type: str,
    new_name: str,
    existing_name: str,
    similarity_score: float,
    similarity_threshold: float,
    match_is_match: bool,
    match_confidence: float,
    match_reason: str,
    decision: MergeDisputeDecision,
    domain: str,
    article_id: Optional[str],
) -> Dict[str, Any]:
    """Build a JSON-serializable record for the review queue."""
    return {
        "domain": domain,
        "entity_type": entity_type,
        "new_key": new_name,
        "candidate_key": existing_name,
        "similarity_score": similarity_score,
        "similarity_threshold": similarity_threshold,
        "match_check": {
            "is_match": match_is_match,
            "confidence": match_confidence,
            "reason": match_reason,
        },
        "dispute_decision": {
            "action": decision.action.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
        },
        "article_id": article_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def append_merge_dispute_review_queue(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON line to the merge dispute review queue."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log(f"Appended dispute review record to {path}", level="info")
    except Exception as e:
        log(f"Failed to write dispute review record: {e}", level="error")
