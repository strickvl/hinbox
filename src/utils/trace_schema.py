"""Canonical trace schema for machine-readable pipeline decision events."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TraceStage(str, Enum):
    """Pipeline stages that emit trace events."""

    RELEVANCE = "relevance"
    EXTRACTION = "extraction"
    EXTRACTION_QC = "extraction.qc"
    EXTRACTION_RETRY = "extraction.retry"
    MERGE_LEXICAL = "merge.lexical_block"
    MERGE_EMBEDDING = "merge.embedding"
    MERGE_MATCH_CHECK = "merge.match_check"
    MERGE_DISPUTE = "merge.dispute"
    MERGE_DECISION = "merge.decision"
    PROFILE_CREATE = "profile.create"
    PROFILE_UPDATE = "profile.update"
    PROFILE_QC = "profile.qc"
    GROUNDING = "grounding"


class TraceEvent(BaseModel):
    """A single machine-readable pipeline decision event."""

    # Identity
    trace_version: str = "1.0"
    run_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: TraceStage

    # Context
    article_id: Optional[str] = None
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    candidate_name: Optional[str] = None

    # Inputs (by reference)
    inputs: Dict[str, str] = Field(default_factory=dict)

    # Model + prompt
    model: Optional[Dict[str, Any]] = None
    prompt_fingerprint: Optional[str] = None

    # Decision output
    decision: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None

    # Quality
    qc_flags: List[str] = Field(default_factory=list)

    # Metrics
    token_count: Optional[int] = None
    latency_ms: Optional[int] = None
    counts: Dict[str, int] = Field(default_factory=dict)

    # Extensible payload
    meta: Dict[str, Any] = Field(default_factory=dict)
