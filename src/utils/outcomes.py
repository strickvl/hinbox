"""Structured phase outcomes for pipeline observability.

PhaseOutcome replaces silent fallbacks (return [], return False, etc.) with
objects that carry success/failure status, error information, and QC flags
while still providing a fallback value so the pipeline continues.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OutcomeError(BaseModel):
    """Serializable error payload for a failed phase."""

    type: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class PhaseOutcome(BaseModel):
    """Result of a pipeline phase that preserves both value and metadata.

    The key idea: even failed phases return a usable `value` (the fallback)
    so downstream code can proceed. But `success`, `error`, and `flags` make
    the failure *visible* in processing_metadata.
    """

    phase: str
    success: bool
    value: Any = None
    error: Optional[OutcomeError] = None
    counts: Dict[str, int] = Field(default_factory=dict)
    flags: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Serialize for storage in processing_metadata (drops `value` to save space)."""
        d = self.model_dump(exclude_none=True)
        d.pop("value", None)
        return d

    @classmethod
    def ok(
        cls,
        phase: str,
        *,
        value: Any,
        counts: Optional[Dict[str, int]] = None,
        flags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "PhaseOutcome":
        return cls(
            phase=phase,
            success=True,
            value=value,
            counts=counts or {},
            flags=flags or [],
            meta=meta or {},
        )

    @classmethod
    def fail(
        cls,
        phase: str,
        *,
        error: Exception,
        fallback: Any,
        context: Optional[Dict[str, Any]] = None,
        counts: Optional[Dict[str, int]] = None,
        flags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "PhaseOutcome":
        return cls(
            phase=phase,
            success=False,
            value=fallback,
            error=OutcomeError(
                type=type(error).__name__,
                message=str(error),
                context=context or {},
            ),
            counts=counts or {},
            flags=flags or [],
            meta=meta or {},
        )
