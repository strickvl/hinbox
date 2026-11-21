"""Typed human judgments over immutable research proposals."""

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator


class StaleReviewError(RuntimeError):
    """Raised when a review was based on a superseded review state."""


class UnacceptableProposalError(ValueError):
    """Raised when deterministic evidence invariants forbid acceptance."""


class ProposalJudgment(str, Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"
    DEFER = "defer"


class CorrectnessJudgment(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNCERTAIN = "uncertain"
    NOT_ASSESSED = "not_assessed"


class RelevanceJudgment(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    UNCERTAIN = "uncertain"
    NOT_ASSESSED = "not_assessed"


class WorkspaceInclusion(str, Enum):
    NOT_DECIDED = "not_decided"
    INCLUDED = "included"
    EXCLUDED = "excluded"


class ProposalReviewCommand(BaseModel):
    episode_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    expected_version: int = Field(ge=0)
    judgment: ProposalJudgment
    correctness: CorrectnessJudgment = CorrectnessJudgment.NOT_ASSESSED
    relevance: RelevanceJudgment = RelevanceJudgment.NOT_ASSESSED
    inclusion: WorkspaceInclusion = WorkspaceInclusion.NOT_DECIDED
    reason: str | None = None
    inclusion_reason: str | None = None
    edited_payload: Dict[str, Any] | None = None
    actor_id: str = "historian"
    logical_record_id: str | None = None
    supersedes_record_version_id: str | None = None
    evaluation_eligible: bool = True
    training_eligible: bool = False

    @model_validator(mode="after")
    def edit_payload_matches_judgment(self) -> "ProposalReviewCommand":
        if self.judgment is ProposalJudgment.EDIT and self.edited_payload is None:
            raise ValueError("edit review requires edited_payload")
        if (
            self.judgment is not ProposalJudgment.EDIT
            and self.edited_payload is not None
        ):
            raise ValueError("edited_payload is only valid for an edit review")
        return self


class ProposalReviewResult(BaseModel):
    subject_id: str
    decision_id: str
    episode_id: str
    action_id: str
    version: int
    judgment: ProposalJudgment
    correctness: CorrectnessJudgment
    relevance: RelevanceJudgment
    inclusion: WorkspaceInclusion
    reason: str | None = None
    inclusion_reason: str | None = None
    logical_record_id: str | None = None
    record_version_id: str | None = None
