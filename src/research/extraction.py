"""Typed, replayable proposal generation for evidence-first research."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Callable, Dict, List, Literal, Union

from pydantic import BaseModel, Field, model_validator

from src.research.locators import TextLocator
from src.research.models import new_research_id
from src.utils.cache_utils import sha256_jsonable


class KnowledgeBasis(str, Enum):
    OBSERVATION = "observation"
    PARTICIPATION = "participation"
    ATTRIBUTED_REPORT = "attributed_report"
    HEARSAY = "hearsay"
    DOCUMENTARY_ACCESS = "documentary_access"
    RETROSPECTIVE_RECOLLECTION = "retrospective_recollection"
    LATER_SYNTHESIS = "later_synthesis"
    UNKNOWN = "unknown"


class EvidenceScope(str, Enum):
    WORKSPACE = "workspace"
    EXPLICIT_OUT_OF_WORKSPACE = "explicit_out_of_workspace"


class ToolResult(BaseModel):
    tool_call_id: str = Field(default_factory=lambda: new_research_id("tool_call"))
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any]


class PassageAction(BaseModel):
    action_type: Literal["passage"] = "passage"
    action_id: str = Field(default_factory=lambda: new_research_id("action"))
    locator: TextLocator
    selected_quote: str = Field(min_length=1)
    context_before: str = ""
    context_after: str = ""
    boundary_state: Literal["provisional"] = "provisional"


class SourceAssertionAction(BaseModel):
    action_type: Literal["source_assertion"] = "source_assertion"
    action_id: str = Field(default_factory=lambda: new_research_id("action"))
    passage_action_ids: List[str] = Field(min_length=1)
    assertion_text: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    occurrence_time: str | None = None
    occurrence_time_precision: Literal[
        "exact", "day", "month", "year", "range", "approximate", "unknown"
    ] = "unknown"
    allegation: bool = False
    uncertainty: str | None = None


class NormalizedClaimAction(BaseModel):
    action_type: Literal["normalized_claim"] = "normalized_claim"
    action_id: str = Field(default_factory=lambda: new_research_id("action"))
    assertion_action_ids: List[str] = Field(min_length=1)
    passage_action_ids: List[str] = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    material_qualifiers: List[str] = Field(default_factory=list)
    allegation: bool = False
    attribution: str | None = None
    state: Literal["provisional"] = "provisional"


class MentionAction(BaseModel):
    action_type: Literal["mention"] = "mention"
    action_id: str = Field(default_factory=lambda: new_research_id("action"))
    passage_action_ids: List[str] = Field(min_length=1)
    anchor_kind: Literal["person", "organization", "place", "event", "source"]
    surface_form: str = Field(min_length=1)


class RelationshipAction(BaseModel):
    action_type: Literal["relationship"] = "relationship"
    action_id: str = Field(default_factory=lambda: new_research_id("action"))
    passage_action_ids: List[str] = Field(min_length=1)
    subject_candidate: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_candidate: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    allegation: bool = False


class KnowledgePositionAction(BaseModel):
    action_type: Literal["knowledge_position"] = "knowledge_position"
    action_id: str = Field(default_factory=lambda: new_research_id("action"))
    assertion_action_id: str
    passage_action_ids: List[str] = Field(min_length=1)
    basis: KnowledgeBasis
    rationale: str = Field(min_length=1)
    assessor: str = Field(min_length=1)
    uncertainty: str = Field(min_length=1)


ProposalAction = Annotated[
    Union[
        PassageAction,
        SourceAssertionAction,
        NormalizedClaimAction,
        MentionAction,
        RelationshipAction,
        KnowledgePositionAction,
    ],
    Field(discriminator="action_type"),
]


class ProposalBatch(BaseModel):
    actions: List[ProposalAction] = Field(min_length=1)
    tool_results: List[ToolResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def action_ids_are_unique(self) -> "ProposalBatch":
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("proposal action IDs must be unique within a batch")
        return self


class ObservationRequest(BaseModel):
    task_id: str
    workspace_revision_id: str
    representation_id: str
    source_unit_id: str
    candidate_context: Dict[str, Any]
    prior_human_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    available_tool_results: List[ToolResult] = Field(default_factory=list)
    evidence_scope: EvidenceScope = EvidenceScope.WORKSPACE
    out_of_workspace_reason: str | None = None
    model: str
    prompt_version: str
    task_version: str
    schema_version: str
    code_version: str
    observed_at: datetime

    @model_validator(mode="after")
    def explicit_scope_requires_reason(self) -> "ObservationRequest":
        if (
            self.evidence_scope is EvidenceScope.EXPLICIT_OUT_OF_WORKSPACE
            and not (self.out_of_workspace_reason or "").strip()
        ):
            raise ValueError("explicit out-of-workspace evidence requires a reason")
        if (
            self.evidence_scope is EvidenceScope.WORKSPACE
            and self.out_of_workspace_reason is not None
        ):
            raise ValueError(
                "workspace evidence cannot carry an out-of-workspace reason"
            )
        return self


class ObservationManifest(ObservationRequest):
    episode_id: str = Field(default_factory=lambda: new_research_id("episode"))
    accepted_state_before: Dict[str, Any]
    source_created_or_published_at: str | None = None

    @property
    def request_key(self) -> str:
        payload = self.model_dump(mode="json", exclude={"episode_id", "observed_at"})
        return sha256_jsonable({"namespace": "research_proposal", **payload})

    def attempt_key(self, attempt_number: int) -> str:
        return sha256_jsonable(
            {
                "request_key": self.request_key,
                "attempt": attempt_number,
                "namespace": "research_proposal_attempt",
            }
        )


class StoredProposalEpisode(BaseModel):
    episode_id: str
    status: Literal["running", "completed", "review_required", "failed"]
    acceptance_eligible: bool
    attempts: int
    batch: ProposalBatch | None = None
    validation_codes: List[str] = Field(default_factory=list)
    reused: bool = False
    transition: Dict[str, Any] | None = None
    replay_verified: bool = False


ProposalGenerator = Callable[[ObservationManifest, str | None], ProposalBatch]


def proposal_transition(
    manifest: ObservationManifest, batch: ProposalBatch
) -> Dict[str, Any]:
    """Apply primitive actions to an empty provisional proposal state.

    This function intentionally has no repository or filesystem access. Its complete
    input is the recorded observation, actions, and tool call results.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for action in batch.actions:
        grouped.setdefault(action.action_type, []).append(
            action.model_dump(mode="json")
        )
    return {
        "episode_id": manifest.episode_id,
        "task_id": manifest.task_id,
        "workspace_revision_id": manifest.workspace_revision_id,
        "evidence_scope": manifest.evidence_scope.value,
        "out_of_workspace_reason": manifest.out_of_workspace_reason,
        "representation_id": manifest.representation_id,
        "source_unit_id": manifest.source_unit_id,
        "accepted_state_before": manifest.accepted_state_before,
        "accepted_state_after": manifest.accepted_state_before,
        "proposal_state": "provisional",
        "actions_by_type": grouped,
        "tool_results": [item.model_dump(mode="json") for item in batch.tool_results],
        "versions": {
            "model": manifest.model,
            "prompt": manifest.prompt_version,
            "task": manifest.task_version,
            "schema": manifest.schema_version,
            "code": manifest.code_version,
        },
    }
