"""Deterministic grounding checks for provisional research proposals."""

from __future__ import annotations

from typing import Callable, List

from pydantic import BaseModel

from src.research.extraction import (
    KnowledgePositionAction,
    NormalizedClaimAction,
    PassageAction,
    ProposalBatch,
    RelationshipAction,
    SourceAssertionAction,
)


class ValidationObservation(BaseModel):
    action_id: str | None
    code: str
    severity: str
    message: str
    deterministic: bool = True


def validate_proposal_batch(
    batch: ProposalBatch,
    retrieve_quote: Callable[[str, int, int], str],
    *,
    representation_id: str,
) -> List[ValidationObservation]:
    """Return appendable observations without changing the proposed actions."""
    observations: List[ValidationObservation] = []
    actions = {action.action_id: action for action in batch.actions}
    passages = {
        action.action_id: action
        for action in batch.actions
        if isinstance(action, PassageAction)
    }

    for passage in passages.values():
        if passage.locator.representation_id != representation_id:
            observations.append(
                _severe(
                    passage.action_id,
                    "wrong_representation",
                    "passage locator names a representation outside the observation",
                )
            )
            continue
        try:
            stored = retrieve_quote(
                passage.locator.representation_id,
                passage.locator.start,
                passage.locator.end,
            )
        except (KeyError, ValueError, OSError) as error:
            observations.append(
                _severe(passage.action_id, "invalid_locator", str(error))
            )
            continue
        if stored != passage.selected_quote:
            observations.append(
                _severe(
                    passage.action_id,
                    "quote_mismatch",
                    "selected quote does not equal text retrieved from the immutable representation",
                )
            )

    for action in batch.actions:
        if isinstance(action, PassageAction):
            continue
        passage_ids = getattr(action, "passage_action_ids", [])
        missing = [item for item in passage_ids if item not in passages]
        if missing:
            observations.append(
                _severe(
                    action.action_id,
                    "missing_passage_support",
                    f"proposal references absent passage actions: {', '.join(missing)}",
                )
            )
        if isinstance(action, NormalizedClaimAction):
            supporting_assertions = [
                actions.get(item) for item in action.assertion_action_ids
            ]
            missing_assertions = [
                item
                for item in action.assertion_action_ids
                if not isinstance(actions.get(item), SourceAssertionAction)
            ]
            if missing_assertions:
                observations.append(
                    _severe(
                        action.action_id,
                        "missing_assertion_support",
                        "normalized claim must reference source assertions from this batch",
                    )
                )
            allegation_support = any(
                isinstance(item, SourceAssertionAction) and item.allegation
                for item in supporting_assertions
            )
            if allegation_support and not action.allegation:
                observations.append(
                    _severe(
                        action.action_id,
                        "allegation_status_laundered",
                        "normalized claim must retain allegation status from its source assertions",
                    )
                )
            if allegation_support and not (action.attribution or "").strip():
                observations.append(
                    _severe(
                        action.action_id,
                        "unattributed_normalized_allegation",
                        "normalized allegation must retain explicit source attribution",
                    )
                )
        if isinstance(action, KnowledgePositionAction) and not isinstance(
            actions.get(action.assertion_action_id), SourceAssertionAction
        ):
            observations.append(
                _severe(
                    action.action_id,
                    "missing_assessed_assertion",
                    "knowledge-position assessment must name a source assertion",
                )
            )
        if isinstance(action, (SourceAssertionAction, RelationshipAction)):
            if action.allegation and not action.attribution.strip():
                observations.append(
                    _severe(
                        action.action_id,
                        "unattributed_allegation",
                        "a consequential allegation must retain explicit attribution",
                    )
                )

    if not observations:
        observations.append(
            ValidationObservation(
                action_id=None,
                code="deterministic_grounding_passed",
                severity="info",
                message="all locators and proposal references passed deterministic validation",
            )
        )
    return observations


def acceptance_eligible(observations: List[ValidationObservation]) -> bool:
    return not any(item.severity == "severe" for item in observations)


def repair_hint(observations: List[ValidationObservation]) -> str:
    severe = [item for item in observations if item.severity == "severe"]
    return "; ".join(f"{item.code}: {item.message}" for item in severe)


def _severe(action_id: str, code: str, message: str) -> ValidationObservation:
    return ValidationObservation(
        action_id=action_id,
        code=code,
        severity="severe",
        message=message,
    )
