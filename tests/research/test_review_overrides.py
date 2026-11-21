import sqlite3
from pathlib import Path

import pytest

from src.research.review import (
    CorrectnessJudgment,
    ProposalJudgment,
    ProposalReviewCommand,
    RelevanceJudgment,
    WorkspaceInclusion,
)
from tests.research.test_review_workflow import prepared_proposal


def test_later_model_disagreement_does_not_restore_human_rejection(
    tmp_path: Path,
) -> None:
    service, first_episode, first_action = prepared_proposal(tmp_path, suffix="first")
    accepted = service.review_proposal(
        ProposalReviewCommand(
            episode_id=first_episode,
            action_id=first_action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.EDIT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.INCLUDED,
            edited_payload={"context_before": "Historian-corrected context"},
            reason="Corrected after inspecting the image.",
        )
    )
    assert accepted.record_version_id is not None

    service, later_episode, later_action = prepared_proposal(tmp_path, suffix="later")
    rejected = service.review_proposal(
        ProposalReviewCommand(
            episode_id=later_episode,
            action_id=later_action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.REJECT,
            correctness=CorrectnessJudgment.INCORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.EXCLUDED,
            reason="Later model repeats the rejected reading.",
        )
    )

    assert rejected.record_version_id is None
    current = service.get_current_record(accepted.logical_record_id)
    assert current.id == accepted.record_version_id
    assert current.payload["context_before"] == "Historian-corrected context"
    assert current.payload["selected_quote"] == first_action.selected_quote


def test_reject_then_later_include_does_not_change_correctness(tmp_path: Path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    rejected = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.REJECT,
            correctness=CorrectnessJudgment.INCORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.EXCLUDED,
        )
    )
    changed = service.set_review_inclusion(
        episode_id,
        action.action_id,
        expected_version=rejected.version,
        inclusion=WorkspaceInclusion.INCLUDED,
        reason="Included as evidence of what the source says, not as accepted fact.",
    )

    assert changed.correctness is CorrectnessJudgment.INCORRECT
    assert changed.judgment is ProposalJudgment.REJECT
    assert changed.inclusion is WorkspaceInclusion.INCLUDED


def test_historian_can_reverse_acceptance_without_erasing_history(
    tmp_path: Path,
) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    accepted = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
        )
    )
    rejected = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=accepted.version,
            judgment=ProposalJudgment.REJECT,
            correctness=CorrectnessJudgment.INCORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            reason="A closer reading changed the assessment.",
        )
    )

    assert rejected.version == 2
    assert rejected.record_version_id is None
    with pytest.raises(KeyError):
        service.get_current_record(accepted.logical_record_id)
    history = service.get_record_history(accepted.logical_record_id)
    assert len(history) == 1
    assert history[0].status == "superseded"
    exported = service.export_proposal_review(episode_id, action.action_id)
    assert [item["judgment"] for item in exported["historian_decisions"]] == [
        "accept",
        "reject",
    ]


def test_reaccept_after_rejection_links_to_withdrawn_version(tmp_path: Path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    accepted = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
        )
    )
    rejected = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=accepted.version,
            judgment=ProposalJudgment.REJECT,
            correctness=CorrectnessJudgment.INCORRECT,
            relevance=RelevanceJudgment.RELEVANT,
        )
    )
    reaccepted = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=rejected.version,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
        )
    )

    history = service.get_record_history(accepted.logical_record_id)
    assert reaccepted.logical_record_id == accepted.logical_record_id
    assert history[-1].supersedes_id == accepted.record_version_id
    assert history[-1].id == reaccepted.record_version_id


def test_defer_leaves_previously_accepted_evidence_selected(tmp_path: Path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    accepted = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
        )
    )
    deferred = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=accepted.version,
            judgment=ProposalJudgment.DEFER,
            correctness=CorrectnessJudgment.UNCERTAIN,
            relevance=RelevanceJudgment.RELEVANT,
            reason="Needs another source, but does not withdraw the accepted reading.",
        )
    )

    assert deferred.record_version_id == accepted.record_version_id
    assert service.get_current_record(accepted.logical_record_id).id == (
        accepted.record_version_id
    )


def test_model_actor_cannot_create_accepted_evidence(tmp_path: Path) -> None:
    service, _episode_id, _action = prepared_proposal(tmp_path)
    with service.store.transaction() as connection:
        connection.execute(
            "INSERT INTO audit_actors(id,actor_type,display_name) VALUES(?,?,?)",
            ("later-model", "model", "Later model"),
        )

    with pytest.raises(sqlite3.IntegrityError, match="human actor"):
        service.create_record_version(
            kind="source_assertion",
            payload={"assertion_text": "Automated overwrite"},
            actor_id="later-model",
        )
