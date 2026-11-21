import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.research.extraction import (
    ObservationRequest,
    PassageAction,
    ProposalBatch,
)
from src.research.ingestion import CorpusIngestor
from src.research.locators import TextLocator
from src.research.review import (
    CorrectnessJudgment,
    ProposalJudgment,
    ProposalReviewCommand,
    RelevanceJudgment,
    StaleReviewError,
    UnacceptableProposalError,
    WorkspaceInclusion,
)
from src.research.service import ResearchService


def prepared_proposal(
    tmp_path: Path, *, suffix: str = "first", invalid_quote: bool = False
):
    text = "The witness described an event in 2002."
    source = tmp_path / f"source-{suffix}.txt"
    source.write_text(text)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / f"manifest-{suffix}.yaml"
    manifest_path.write_text(
        f"""schema_version: 1
corpus: {{id: corpus-{suffix}, title: Corpus, version: '1'}}
sources:
  - import_key: memoir
    title: Memoir
    source_type: memoir
    status: included
    metadata: {{}}
    representations: [{{kind: original_text, path: {source.name}, sha256: {digest}}}]
"""
    )
    database = tmp_path / "research.sqlite"
    service = ResearchService.open(database)
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest_path)
    representation = service.list_representations(f"corpus-{suffix}", "memoir")[0]
    unit_id = service.get_source_unit_id(representation.id, "document")
    workspace = service.ensure_workspace("review", "Review")
    revision = service.create_workspace_revision(
        workspace.id,
        memberships=[("source", representation.source_id, "included")],
    )
    request = ObservationRequest(
        task_id="review/assertion",
        workspace_revision_id=revision.id,
        representation_id=representation.id,
        source_unit_id=unit_id,
        candidate_context={"source": suffix},
        model=f"local/{suffix}",
        prompt_version="review-v1",
        task_version="1",
        schema_version="1",
        code_version="test",
        observed_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    action = PassageAction(
        action_id=f"passage_{suffix}",
        locator=TextLocator(
            representation_id=representation.id,
            start=0,
            end=len(text),
        ),
        selected_quote="hallucinated quotation" if invalid_quote else text,
    )
    episode = service.run_proposal_episode(
        service.build_observation_manifest(request),
        lambda _manifest, _hint: ProposalBatch(actions=[action]),
    )
    return service, episode.episode_id, action


@pytest.mark.parametrize(
    ("judgment", "edited_payload", "record_expected"),
    [
        (ProposalJudgment.ACCEPT, None, True),
        (ProposalJudgment.EDIT, {"context_before": "Editorial context"}, True),
        (ProposalJudgment.REJECT, None, False),
        (ProposalJudgment.DEFER, None, False),
    ],
)
def test_review_outcomes_preserve_proposal_and_reason(
    tmp_path: Path,
    judgment: ProposalJudgment,
    edited_payload,
    record_expected: bool,
) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    result = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=judgment,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.NOT_DECIDED,
            reason="Reviewed against the original page.",
            edited_payload=edited_payload,
        )
    )

    assert result.version == 1
    assert result.reason == "Reviewed against the original page."
    assert (result.record_version_id is not None) is record_expected
    exported = service.export_proposal_review(episode_id, action.action_id)
    assert exported["model_proposal"] == action.model_dump(mode="json")
    assert exported["historian_decisions"][0]["reason"] == result.reason
    if judgment is ProposalJudgment.EDIT:
        assert exported["human_edit"] == edited_payload
        assert exported["human_edit"] != exported["model_proposal"]
    else:
        assert exported["human_edit"] is None


def test_correct_irrelevant_and_workspace_exclusion_are_independent(tmp_path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    result = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.IRRELEVANT,
            inclusion=WorkspaceInclusion.EXCLUDED,
            reason="Accurate, but outside this timeline's scope.",
        )
    )

    assert result.correctness is CorrectnessJudgment.CORRECT
    assert result.relevance is RelevanceJudgment.IRRELEVANT
    assert result.inclusion is WorkspaceInclusion.EXCLUDED
    assert result.record_version_id is not None


def test_accept_only_include_later_and_exclude_are_distinct_events(tmp_path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    accepted = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.NOT_DECIDED,
        )
    )
    included = service.set_review_inclusion(
        episode_id,
        action.action_id,
        expected_version=accepted.version,
        inclusion=WorkspaceInclusion.INCLUDED,
        reason="Needed on the evidence timeline.",
    )
    excluded = service.set_review_inclusion(
        episode_id,
        action.action_id,
        expected_version=included.version,
        inclusion=WorkspaceInclusion.EXCLUDED,
        reason="Removed from this workspace, not rejected as false.",
    )

    assert [accepted.inclusion, included.inclusion, excluded.inclusion] == [
        WorkspaceInclusion.NOT_DECIDED,
        WorkspaceInclusion.INCLUDED,
        WorkspaceInclusion.EXCLUDED,
    ]
    export = service.export_proposal_review(episode_id, action.action_id)
    assert [item["inclusion"] for item in export["inclusion_history"]] == [
        "not_decided",
        "included",
        "excluded",
    ]
    assert export["historian_decisions"][0]["judgment"] == "accept"


def test_concurrent_reviews_cannot_select_two_current_versions(tmp_path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    barrier = threading.Barrier(2)
    outcomes = []

    def submit(label: str) -> None:
        contender = ResearchService.open(service.store.path)
        barrier.wait()
        try:
            result = contender.review_proposal(
                ProposalReviewCommand(
                    episode_id=episode_id,
                    action_id=action.action_id,
                    expected_version=0,
                    judgment=ProposalJudgment.EDIT,
                    correctness=CorrectnessJudgment.CORRECT,
                    relevance=RelevanceJudgment.RELEVANT,
                    inclusion=WorkspaceInclusion.INCLUDED,
                    edited_payload={"context_before": label},
                )
            )
            outcomes.append(("accepted", result.record_version_id))
        except StaleReviewError:
            outcomes.append(("stale", None))

    threads = [threading.Thread(target=submit, args=(label,)) for label in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(item[0] for item in outcomes) == ["accepted", "stale"]
    with service.store.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM record_versions WHERE is_current=1"
            ).fetchone()[0]
            == 1
        )


def test_invalid_grounding_cannot_be_accepted_or_edited_around(tmp_path: Path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path, invalid_quote=True)

    with pytest.raises(UnacceptableProposalError, match="deterministic validation"):
        service.review_proposal(
            ProposalReviewCommand(
                episode_id=episode_id,
                action_id=action.action_id,
                expected_version=0,
                judgment=ProposalJudgment.ACCEPT,
                correctness=CorrectnessJudgment.CORRECT,
                relevance=RelevanceJudgment.RELEVANT,
            )
        )
    with pytest.raises(UnacceptableProposalError, match="quote_mismatch"):
        service.review_proposal(
            ProposalReviewCommand(
                episode_id=episode_id,
                action_id=action.action_id,
                expected_version=0,
                judgment=ProposalJudgment.EDIT,
                correctness=CorrectnessJudgment.CORRECT,
                relevance=RelevanceJudgment.RELEVANT,
                edited_payload={"context_before": "Does not repair the quotation."},
            )
        )


def test_human_edit_can_repair_grounding_and_preserves_complete_action(
    tmp_path: Path,
) -> None:
    service, episode_id, action = prepared_proposal(tmp_path, invalid_quote=True)
    source_text = service.retrieve_quote(
        action.locator.representation_id,
        action.locator.start,
        action.locator.end,
    )

    repaired = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.EDIT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            edited_payload={"selected_quote": source_text},
            reason="Transcribed directly from the immutable representation.",
        )
    )

    record = service.get_current_record(repaired.logical_record_id)
    assert record.payload["selected_quote"] == source_text
    assert record.payload["locator"] == action.locator.model_dump(mode="json")


def test_review_history_rejects_nonhuman_writes_and_mutation(tmp_path: Path) -> None:
    service, episode_id, action = prepared_proposal(tmp_path)
    reviewed = service.review_proposal(
        ProposalReviewCommand(
            episode_id=episode_id,
            action_id=action.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.RELEVANT,
        )
    )
    with service.store.transaction() as connection:
        connection.execute(
            "INSERT INTO audit_actors(id,actor_type,display_name) VALUES(?,?,?)",
            ("review-model", "model", "Review model"),
        )
    with service.store.connection() as connection:
        inclusion_id = connection.execute(
            "SELECT id FROM proposal_workspace_decisions WHERE subject_id=?",
            (reviewed.subject_id,),
        ).fetchone()[0]

    with pytest.raises(sqlite3.IntegrityError, match="human actor"):
        with service.store.transaction() as connection:
            connection.execute(
                "INSERT INTO proposal_review_decisions("
                "id,subject_id,subject_version,judgment,correctness,relevance,"
                "actor_id,evaluation_eligible,training_eligible,created_at) "
                "VALUES('model-review',?,2,'reject','incorrect','relevant',"
                "'review-model',1,0,CURRENT_TIMESTAMP)",
                (reviewed.subject_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="human actor"):
        with service.store.transaction() as connection:
            connection.execute(
                "INSERT INTO proposal_workspace_decisions("
                "id,subject_id,subject_version,inclusion,actor_id,created_at) "
                "VALUES('model-inclusion',?,2,'excluded','review-model',CURRENT_TIMESTAMP)",
                (reviewed.subject_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with service.store.transaction() as connection:
            connection.execute(
                "UPDATE proposal_review_decisions SET reason='changed' WHERE id=?",
                (reviewed.decision_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with service.store.transaction() as connection:
            connection.execute(
                "DELETE FROM proposal_workspace_decisions WHERE id=?",
                (inclusion_id,),
            )
