import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.research.extraction import (
    EvidenceScope,
    KnowledgeBasis,
    KnowledgePositionAction,
    NormalizedClaimAction,
    ObservationRequest,
    PassageAction,
    ProposalBatch,
    SourceAssertionAction,
    ToolResult,
)
from src.research.ingestion import CorpusIngestor
from src.research.locators import TextLocator
from src.research.service import ResearchService


def setup_evidence(tmp_path: Path):
    text = "In 2002, the witness said the allegation was reported by another guard."
    source = tmp_path / "source.txt"
    source.write_text(text)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        f"""schema_version: 1
corpus: {{id: release-zero, title: Release Zero, version: '1'}}
sources:
  - import_key: memoir
    title: Memoir
    source_type: memoir
    status: included
    metadata: {{publication_date: '2010-01-01'}}
    representations: [{{kind: original_text, path: source.txt, sha256: {digest}}}]
"""
    )
    service = ResearchService.open(tmp_path / "research.sqlite")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest_path)
    representation = service.list_representations("release-zero", "memoir")[0]
    source_unit_id = service.get_source_unit_id(representation.id, "document")
    workspace = service.ensure_workspace("release-zero", "Release Zero")
    revision = service.create_workspace_revision(
        workspace.id,
        memberships=[("source", representation.source_id, "included")],
    )
    return service, representation, source_unit_id, revision, text


def observation(service, representation, source_unit_id, revision):
    return service.build_observation_manifest(
        ObservationRequest(
            task_id="release-zero/assertions",
            workspace_revision_id=revision.id,
            representation_id=representation.id,
            source_unit_id=source_unit_id,
            candidate_context={"passage_candidates": [{"start": 0, "end": 7}]},
            prior_human_decisions=[],
            available_tool_results=[
                ToolResult(tool_name="candidate_search", result={"count": 1})
            ],
            model="local/test-model",
            prompt_version="passages-v1+assertions-v1",
            task_version="1",
            schema_version="1",
            code_version="git:test",
            observed_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        )
    )


def valid_batch(representation_id: str, text: str) -> ProposalBatch:
    passage = PassageAction(
        action_id="passage_1",
        locator=TextLocator(
            representation_id=representation_id,
            start=0,
            end=len(text),
        ),
        selected_quote=text,
    )
    assertion = SourceAssertionAction(
        action_id="assertion_1",
        passage_action_ids=[passage.action_id],
        assertion_text="The witness attributes an allegation to another guard.",
        attribution="The memoir's witness states",
        occurrence_time="2002",
        occurrence_time_precision="year",
        allegation=True,
        uncertainty="The other guard is not identified.",
    )
    claim = NormalizedClaimAction(
        action_id="claim_1",
        assertion_action_ids=[assertion.action_id],
        passage_action_ids=[passage.action_id],
        normalized_text="An unidentified guard reportedly made an allegation in 2002.",
        rationale="Preserves the source's attribution and year precision.",
        material_qualifiers=["reportedly", "unidentified"],
        allegation=True,
        attribution="The memoir's witness states",
    )
    position = KnowledgePositionAction(
        action_id="position_1",
        assertion_action_id=assertion.action_id,
        passage_action_ids=[passage.action_id],
        basis=KnowledgeBasis.ATTRIBUTED_REPORT,
        rationale="The witness says another guard supplied the allegation.",
        assessor="research-agent/test",
        uncertainty="The reporting chain cannot yet be independently checked.",
    )
    return ProposalBatch(
        actions=[passage, assertion, claim, position],
        tool_results=[
            ToolResult(
                tool_call_id="tool_call_locator",
                tool_name="locator_check",
                arguments={"representation_id": representation_id},
                result={"ok": True},
            )
        ],
    )


def test_complete_episode_is_atomic_provisional_and_replayable(tmp_path: Path):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    seen = []

    def generate(manifest, hint):
        seen.append((manifest.model_dump(mode="json"), hint))
        return valid_batch(representation.id, text)

    result = service.run_proposal_episode(
        observation(service, representation, unit_id, revision), generate
    )
    replay = service.replay_proposal_episode(result.episode_id)

    assert result.status == "completed"
    assert result.acceptance_eligible is True
    assert result.attempts == 1
    assert result.batch is not None
    assert all(action.action_type != "profile" for action in result.batch.actions)
    assert replay.model_dump(
        exclude={"reused", "replay_verified"}
    ) == result.model_dump(exclude={"reused", "replay_verified"})
    assert replay.replay_verified is True
    assert seen[0][0]["workspace_revision_id"] == revision.id

    with service.store.connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM research_episodes").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM proposal_actions").fetchone()[0]
            == 4
        )
        assert (
            connection.execute("SELECT state FROM proposal_batches").fetchone()[0]
            == "provisional"
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM record_versions").fetchone()[0]
            == 0
        )
        episode = connection.execute(
            "SELECT observation_manifest_json,accepted_state_after_json "
            "FROM research_episodes"
        ).fetchone()
        assert '"accepted_record_ids":[]' in episode["observation_manifest_json"]
        assert episode["accepted_state_after_json"] == '{"accepted_record_ids": []}'
        tool = connection.execute(
            "SELECT tool_call_id,arguments_json,result_json FROM proposal_tool_results"
        ).fetchone()
        assert tool["tool_call_id"] == "tool_call_locator"
        assert representation.id in tool["arguments_json"]
        assert '"ok": true' in tool["result_json"]


def test_one_repair_attempt_then_invalid_batch_remains_reviewable(tmp_path: Path):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    hints = []

    def generate(manifest, hint):
        hints.append(hint)
        batch = valid_batch(representation.id, text)
        passage = batch.actions[0]
        assert isinstance(passage, PassageAction)
        passage.selected_quote = "hallucinated quotation"
        return batch

    result = service.run_proposal_episode(
        observation(service, representation, unit_id, revision), generate
    )

    assert result.status == "review_required"
    assert result.acceptance_eligible is False
    assert result.attempts == 2
    assert hints[0] is None
    assert "quote_mismatch" in hints[1]
    with service.store.connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM proposal_batches").fetchone()[0]
            == 2
        )


@pytest.mark.parametrize(
    "record_type",
    ["attempt", "batch", "action", "tool_result", "validation"],
)
def test_persistence_failure_then_retry_has_no_partial_batch(
    tmp_path: Path, record_type: str
):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)
    fired = False

    def inject(kind):
        nonlocal fired
        if not fired and kind == record_type:
            fired = True
            raise RuntimeError(f"injected after {kind}")

    with pytest.raises(RuntimeError, match="injected"):
        service.run_proposal_episode(
            manifest,
            lambda observation, hint: valid_batch(representation.id, text),
            failure_injector=inject,
        )

    result = service.run_proposal_episode(
        manifest,
        lambda observation, hint: valid_batch(representation.id, text),
    )
    assert result.status == "completed"
    with service.store.connection() as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM proposal_batches").fetchone()[0]
            == 1
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM proposal_actions").fetchone()[0]
            == 4
        )
        failure = connection.execute(
            "SELECT attempt_number,failed_after_record_type "
            "FROM proposal_persistence_failures"
        ).fetchone()
        assert tuple(failure) == (1, record_type)


def test_identical_request_reuses_episode_but_version_change_does_not(tmp_path: Path):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)
    calls = 0

    def generate(observation, hint):
        nonlocal calls
        calls += 1
        return valid_batch(representation.id, text)

    first = service.run_proposal_episode(manifest, generate)
    second = service.run_proposal_episode(manifest, generate)
    changed = observation(service, representation, unit_id, revision).model_copy(
        update={"prompt_version": "passages-v2"}
    )
    third = service.run_proposal_episode(changed, generate)

    assert first.episode_id == second.episode_id
    assert second.reused is True
    assert third.episode_id != first.episode_id
    assert calls == 2


def test_generation_failure_is_reviewable_without_a_partial_batch(tmp_path: Path):
    service, representation, unit_id, revision, _text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)

    with pytest.raises(RuntimeError, match="model unavailable"):
        service.run_proposal_episode(
            manifest,
            lambda observation, hint: (_ for _ in ()).throw(
                RuntimeError("model unavailable")
            ),
        )

    result = service.repository.load_proposal_episode(manifest.episode_id)
    assert result.status == "running"
    assert result.attempts == 1
    assert result.batch is None
    with service.store.connection() as connection:
        error = connection.execute(
            "SELECT error_json FROM proposal_attempts WHERE episode_id=?",
            (manifest.episode_id,),
        ).fetchone()[0]
        assert "model unavailable" in error
        assert (
            connection.execute("SELECT COUNT(*) FROM proposal_batches").fetchone()[0]
            == 0
        )


def test_interrupted_repair_resumes_as_attempt_two_with_original_hint(tmp_path: Path):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)
    calls = 0

    def invalid_then_valid(observation, hint):
        nonlocal calls
        calls += 1
        batch = valid_batch(representation.id, text)
        if calls == 1:
            passage = batch.actions[0]
            assert isinstance(passage, PassageAction)
            passage.selected_quote = "not in source"
        return batch

    def interrupt_second_attempt(record_type):
        if calls == 2 and record_type == "attempt":
            raise RuntimeError("interrupted repair persistence")

    with pytest.raises(RuntimeError, match="interrupted repair"):
        service.run_proposal_episode(
            manifest,
            invalid_then_valid,
            failure_injector=interrupt_second_attempt,
        )

    seen_hint = None

    def resumed(observation, hint):
        nonlocal seen_hint
        seen_hint = hint
        return valid_batch(representation.id, text)

    result = service.run_proposal_episode(manifest, resumed)

    assert result.status == "completed"
    assert result.attempts == 2
    assert "quote_mismatch" in seen_hint


def test_proposal_history_cannot_be_updated_or_deleted(tmp_path: Path):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    result = service.run_proposal_episode(
        observation(service, representation, unit_id, revision),
        lambda manifest, hint: valid_batch(representation.id, text),
    )

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with service.store.transaction() as connection:
            connection.execute(
                "UPDATE proposal_batches SET acceptance_eligible=0 "
                "WHERE id=(SELECT id FROM proposal_batches LIMIT 1)"
            )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with service.store.transaction() as connection:
            connection.execute(
                "DELETE FROM research_episodes WHERE id=?", (result.episode_id,)
            )


def test_transient_generation_failure_can_resume_once_but_not_unbounded(
    tmp_path: Path,
):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)

    with pytest.raises(RuntimeError, match="transient"):
        service.run_proposal_episode(
            manifest,
            lambda observed, hint: (_ for _ in ()).throw(RuntimeError("transient")),
        )

    recovered = service.run_proposal_episode(
        manifest,
        lambda observed, hint: valid_batch(representation.id, text),
    )
    assert recovered.status == "completed"
    assert recovered.attempts == 2
    with service.store.connection() as connection:
        statuses = connection.execute(
            "SELECT status FROM proposal_attempts ORDER BY attempt_number"
        ).fetchall()
        assert [row["status"] for row in statuses] == [
            "generation_failed",
            "eligible",
        ]

    assert service.run_proposal_episode(
        manifest,
        lambda observed, hint: pytest.fail("bounded episode reran"),
    ).reused


def test_second_generation_failure_finalizes_bounded_episode(tmp_path: Path):
    service, representation, unit_id, revision, _text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)

    for expected_status in ("running", "failed"):
        with pytest.raises(RuntimeError, match="still unavailable"):
            service.run_proposal_episode(
                manifest,
                lambda observed, hint: (_ for _ in ()).throw(
                    RuntimeError("still unavailable")
                ),
            )
        assert service.repository.episode_status(manifest.episode_id) == expected_status

    reused = service.run_proposal_episode(
        manifest,
        lambda observed, hint: pytest.fail("third attempt must not run"),
    )
    assert reused.status == "failed"
    assert reused.attempts == 2
    assert reused.reused is True


def test_observation_provenance_is_derived_and_mismatches_are_rejected(
    tmp_path: Path,
):
    service, representation, unit_id, revision, _text = setup_evidence(tmp_path)
    manifest = observation(service, representation, unit_id, revision)

    assert manifest.source_created_or_published_at == "2010-01-01"
    assert manifest.accepted_state_before == {"accepted_record_ids": []}

    fictional_source_time = manifest.model_copy(
        update={"source_created_or_published_at": "1900-01-01"}
    )
    with pytest.raises(ValueError, match="stored source publication"):
        service.run_proposal_episode(
            fictional_source_time,
            lambda observed, hint: pytest.fail("generator must not run"),
        )

    fictional_state = manifest.model_copy(
        update={"accepted_state_before": {"accepted_record_ids": ["fiction"]}}
    )
    with pytest.raises(ValueError, match="accepted state"):
        service.run_proposal_episode(
            fictional_state,
            lambda observed, hint: pytest.fail("generator must not run"),
        )


def test_workspace_scope_requires_evidence_or_explicit_out_of_workspace_mode(
    tmp_path: Path,
):
    service, representation, unit_id, _revision, _text = setup_evidence(tmp_path)
    other = service.ensure_workspace("other", "Other")
    empty_revision = service.create_workspace_revision(other.id)
    request = ObservationRequest(
        task_id="release-zero/assertions",
        workspace_revision_id=empty_revision.id,
        representation_id=representation.id,
        source_unit_id=unit_id,
        candidate_context={},
        model="local/test-model",
        prompt_version="v1",
        task_version="1",
        schema_version="1",
        code_version="git:test",
        observed_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )

    with pytest.raises(ValueError, match="workspace revision does not include"):
        service.build_observation_manifest(request)

    explicit = service.build_observation_manifest(
        request.model_copy(
            update={
                "evidence_scope": EvidenceScope.EXPLICIT_OUT_OF_WORKSPACE,
                "out_of_workspace_reason": "Ad hoc source comparison",
            }
        )
    )
    assert explicit.evidence_scope is EvidenceScope.EXPLICIT_OUT_OF_WORKSPACE
    assert explicit.out_of_workspace_reason == "Ad hoc source comparison"


def test_replay_uses_only_stored_observation_and_actions(tmp_path: Path, monkeypatch):
    service, representation, unit_id, revision, text = setup_evidence(tmp_path)
    result = service.run_proposal_episode(
        observation(service, representation, unit_id, revision),
        lambda manifest, hint: valid_batch(representation.id, text),
    )

    monkeypatch.setattr(
        service,
        "retrieve_quote",
        lambda *args: pytest.fail("replay read hidden representation context"),
    )
    monkeypatch.setattr(
        service.repository,
        "get_revision",
        lambda *args: pytest.fail("replay read hidden workspace context"),
    )

    replayed = service.replay_proposal_episode(result.episode_id)
    assert replayed.replay_verified is True
    assert replayed.transition == result.transition
