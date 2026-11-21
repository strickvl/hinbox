"""Synthetic Release 0 checks that do not claim a research advantage."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.research.extraction import (
    ObservationRequest,
    PassageAction,
    ProposalBatch,
    SourceAssertionAction,
)
from src.research.ingestion import CorpusIngestor
from src.research.locators import TextLocator
from src.research.review import (
    CorrectnessJudgment,
    ProposalJudgment,
    ProposalReviewCommand,
    RelevanceJudgment,
    WorkspaceInclusion,
)
from src.research.service import ResearchService

FIXTURE_ROOT = Path("tests/fixtures/research/release_zero")
MANIFEST = FIXTURE_ROOT / "manifest.yaml"
FROZEN_TASKS = (
    "disputed chronology",
    "attribution and correction",
    "coverage and omission",
    "source recovery",
)


@dataclass(frozen=True)
class EvaluationInput:
    condition: str
    ordered_checksums: tuple[str, ...]
    tasks: tuple[str, ...]


def _fixture_inventory() -> tuple[str, ...]:
    manifest = yaml.safe_load(MANIFEST.read_text())
    sources = {item["import_key"]: item for item in manifest["sources"]}
    checksums = []
    for import_key in manifest["corpus"]["frozen_order"]:
        for representation in sources[import_key]["representations"]:
            path = FIXTURE_ROOT / representation["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == representation["sha256"]
            checksums.append(digest)
    return tuple(checksums)


def _prepare_episode(tmp_path: Path):
    source = FIXTURE_ROOT / "sources/page-01.corrected.txt"
    raw = source.read_bytes()
    local_source = tmp_path / source.name
    local_source.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"""schema_version: 1
corpus: {{id: release-zero-replay, title: Release Zero Replay, version: '1'}}
sources:
  - import_key: corrected-page
    title: Corrected synthetic page
    source_type: text
    status: included
    representations:
      - kind: original_text
        path: {local_source.name}
        sha256: {digest}
        review_state: author_verified
"""
    )
    service = ResearchService.open(tmp_path / "research.sqlite")
    CorpusIngestor(service, tmp_path / "managed").ingest_manifest(manifest)
    representation = service.list_representations(
        "release-zero-replay", "corrected-page"
    )[0]
    source_unit_id = service.get_source_unit_id(representation.id, "document")
    workspace = service.ensure_workspace("release-zero", "Release Zero")
    revision = service.create_workspace_revision(
        workspace.id,
        memberships=[("source", representation.source_id, "included")],
    )
    text = raw.decode("utf-8")
    quote = "According to counsel, the fictional detainee\nwas released on 7 June"
    start = text.index(quote)
    passage = PassageAction(
        action_id="passage_corrected_date",
        locator=TextLocator(
            representation_id=representation.id,
            start=start,
            end=start + len(quote),
        ),
        selected_quote=quote,
    )
    assertion = SourceAssertionAction(
        action_id="assertion_corrected_date",
        passage_action_ids=[passage.action_id],
        assertion_text="Counsel is attributed with a 7 June release date.",
        attribution="According to counsel",
        occurrence_time_precision="unknown",
        uncertainty="The fixture preserves attribution rather than treating it as direct observation.",
    )
    request = ObservationRequest(
        task_id="release-zero/attribution-and-correction",
        workspace_revision_id=revision.id,
        representation_id=representation.id,
        source_unit_id=source_unit_id,
        candidate_context={"frozen_tasks": list(FROZEN_TASKS)},
        model="deterministic/test-fixture",
        prompt_version="none-synthetic-v1",
        task_version="2026-07-15.1",
        schema_version="1",
        code_version="test",
        observed_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
    )
    observation = service.build_observation_manifest(request)
    episode = service.run_proposal_episode(
        observation,
        lambda _manifest, _hint: ProposalBatch(actions=[passage, assertion]),
    )
    return service, episode, passage, assertion, observation


def test_all_conditions_receive_identical_fixture_bytes_and_tasks() -> None:
    inventory = _fixture_inventory()
    inputs = [
        EvaluationInput(condition, inventory, FROZEN_TASKS)
        for condition in ("raw-folder", "prose", "ledger")
    ]

    assert len({item.ordered_checksums for item in inputs}) == 1
    assert len({item.tasks for item in inputs}) == 1


def test_every_ledger_assertion_quote_round_trips_to_fixture_bytes(tmp_path: Path):
    service, episode, passage, assertion, _observation = _prepare_episode(tmp_path)

    stored_assertions = [
        action
        for action in episode.batch.actions
        if isinstance(action, SourceAssertionAction)
    ]
    passages = {
        action.action_id: action
        for action in episode.batch.actions
        if isinstance(action, PassageAction)
    }
    assert stored_assertions
    for stored_assertion in stored_assertions:
        for passage_id in stored_assertion.passage_action_ids:
            stored_passage = passages[passage_id]
            assert service.retrieve_passage(stored_passage.locator).quotation == (
                stored_passage.selected_quote
            )
    assert assertion.passage_action_ids == [passage.action_id]


def test_correction_and_material_qualifier_survive_identical_rerun(tmp_path: Path):
    service, episode, _passage, assertion, observation = _prepare_episode(tmp_path)

    replayed = service.replay_proposal_episode(episode.episode_id)
    rerun = service.run_proposal_episode(
        observation,
        lambda *_args: (_ for _ in ()).throw(AssertionError("generator reran")),
    )
    replayed_assertion = next(
        action
        for action in replayed.batch.actions
        if isinstance(action, SourceAssertionAction)
    )

    assert replayed.replay_verified is True
    assert rerun.reused is True
    assert replayed_assertion.assertion_text == assertion.assertion_text
    assert "7 June" in replayed_assertion.assertion_text
    assert replayed_assertion.attribution == "According to counsel"
    assert replayed_assertion.uncertainty == assertion.uncertainty


def test_correctness_relevance_inclusion_and_rejection_are_distinct_states(
    tmp_path: Path,
) -> None:
    service, episode, _passage, assertion, _observation = _prepare_episode(tmp_path)
    commands = (
        ProposalReviewCommand(
            episode_id=episode.episode_id,
            action_id=assertion.action_id,
            expected_version=0,
            judgment=ProposalJudgment.ACCEPT,
            correctness=CorrectnessJudgment.CORRECT,
            relevance=RelevanceJudgment.IRRELEVANT,
            inclusion=WorkspaceInclusion.EXCLUDED,
            reason="Correct extraction, outside this task.",
        ),
        ProposalReviewCommand(
            episode_id=episode.episode_id,
            action_id=assertion.action_id,
            expected_version=1,
            judgment=ProposalJudgment.DEFER,
            correctness=CorrectnessJudgment.UNCERTAIN,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.NOT_DECIDED,
            reason="Needs source comparison.",
        ),
        ProposalReviewCommand(
            episode_id=episode.episode_id,
            action_id=assertion.action_id,
            expected_version=2,
            judgment=ProposalJudgment.REJECT,
            correctness=CorrectnessJudgment.INCORRECT,
            relevance=RelevanceJudgment.RELEVANT,
            inclusion=WorkspaceInclusion.EXCLUDED,
            reason="Rejected as wrong after review.",
        ),
    )

    states = []
    for command in commands:
        result = service.review_proposal(command)
        states.append(
            (
                result.judgment.value,
                result.correctness.value,
                result.relevance.value,
                result.inclusion.value,
            )
        )

    assert states == [
        ("accept", "correct", "irrelevant", "excluded"),
        ("defer", "uncertain", "relevant", "not_decided"),
        ("reject", "incorrect", "relevant", "excluded"),
    ]
    assert len(set(states)) == 3
