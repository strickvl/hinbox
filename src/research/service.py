"""Shared application boundary for research operations."""

from pathlib import Path
from typing import Any, Callable, Dict, List

from src.research.citations import project_citation
from src.research.extraction import (
    ObservationManifest,
    ObservationRequest,
    ProposalGenerator,
    StoredProposalEpisode,
    proposal_transition,
)
from src.research.locators import RetrievedPassage, TextLocator
from src.research.models import (
    Anchor,
    AnchorCreate,
    RecordVersion,
    Representation,
)
from src.research.repository import ResearchRepository
from src.research.review import (
    ProposalReviewCommand,
    ProposalReviewResult,
    WorkspaceInclusion,
)
from src.research.review_repository import ReviewRepository
from src.research.store import ResearchStore
from src.research.validation import (
    acceptance_eligible,
    repair_hint,
    validate_proposal_batch,
)
from src.research.workspaces import Membership, Workspace, WorkspaceRevision


class ResearchService:
    def __init__(self, store: ResearchStore) -> None:
        self.store = store
        self.repository = ResearchRepository(store)
        self.review_repository = ReviewRepository(store, self.retrieve_quote)

    def research_counts(self) -> Dict[str, int]:
        counts = self.repository.counts()
        counts["blobs"] = counts.pop("representation_blobs")
        return counts

    def register_manifest_records(
        self,
        corpus: Dict[str, Any],
        manifest_path: Path,
        manifest_checksum: str,
        snapshot: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        return self.repository.upsert_corpus_and_sources(
            corpus, manifest_path, manifest_checksum, snapshot, sources
        )

    def has_representation(self, source_id: str, kind: str, checksum: str) -> bool:
        return self.repository.representation_exists(source_id, kind, checksum)

    def register_representation(
        self,
        *,
        source_id: str,
        kind: str,
        checksum: str,
        byte_length: int,
        character_length: int,
        managed_path: Path,
        review_state: str,
        metadata: Dict[str, Any],
    ) -> str:
        return self.repository.insert_representation(
            source_id=source_id,
            kind=kind,
            checksum=checksum,
            byte_length=byte_length,
            character_length=character_length,
            managed_path=managed_path,
            review_state=review_state,
            metadata=metadata,
        )

    def list_representations(
        self, corpus_id: str, import_key: str
    ) -> List[Representation]:
        return [
            Representation(**dict(row))
            for row in self.repository.list_representations(corpus_id, import_key)
        ]

    def retrieve_quote(self, representation_id: str, start: int, end: int) -> str:
        import hashlib

        row = self.repository.get_representation(representation_id)
        path = Path(row["managed_path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["checksum"]:
            raise ValueError("representation checksum mismatch")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("representation is not UTF-8 text") from error
        if start < 0 or end <= start or end > len(text):
            raise ValueError("locator bounds do not fit representation")
        return text[start:end]

    def retrieve_passage(self, locator: TextLocator) -> RetrievedPassage:
        row = self.repository.get_representation(locator.representation_id)
        quotation = self.retrieve_quote(
            locator.representation_id, locator.start, locator.end
        )
        return RetrievedPassage(
            locator=locator,
            quotation=quotation,
            textual_form=row["textual_form"],
            checksum=row["checksum"],
        )

    def get_source_unit_id(self, representation_id: str, unit_key: str) -> str:
        return self.repository.get_source_unit_id(representation_id, unit_key)

    def build_observation_manifest(
        self, request: ObservationRequest
    ) -> ObservationManifest:
        """Bind caller task inputs to provenance derived from stored research state."""
        return self.repository.derive_observation_manifest(request)

    def run_proposal_episode(
        self,
        manifest: ObservationManifest,
        generator: ProposalGenerator,
        *,
        failure_injector: Callable[[str], None] | None = None,
    ) -> StoredProposalEpisode:
        """Generate at most one repair and append only provisional research."""
        episode_id = self.repository.ensure_research_episode(manifest)
        existing_status = self.repository.episode_status(episode_id)
        if existing_status != "running":
            existing = self.repository.load_proposal_episode(episode_id)
            return existing.model_copy(update={"reused": True})

        attempt_number = self.repository.proposal_attempt_count(episode_id) + 1
        hint: str | None = None
        if attempt_number > 1:
            hint = repair_hint(
                self.repository.latest_validation_observations(episode_id)
            )
        while attempt_number <= 2:
            try:
                batch = generator(manifest, hint)
            except Exception as error:
                self.repository.persist_generation_failure(
                    episode_id=episode_id,
                    manifest=manifest,
                    attempt_number=attempt_number,
                    error=error,
                    repair_hint=hint,
                    final=attempt_number == 2,
                )
                raise
            observations = validate_proposal_batch(
                batch,
                self.retrieve_quote,
                representation_id=manifest.representation_id,
            )
            eligible = acceptance_eligible(observations)
            final = eligible or attempt_number == 2
            failed_after = "attempt"

            def inject(record_type: str) -> None:
                nonlocal failed_after
                failed_after = record_type
                if failure_injector is not None:
                    failure_injector(record_type)

            try:
                self.repository.persist_proposal_attempt(
                    episode_id=episode_id,
                    manifest=manifest,
                    attempt_number=attempt_number,
                    batch=batch,
                    observations=observations,
                    repair_hint=hint,
                    eligible=eligible,
                    final=final,
                    failure_injector=inject,
                )
            except Exception as error:
                self.repository.persist_persistence_failure(
                    episode_id=episode_id,
                    attempt_number=attempt_number,
                    failed_after_record_type=failed_after,
                    error=error,
                )
                raise
            if final:
                return self.repository.load_proposal_episode(episode_id)
            hint = repair_hint(observations)
            attempt_number += 1
        raise AssertionError("proposal attempt limit was not finalized")

    def replay_proposal_episode(self, episode_id: str) -> StoredProposalEpisode:
        """Rebuild and compare the transition without consulting hidden context."""
        stored = self.repository.load_proposal_episode(episode_id)
        manifest = self.repository.load_episode_manifest(episode_id)
        if stored.batch is None:
            return stored
        replayed = proposal_transition(manifest, stored.batch)
        if replayed != stored.transition:
            raise ValueError("stored proposal transition does not replay exactly")
        return stored.model_copy(update={"replay_verified": True})

    def review_proposal(self, command: ProposalReviewCommand) -> ProposalReviewResult:
        return self.review_repository.review_proposal(command)

    def set_review_inclusion(
        self,
        episode_id: str,
        action_id: str,
        *,
        expected_version: int,
        inclusion: WorkspaceInclusion,
        reason: str | None = None,
        actor_id: str = "historian",
    ) -> ProposalReviewResult:
        return self.review_repository.set_inclusion(
            episode_id,
            action_id,
            expected_version=expected_version,
            inclusion=inclusion,
            reason=reason,
            actor_id=actor_id,
        )

    def export_proposal_review(self, episode_id: str, action_id: str) -> Dict[str, Any]:
        return self.review_repository.export_review(episode_id, action_id)

    def ensure_workspace(self, key: str, title: str) -> Workspace:
        return self.repository.ensure_workspace(key, title)

    def create_workspace_revision(
        self,
        workspace_id: str,
        memberships: List[Membership] | None = None,
        reviewed_context: Dict[str, Any] | None = None,
    ) -> WorkspaceRevision:
        return self.repository.create_revision(
            workspace_id, memberships or [], reviewed_context or {}
        )

    def get_workspace_revision(self, revision_id: str) -> WorkspaceRevision:
        return self.repository.get_revision(revision_id)

    def project_source_citation(
        self, corpus_id: str, import_key: str, *, locator: TextLocator
    ) -> Dict[str, Any]:
        metadata = self.repository.get_source_metadata_for_representation(
            locator.representation_id, corpus_id, import_key
        )
        projected = project_citation(metadata, locator=locator.model_dump())
        projected["quotation"] = self.retrieve_passage(locator).quotation
        return projected

    def get_source_metadata(self, corpus_id: str, import_key: str) -> Dict[str, Any]:
        return self.repository.get_source_metadata(corpus_id, import_key)

    def get_corpus_metadata(self, corpus_id: str) -> Dict[str, Any]:
        return self.repository.get_corpus_metadata(corpus_id)

    def manifest_import_count(self, corpus_id: str) -> int:
        return self.repository.manifest_import_count(corpus_id)

    def referenced_blob_checksums(self) -> set[str]:
        return self.repository.referenced_blob_checksums()

    @classmethod
    def open(cls, database_path: str | Path) -> "ResearchService":
        store = ResearchStore(database_path)
        store.initialize()
        return cls(store)

    def create_anchor(self, candidate: AnchorCreate) -> Anchor:
        return self.repository.create_anchor(candidate)

    def get_anchor(self, anchor_id: str) -> Anchor:
        return self.repository.get_anchor(anchor_id)

    def create_record_version(
        self, *, kind: str, payload: Dict[str, Any], actor_id: str
    ) -> RecordVersion:
        return self.repository.create_record_version(
            kind=kind, payload=payload, actor_id=actor_id
        )

    def supersede_record_version(
        self, previous_id: str, *, payload: Dict[str, Any], actor_id: str
    ) -> RecordVersion:
        return self.repository.supersede_record_version(
            previous_id, payload=payload, actor_id=actor_id
        )

    def get_record_history(self, logical_record_id: str) -> List[RecordVersion]:
        return self.repository.get_record_history(logical_record_id)

    def get_current_record(self, logical_record_id: str) -> RecordVersion:
        return self.repository.get_current_record(logical_record_id)
