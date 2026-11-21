"""All SQL operations for source evidence and workspaces."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from src.research.extraction import (
    EvidenceScope,
    ObservationManifest,
    ObservationRequest,
    ProposalBatch,
    StoredProposalEpisode,
    proposal_transition,
)
from src.research.models import (
    Anchor,
    AnchorCreate,
    RecordVersion,
    new_research_id,
    utc_now,
)
from src.research.validation import ValidationObservation
from src.research.workspaces import Workspace, WorkspaceRevision, revision_hash


class ResearchRepository:
    def __init__(self, store) -> None:
        self.store = store

    def create_anchor(self, candidate: AnchorCreate) -> Anchor:
        created_at = utc_now()
        with self.store.transaction() as connection:
            connection.execute(
                "INSERT INTO anchors(id, kind, display_name, created_at) VALUES (?, ?, ?, ?)",
                (
                    candidate.id,
                    candidate.kind.value,
                    candidate.display_name,
                    created_at.isoformat(),
                ),
            )
            for alias in candidate.aliases:
                connection.execute(
                    "INSERT INTO anchor_aliases(id, anchor_id, alias) VALUES (?, ?, ?)",
                    (new_research_id("alias"), candidate.id, alias),
                )
        return Anchor(**candidate.model_dump(), created_at=created_at)

    def get_anchor(self, anchor_id: str) -> Anchor:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT * FROM anchors WHERE id = ?", (anchor_id,)
            ).fetchone()
            if row is None:
                raise KeyError(anchor_id)
            aliases = connection.execute(
                "SELECT alias FROM anchor_aliases WHERE anchor_id = ? ORDER BY rowid",
                (anchor_id,),
            ).fetchall()
        return Anchor(
            id=row["id"],
            kind=row["kind"],
            display_name=row["display_name"],
            aliases=[alias["alias"] for alias in aliases],
            created_at=row["created_at"],
        )

    def create_record_version(
        self, *, kind: str, payload: Dict[str, Any], actor_id: str
    ) -> RecordVersion:
        record = self._new_record_version(
            new_research_id("record"), kind, payload, actor_id, None
        )
        with self.store.transaction() as connection:
            self._write_version(connection, record)
        return record

    def supersede_record_version(
        self, previous_id: str, *, payload: Dict[str, Any], actor_id: str
    ) -> RecordVersion:
        with self.store.transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM record_versions WHERE id = ? AND is_current = 1",
                (previous_id,),
            ).fetchone()
            if previous is None:
                raise KeyError(previous_id)
            connection.execute(
                "UPDATE record_versions SET status = 'superseded', is_current = 0 WHERE id = ?",
                (previous_id,),
            )
            record = self._new_record_version(
                previous["logical_record_id"],
                previous["kind"],
                payload,
                actor_id,
                previous_id,
            )
            self._write_version(connection, record)
        return record

    @staticmethod
    def _new_record_version(
        logical_id: str,
        kind: str,
        payload: Dict[str, Any],
        actor_id: str,
        supersedes_id: str | None,
    ) -> RecordVersion:
        return RecordVersion(
            id=new_research_id("version"),
            logical_record_id=logical_id,
            kind=kind,
            payload=payload,
            supersedes_id=supersedes_id,
            actor_id=actor_id,
            status="accepted",
            is_current=True,
            created_at=utc_now(),
        )

    @staticmethod
    def _write_version(connection, record: RecordVersion) -> None:
        connection.execute(
            "INSERT INTO record_versions(id, logical_record_id, kind, payload_json, "
            "supersedes_id, actor_id, status, is_current, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.logical_record_id,
                record.kind,
                json.dumps(record.payload, sort_keys=True),
                record.supersedes_id,
                record.actor_id,
                record.status,
                int(record.is_current),
                record.created_at.isoformat(),
            ),
        )

    def get_record_history(self, logical_record_id: str) -> List[RecordVersion]:
        with self.store.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM record_versions WHERE logical_record_id = ? "
                "ORDER BY created_at, rowid",
                (logical_record_id,),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get_current_record(self, logical_record_id: str) -> RecordVersion:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT * FROM record_versions WHERE logical_record_id = ? AND is_current = 1",
                (logical_record_id,),
            ).fetchone()
        if row is None:
            raise KeyError(logical_record_id)
        return self._record_from_row(row)

    @staticmethod
    def _record_from_row(row) -> RecordVersion:
        return RecordVersion(
            id=row["id"],
            logical_record_id=row["logical_record_id"],
            kind=row["kind"],
            payload=json.loads(row["payload_json"]),
            supersedes_id=row["supersedes_id"],
            actor_id=row["actor_id"],
            status=row["status"],
            is_current=bool(row["is_current"]),
            created_at=row["created_at"],
        )

    def upsert_corpus_and_sources(
        self,
        corpus: Dict[str, Any],
        manifest_path: Path,
        manifest_checksum: str,
        snapshot: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        now = utc_now().isoformat()
        with self.store.transaction() as connection:
            connection.execute(
                "INSERT INTO corpora(id,title,version,manifest_path,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING",
                (
                    corpus["id"],
                    corpus["title"],
                    corpus["version"],
                    str(manifest_path),
                    json.dumps(corpus, sort_keys=True),
                    now,
                ),
            )
            result: Dict[str, str] = {}
            for source in sources:
                row = connection.execute(
                    "SELECT id, metadata_json FROM sources WHERE corpus_id=? AND import_key=?",
                    (corpus["id"], source["import_key"]),
                ).fetchone()
                source_id = row["id"] if row else new_research_id("source")
                if row:
                    previous = json.loads(row["metadata_json"])
                    if self._source_identity_payload(
                        previous
                    ) != self._source_identity_payload(source):
                        raise ValueError(
                            f"source metadata changed for {source['import_key']}; "
                            "versioned source metadata is not implemented"
                        )
                else:
                    connection.execute(
                        "INSERT INTO sources(id,corpus_id,import_key,title,source_type,processing_state,metadata_json,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            source_id,
                            corpus["id"],
                            source["import_key"],
                            source["title"],
                            source["source_type"],
                            source["status"],
                            json.dumps(source, sort_keys=True),
                            now,
                        ),
                    )
                result[source["import_key"]] = source_id
            connection.execute(
                "INSERT INTO manifest_imports(id,corpus_id,manifest_checksum,manifest_version,manifest_path,snapshot_json,imported_at) "
                "VALUES(?,?,?,?,?,?,?) ON CONFLICT(corpus_id,manifest_checksum) DO NOTHING",
                (
                    new_research_id("manifest"),
                    corpus["id"],
                    manifest_checksum,
                    corpus["version"],
                    str(manifest_path),
                    json.dumps(snapshot, sort_keys=True),
                    now,
                ),
            )
        return result

    @staticmethod
    def _source_identity_payload(source: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in source.items() if key != "representations"}

    def manifest_import_count(self, corpus_id: str) -> int:
        with self.store.connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM manifest_imports WHERE corpus_id=?", (corpus_id,)
            ).fetchone()[0]

    def representation_exists(self, source_id: str, kind: str, checksum: str) -> bool:
        with self.store.connection() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM representations WHERE source_id=? AND kind=? AND checksum=?",
                    (source_id, kind, checksum),
                ).fetchone()
                is not None
            )

    def insert_representation(
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
        now = utc_now().isoformat()
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM representations WHERE source_id=? AND kind=? AND checksum=?",
                (source_id, kind, checksum),
            ).fetchone()
            if existing:
                return existing["id"]
            predecessor = connection.execute(
                "SELECT id FROM representations WHERE source_id=? AND kind=? AND selected=1",
                (source_id, kind),
            ).fetchone()
            representation_id = new_research_id("representation")
            connection.execute(
                "INSERT OR IGNORE INTO representation_blobs(checksum,byte_length,managed_path,created_at) VALUES(?,?,?,?)",
                (checksum, byte_length, str(managed_path), now),
            )
            connection.execute(
                "INSERT INTO representations(id,source_id,kind,checksum,selected,review_state,textual_form,metadata_json,predecessor_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    representation_id,
                    source_id,
                    kind,
                    checksum,
                    0 if predecessor else 1,
                    review_state,
                    kind,
                    json.dumps(metadata, sort_keys=True),
                    predecessor["id"] if predecessor else None,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO source_units(id,representation_id,unit_key,locator_json,processing_state,created_at) VALUES(?,?,?,?,?,?)",
                (
                    new_research_id("unit"),
                    representation_id,
                    "document",
                    json.dumps(
                        {
                            "type": "text",
                            "coordinate_system": "unicode_codepoint",
                            "start": 0,
                            "end": character_length,
                        }
                    ),
                    "usable",
                    now,
                ),
            )
        return representation_id

    def list_representations(
        self, corpus_id: str, import_key: str
    ) -> List[sqlite3.Row]:
        with self.store.connection() as connection:
            return connection.execute(
                "SELECT r.*, b.managed_path, b.byte_length FROM representations r "
                "JOIN representation_blobs b ON b.checksum=r.checksum "
                "JOIN sources s ON s.id=r.source_id WHERE s.corpus_id=? AND s.import_key=? "
                "ORDER BY r.created_at, r.rowid",
                (corpus_id, import_key),
            ).fetchall()

    def get_representation(self, representation_id: str) -> sqlite3.Row:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT r.*, b.managed_path, b.byte_length FROM representations r "
                "JOIN representation_blobs b ON b.checksum=r.checksum WHERE r.id=?",
                (representation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(representation_id)
        return row

    def get_source_unit_id(self, representation_id: str, unit_key: str) -> str:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT id FROM source_units WHERE representation_id=? AND unit_key=?",
                (representation_id, unit_key),
            ).fetchone()
        if row is None:
            raise KeyError((representation_id, unit_key))
        return row["id"]

    def ensure_research_episode(self, manifest: ObservationManifest) -> str:
        """Create an episode after proving its observation references cohere."""
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM research_episodes WHERE request_key=?",
                (manifest.request_key,),
            ).fetchone()
            if existing is not None:
                return existing["id"]
            source_time, accepted_state, included = self._observation_provenance(
                connection,
                manifest.workspace_revision_id,
                manifest.representation_id,
                manifest.source_unit_id,
            )
            if manifest.source_created_or_published_at != source_time:
                raise ValueError(
                    "observation does not match stored source publication data"
                )
            if manifest.accepted_state_before != accepted_state:
                raise ValueError("observation does not match stored accepted state")
            if manifest.evidence_scope is EvidenceScope.WORKSPACE and not included:
                raise ValueError(
                    "workspace revision does not include intended evidence"
                )
            if (
                manifest.evidence_scope is EvidenceScope.EXPLICIT_OUT_OF_WORKSPACE
                and not (manifest.out_of_workspace_reason or "").strip()
            ):
                raise ValueError("explicit out-of-workspace evidence requires a reason")
            connection.execute(
                "INSERT INTO research_episodes("
                "id,request_key,task_id,workspace_revision_id,representation_id,"
                "source_unit_id,observation_manifest_json,model,prompt_version,"
                "task_version,schema_version,code_version,status,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    manifest.episode_id,
                    manifest.request_key,
                    manifest.task_id,
                    manifest.workspace_revision_id,
                    manifest.representation_id,
                    manifest.source_unit_id,
                    manifest.model_dump_json(),
                    manifest.model,
                    manifest.prompt_version,
                    manifest.task_version,
                    manifest.schema_version,
                    manifest.code_version,
                    "running",
                    utc_now().isoformat(),
                ),
            )
        return manifest.episode_id

    def derive_observation_manifest(
        self, request: ObservationRequest
    ) -> ObservationManifest:
        with self.store.connection() as connection:
            source_time, accepted_state, included = self._observation_provenance(
                connection,
                request.workspace_revision_id,
                request.representation_id,
                request.source_unit_id,
            )
        if request.evidence_scope is EvidenceScope.WORKSPACE and not included:
            raise ValueError("workspace revision does not include intended evidence")
        return ObservationManifest(
            **request.model_dump(),
            accepted_state_before=accepted_state,
            source_created_or_published_at=source_time,
        )

    @staticmethod
    def _observation_provenance(
        connection,
        workspace_revision_id: str,
        representation_id: str,
        source_unit_id: str,
    ) -> Tuple[str | None, Dict[str, Any], bool]:
        row = connection.execute(
            "SELECT s.id AS source_id,s.metadata_json FROM workspace_revisions wr "
            "JOIN source_units su ON su.id=? "
            "JOIN representations r ON r.id=su.representation_id "
            "JOIN sources s ON s.id=r.source_id "
            "WHERE wr.id=? AND su.representation_id=? AND r.id=?",
            (
                source_unit_id,
                workspace_revision_id,
                representation_id,
                representation_id,
            ),
        ).fetchone()
        if row is None:
            raise ValueError(
                "observation must name an existing workspace revision and a source "
                "unit belonging to its immutable representation"
            )
        metadata = json.loads(row["metadata_json"])
        bibliographic = metadata.get("metadata", {})
        source_time = next(
            (
                bibliographic[key]
                for key in ("publication_date", "creation_date", "date")
                if bibliographic.get(key) is not None
            ),
            None,
        )
        included = (
            connection.execute(
                "SELECT 1 FROM workspace_memberships WHERE revision_id=? "
                "AND inclusion_state='included' AND ((record_kind='source' AND record_id=?) "
                "OR (record_kind='representation' AND record_id=?) "
                "OR (record_kind='source_unit' AND record_id=?)) LIMIT 1",
                (
                    workspace_revision_id,
                    row["source_id"],
                    representation_id,
                    source_unit_id,
                ),
            ).fetchone()
            is not None
        )
        accepted_ids = [
            item["id"]
            for item in connection.execute(
                "SELECT rv.id FROM workspace_memberships wm "
                "JOIN record_versions rv ON rv.id=wm.record_id "
                "WHERE wm.revision_id=? AND wm.record_kind='record_version' "
                "AND wm.inclusion_state='included' AND rv.status='accepted' "
                "AND rv.is_current=1 ORDER BY rv.id",
                (workspace_revision_id,),
            ).fetchall()
        ]
        return source_time, {"accepted_record_ids": accepted_ids}, included

    def episode_status(self, episode_id: str) -> str:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT status FROM research_episodes WHERE id=?", (episode_id,)
            ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return row["status"]

    def proposal_attempt_count(self, episode_id: str) -> int:
        with self.store.connection() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM proposal_attempts WHERE episode_id=?",
                (episode_id,),
            ).fetchone()[0]

    def latest_validation_observations(
        self, episode_id: str
    ) -> List[ValidationObservation]:
        with self.store.connection() as connection:
            rows = connection.execute(
                "SELECT pv.action_id,pv.code,pv.severity,pv.deterministic,pv.message "
                "FROM proposal_validations pv "
                "JOIN proposal_batches pb ON pb.id=pv.batch_id "
                "JOIN proposal_attempts pa ON pa.id=pb.attempt_id "
                "WHERE pa.episode_id=? AND pa.attempt_number=("
                "SELECT MAX(attempt_number) FROM proposal_attempts WHERE episode_id=?"
                ") ORDER BY pv.ordinal",
                (episode_id, episode_id),
            ).fetchall()
        return [
            ValidationObservation(
                action_id=row["action_id"],
                code=row["code"],
                severity=row["severity"],
                deterministic=bool(row["deterministic"]),
                message=row["message"],
            )
            for row in rows
        ]

    def persist_persistence_failure(
        self,
        *,
        episode_id: str,
        attempt_number: int,
        failed_after_record_type: str,
        error: Exception,
    ) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                "INSERT INTO proposal_persistence_failures("
                "id,episode_id,attempt_number,failed_after_record_type,error_json,"
                "created_at) VALUES(?,?,?,?,?,?)",
                (
                    new_research_id("persistence_failure"),
                    episode_id,
                    attempt_number,
                    failed_after_record_type,
                    json.dumps(
                        {"type": type(error).__name__, "message": str(error)},
                        sort_keys=True,
                    ),
                    utc_now().isoformat(),
                ),
            )

    def persist_proposal_attempt(
        self,
        *,
        episode_id: str,
        manifest: ObservationManifest,
        attempt_number: int,
        batch: ProposalBatch,
        observations: List[ValidationObservation],
        repair_hint: str | None,
        eligible: bool,
        final: bool,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        """Persist one complete proposal attempt or no part of it."""
        inject = failure_injector or (lambda _kind: None)
        now = utc_now().isoformat()
        attempt_id = new_research_id("attempt")
        batch_id = new_research_id("batch")
        transition = proposal_transition(manifest, batch)
        attempt_status = "eligible" if eligible else "validation_failed"
        with self.store.transaction() as connection:
            connection.execute(
                "INSERT INTO proposal_attempts("
                "id,episode_id,attempt_number,idempotency_key,repair_hint,status,"
                "started_at,completed_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    attempt_id,
                    episode_id,
                    attempt_number,
                    manifest.attempt_key(attempt_number),
                    repair_hint,
                    attempt_status,
                    now,
                    now,
                ),
            )
            inject("attempt")
            connection.execute(
                "INSERT INTO proposal_batches("
                "id,attempt_id,state,acceptance_eligible,batch_json,transition_json,"
                "created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    batch_id,
                    attempt_id,
                    "provisional",
                    int(eligible),
                    batch.model_dump_json(),
                    json.dumps(transition, sort_keys=True),
                    now,
                ),
            )
            inject("batch")
            for ordinal, action in enumerate(batch.actions):
                connection.execute(
                    "INSERT INTO proposal_actions("
                    "id,batch_id,ordinal,action_type,action_json,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        action.action_id,
                        batch_id,
                        ordinal,
                        action.action_type,
                        action.model_dump_json(),
                        now,
                    ),
                )
            inject("action")
            for ordinal, tool_result in enumerate(batch.tool_results):
                connection.execute(
                    "INSERT INTO proposal_tool_results("
                    "id,batch_id,ordinal,tool_call_id,tool_name,arguments_json,"
                    "result_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        new_research_id("tool_result"),
                        batch_id,
                        ordinal,
                        tool_result.tool_call_id,
                        tool_result.tool_name,
                        json.dumps(tool_result.arguments, sort_keys=True),
                        json.dumps(tool_result.result, sort_keys=True),
                        now,
                    ),
                )
            inject("tool_result")
            for ordinal, observation in enumerate(observations):
                connection.execute(
                    "INSERT INTO proposal_validations("
                    "id,batch_id,ordinal,action_id,code,severity,deterministic,"
                    "message,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        new_research_id("validation"),
                        batch_id,
                        ordinal,
                        observation.action_id,
                        observation.code,
                        observation.severity,
                        int(observation.deterministic),
                        observation.message,
                        now,
                    ),
                )
            inject("validation")
            if final:
                status = "completed" if eligible else "review_required"
                connection.execute(
                    "UPDATE research_episodes SET status=?,completed_at=?,"
                    "accepted_state_after_json=? WHERE id=?",
                    (
                        status,
                        now,
                        json.dumps(manifest.accepted_state_before, sort_keys=True),
                        episode_id,
                    ),
                )

    def persist_generation_failure(
        self,
        *,
        episode_id: str,
        manifest: ObservationManifest,
        attempt_number: int,
        error: Exception,
        repair_hint: str | None,
        final: bool,
    ) -> None:
        now = utc_now().isoformat()
        with self.store.transaction() as connection:
            connection.execute(
                "INSERT INTO proposal_attempts("
                "id,episode_id,attempt_number,idempotency_key,repair_hint,status,"
                "error_json,started_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    new_research_id("attempt"),
                    episode_id,
                    attempt_number,
                    manifest.attempt_key(attempt_number),
                    repair_hint,
                    "generation_failed",
                    json.dumps(
                        {"type": type(error).__name__, "message": str(error)},
                        sort_keys=True,
                    ),
                    now,
                    now,
                ),
            )
            if final:
                connection.execute(
                    "UPDATE research_episodes SET status='failed',completed_at=?,"
                    "accepted_state_after_json=? WHERE id=?",
                    (
                        now,
                        json.dumps(manifest.accepted_state_before, sort_keys=True),
                        episode_id,
                    ),
                )

    def load_proposal_episode(self, episode_id: str) -> StoredProposalEpisode:
        with self.store.connection() as connection:
            episode = connection.execute(
                "SELECT status FROM research_episodes WHERE id=?", (episode_id,)
            ).fetchone()
            if episode is None:
                raise KeyError(episode_id)
            attempts = connection.execute(
                "SELECT COUNT(*) FROM proposal_attempts WHERE episode_id=?",
                (episode_id,),
            ).fetchone()[0]
            row = connection.execute(
                "SELECT pb.batch_json,pb.transition_json,pb.acceptance_eligible,pb.id "
                "FROM proposal_batches pb JOIN proposal_attempts pa ON pa.id=pb.attempt_id "
                "WHERE pa.episode_id=? ORDER BY pa.attempt_number DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
            codes: List[str] = []
            batch = None
            eligible = False
            transition = None
            if row is not None:
                batch = ProposalBatch.model_validate_json(row["batch_json"])
                transition = json.loads(row["transition_json"])
                eligible = bool(row["acceptance_eligible"])
                codes = [
                    item["code"]
                    for item in connection.execute(
                        "SELECT code FROM proposal_validations WHERE batch_id=? "
                        "ORDER BY ordinal",
                        (row["id"],),
                    ).fetchall()
                ]
        return StoredProposalEpisode(
            episode_id=episode_id,
            status=episode["status"],
            acceptance_eligible=eligible,
            attempts=attempts,
            batch=batch,
            validation_codes=codes,
            transition=transition,
        )

    def load_episode_manifest(self, episode_id: str) -> ObservationManifest:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT observation_manifest_json FROM research_episodes WHERE id=?",
                (episode_id,),
            ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return ObservationManifest.model_validate_json(row["observation_manifest_json"])

    def counts(self) -> Dict[str, int]:
        with self.store.connection() as connection:
            return {
                "sources": connection.execute(
                    "SELECT COUNT(*) FROM sources"
                ).fetchone()[0],
                "representation_blobs": connection.execute(
                    "SELECT COUNT(*) FROM representation_blobs"
                ).fetchone()[0],
                "representations": connection.execute(
                    "SELECT COUNT(*) FROM representations"
                ).fetchone()[0],
                "source_units": connection.execute(
                    "SELECT COUNT(*) FROM source_units"
                ).fetchone()[0],
            }

    def get_source_metadata(self, corpus_id: str, import_key: str) -> Dict[str, Any]:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM sources WHERE corpus_id=? AND import_key=?",
                (corpus_id, import_key),
            ).fetchone()
        if row is None:
            raise KeyError((corpus_id, import_key))
        payload = json.loads(row["metadata_json"])
        nested = payload.pop("metadata", {})
        return {**nested, **payload}

    def get_corpus_metadata(self, corpus_id: str) -> Dict[str, Any]:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT metadata_json FROM corpora WHERE id=?", (corpus_id,)
            ).fetchone()
        if row is None:
            raise KeyError(corpus_id)
        return json.loads(row["metadata_json"])

    def get_source_metadata_for_representation(
        self, representation_id: str, corpus_id: str, import_key: str
    ) -> Dict[str, Any]:
        with self.store.connection() as connection:
            representation = connection.execute(
                "SELECT 1 FROM representations WHERE id=?", (representation_id,)
            ).fetchone()
            if representation is None:
                raise KeyError(representation_id)
            row = connection.execute(
                "SELECT s.metadata_json FROM representations r "
                "JOIN sources s ON s.id=r.source_id "
                "WHERE r.id=? AND s.corpus_id=? AND s.import_key=?",
                (representation_id, corpus_id, import_key),
            ).fetchone()
        if row is None:
            raise ValueError("representation does not belong to requested source")
        payload = json.loads(row["metadata_json"])
        nested = payload.pop("metadata", {})
        return {**nested, **payload}

    def referenced_blob_checksums(self) -> set[str]:
        with self.store.connection() as connection:
            return {
                row["checksum"]
                for row in connection.execute(
                    "SELECT DISTINCT checksum FROM representation_blobs"
                ).fetchall()
            }

    def ensure_workspace(self, key: str, title: str) -> Workspace:
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM research_workspaces WHERE workspace_key=?", (key,)
            ).fetchone()
            if row is None:
                workspace_id = new_research_id("workspace")
                connection.execute(
                    "INSERT INTO research_workspaces(id,workspace_key,title,created_at) VALUES(?,?,?,?)",
                    (workspace_id, key, title, utc_now().isoformat()),
                )
            else:
                workspace_id = row["id"]
        return Workspace(id=workspace_id, workspace_key=key, title=title)

    def create_revision(
        self,
        workspace_id: str,
        memberships: List[Tuple[str, str, str]],
        reviewed_context: Dict[str, Any],
    ) -> WorkspaceRevision:
        self._validate_memberships(memberships)
        digest = revision_hash(memberships, reviewed_context)
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT id FROM workspace_revisions WHERE workspace_id=? AND content_hash=?",
                (workspace_id, digest),
            ).fetchone()
            revision_id = row["id"] if row else new_research_id("revision")
            if row is None:
                connection.execute(
                    "INSERT INTO workspace_revisions(id,workspace_id,content_hash,reviewed_context_json,created_at) VALUES(?,?,?,?,?)",
                    (
                        revision_id,
                        workspace_id,
                        digest,
                        json.dumps(reviewed_context, sort_keys=True),
                        utc_now().isoformat(),
                    ),
                )
                connection.executemany(
                    "INSERT INTO workspace_memberships(revision_id,record_kind,record_id,inclusion_state) VALUES(?,?,?,?)",
                    [(revision_id, *item) for item in sorted(memberships)],
                )
        return self.get_revision(revision_id)

    def _validate_memberships(self, memberships: List[Tuple[str, str, str]]) -> None:
        seen: set[Tuple[str, str]] = set()
        allowed = {
            "source",
            "representation",
            "source_unit",
            "anchor",
            "record_version",
        }
        for record_kind, record_id, state in memberships:
            key = (record_kind, record_id)
            if key in seen:
                raise ValueError(
                    f"duplicate workspace membership for {record_kind}:{record_id}"
                )
            seen.add(key)
            if record_kind not in allowed:
                raise ValueError(f"unsupported workspace record kind: {record_kind}")
            if state not in {"included", "excluded"}:
                raise ValueError(f"unsupported workspace inclusion state: {state}")
        with self.store.connection() as connection:
            for record_kind, record_id, _state in memberships:
                if not self._membership_record_exists(
                    connection, record_kind, record_id
                ):
                    raise KeyError(key)

    @staticmethod
    def _membership_record_exists(connection, kind: str, record_id: str) -> bool:
        queries = {
            "source": "SELECT 1 FROM sources WHERE id=?",
            "representation": "SELECT 1 FROM representations WHERE id=?",
            "source_unit": "SELECT 1 FROM source_units WHERE id=?",
            "anchor": "SELECT 1 FROM anchors WHERE id=?",
            "record_version": "SELECT 1 FROM record_versions WHERE id=?",
        }
        return connection.execute(queries[kind], (record_id,)).fetchone() is not None

    def get_revision(self, revision_id: str) -> WorkspaceRevision:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT * FROM workspace_revisions WHERE id=?", (revision_id,)
            ).fetchone()
            if row is None:
                raise KeyError(revision_id)
            memberships = connection.execute(
                "SELECT record_kind,record_id,inclusion_state FROM workspace_memberships WHERE revision_id=? ORDER BY record_kind,record_id",
                (revision_id,),
            ).fetchall()
        return WorkspaceRevision(
            id=row["id"],
            workspace_id=row["workspace_id"],
            content_hash=row["content_hash"],
            memberships=[tuple(item) for item in memberships],
            reviewed_context=json.loads(row["reviewed_context_json"]),
        )
