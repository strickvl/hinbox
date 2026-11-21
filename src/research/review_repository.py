"""Transactional persistence for human proposal review."""

import json
from typing import Any, Callable, Dict, Tuple

from src.research.extraction import ProposalBatch
from src.research.models import RecordVersion, new_research_id, utc_now
from src.research.review import (
    ProposalJudgment,
    ProposalReviewCommand,
    ProposalReviewResult,
    StaleReviewError,
    UnacceptableProposalError,
    WorkspaceInclusion,
)
from src.research.validation import acceptance_eligible, validate_proposal_batch


class ReviewRepository:
    """Keep review SQL out of application and route layers."""

    def __init__(self, store, quote_retriever: Callable[[str, int, int], str]) -> None:
        self.store = store
        self.quote_retriever = quote_retriever

    def review_proposal(self, command: ProposalReviewCommand) -> ProposalReviewResult:
        now = utc_now()
        with self.store.transaction() as connection:
            proposal = self._proposal_row(
                connection, command.episode_id, command.action_id
            )
            subject = connection.execute(
                "SELECT * FROM proposal_review_subjects "
                "WHERE episode_id=? AND action_id=?",
                (command.episode_id, command.action_id),
            ).fetchone()
            if subject is None:
                if command.expected_version != 0:
                    raise StaleReviewError(
                        f"expected review version {command.expected_version}, found 0"
                    )
                subject_id = new_research_id("review_subject")
                connection.execute(
                    "INSERT INTO proposal_review_subjects("
                    "id,episode_id,batch_id,action_id,version,created_at,updated_at) "
                    "VALUES(?,?,?,?,0,?,?)",
                    (
                        subject_id,
                        command.episode_id,
                        proposal["batch_id"],
                        command.action_id,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
                current_version = 0
                logical_record_id = None
                selected_record_version_id = None
                last_record_version_id = None
            else:
                subject_id = subject["id"]
                current_version = subject["version"]
                logical_record_id = subject["logical_record_id"]
                selected_record_version_id = subject["selected_record_version_id"]
                last_record_version_id = subject["last_record_version_id"]
            self._require_expected_version(command.expected_version, current_version)

            next_version = current_version + 1
            record, selected_after, last_after = self._reviewed_record(
                connection,
                command=command,
                proposal=proposal,
                logical_record_id=logical_record_id,
                selected_record_version_id=selected_record_version_id,
                last_record_version_id=last_record_version_id,
                now=now,
            )
            decision_id = new_research_id("review_decision")
            connection.execute(
                "INSERT INTO proposal_review_decisions("
                "id,subject_id,subject_version,judgment,correctness,relevance,reason,"
                "edited_payload_json,actor_id,resulting_record_version_id,"
                "evaluation_eligible,training_eligible,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    decision_id,
                    subject_id,
                    next_version,
                    command.judgment.value,
                    command.correctness.value,
                    command.relevance.value,
                    command.reason,
                    self._json_or_none(command.edited_payload),
                    command.actor_id,
                    record.id if record else None,
                    int(command.evaluation_eligible),
                    int(command.training_eligible),
                    now.isoformat(),
                ),
            )
            self._insert_inclusion_decision(
                connection,
                subject_id=subject_id,
                subject_version=next_version,
                inclusion=command.inclusion,
                reason=command.inclusion_reason,
                actor_id=command.actor_id,
                now=now.isoformat(),
            )
            changed = connection.execute(
                "UPDATE proposal_review_subjects SET version=?,current_decision_id=?,"
                "logical_record_id=?,selected_record_version_id=?,"
                "last_record_version_id=?,updated_at=? "
                "WHERE id=? AND version=?",
                (
                    next_version,
                    decision_id,
                    record.logical_record_id if record else logical_record_id,
                    selected_after,
                    last_after,
                    now.isoformat(),
                    subject_id,
                    command.expected_version,
                ),
            ).rowcount
            if changed != 1:
                raise StaleReviewError("review state changed before submission")

        return ProposalReviewResult(
            subject_id=subject_id,
            decision_id=decision_id,
            episode_id=command.episode_id,
            action_id=command.action_id,
            version=next_version,
            judgment=command.judgment,
            correctness=command.correctness,
            relevance=command.relevance,
            inclusion=command.inclusion,
            reason=command.reason,
            inclusion_reason=command.inclusion_reason,
            logical_record_id=record.logical_record_id if record else logical_record_id,
            record_version_id=selected_after,
        )

    def set_inclusion(
        self,
        episode_id: str,
        action_id: str,
        *,
        expected_version: int,
        inclusion: WorkspaceInclusion,
        reason: str | None,
        actor_id: str,
    ) -> ProposalReviewResult:
        now = utc_now()
        with self.store.transaction() as connection:
            subject = connection.execute(
                "SELECT * FROM proposal_review_subjects "
                "WHERE episode_id=? AND action_id=?",
                (episode_id, action_id),
            ).fetchone()
            if subject is None:
                raise KeyError((episode_id, action_id))
            self._require_expected_version(expected_version, subject["version"])
            decision = connection.execute(
                "SELECT * FROM proposal_review_decisions WHERE id=?",
                (subject["current_decision_id"],),
            ).fetchone()
            next_version = expected_version + 1
            self._insert_inclusion_decision(
                connection,
                subject_id=subject["id"],
                subject_version=next_version,
                inclusion=inclusion,
                reason=reason,
                actor_id=actor_id,
                now=now.isoformat(),
            )
            changed = connection.execute(
                "UPDATE proposal_review_subjects SET version=?,updated_at=? "
                "WHERE id=? AND version=?",
                (next_version, now.isoformat(), subject["id"], expected_version),
            ).rowcount
            if changed != 1:
                raise StaleReviewError("review state changed before submission")

        return ProposalReviewResult(
            subject_id=subject["id"],
            decision_id=decision["id"],
            episode_id=episode_id,
            action_id=action_id,
            version=next_version,
            judgment=decision["judgment"],
            correctness=decision["correctness"],
            relevance=decision["relevance"],
            inclusion=inclusion,
            reason=decision["reason"],
            inclusion_reason=reason,
            logical_record_id=subject["logical_record_id"],
            record_version_id=subject["selected_record_version_id"],
        )

    def export_review(self, episode_id: str, action_id: str) -> Dict[str, Any]:
        with self.store.connection() as connection:
            subject = connection.execute(
                "SELECT * FROM proposal_review_subjects "
                "WHERE episode_id=? AND action_id=?",
                (episode_id, action_id),
            ).fetchone()
            if subject is None:
                raise KeyError((episode_id, action_id))
            proposal = connection.execute(
                "SELECT action_json FROM proposal_actions WHERE batch_id=? AND id=?",
                (subject["batch_id"], action_id),
            ).fetchone()
            decisions = connection.execute(
                "SELECT * FROM proposal_review_decisions WHERE subject_id=? "
                "ORDER BY subject_version",
                (subject["id"],),
            ).fetchall()
            inclusions = connection.execute(
                "SELECT * FROM proposal_workspace_decisions WHERE subject_id=? "
                "ORDER BY subject_version",
                (subject["id"],),
            ).fetchall()
        edits = [
            json.loads(row["edited_payload_json"])
            for row in decisions
            if row["edited_payload_json"] is not None
        ]
        return {
            "episode_id": episode_id,
            "action_id": action_id,
            "model_proposal": json.loads(proposal["action_json"]),
            "human_edit": edits[-1] if edits else None,
            "historian_decisions": [self._decision_export(row) for row in decisions],
            "inclusion_history": [self._inclusion_export(row) for row in inclusions],
        }

    @staticmethod
    def _proposal_row(connection, episode_id: str, action_id: str):
        row = connection.execute(
            "SELECT action.batch_id,action.action_type,action.action_json,"
            "batch.batch_json,batch.acceptance_eligible,episode.representation_id "
            "FROM proposal_actions action "
            "JOIN proposal_batches batch ON batch.id=action.batch_id "
            "JOIN proposal_attempts attempt ON attempt.id=batch.attempt_id "
            "JOIN research_episodes episode ON episode.id=attempt.episode_id "
            "WHERE attempt.episode_id=? AND action.id=? "
            "ORDER BY attempt.attempt_number DESC LIMIT 1",
            (episode_id, action_id),
        ).fetchone()
        if row is None:
            raise KeyError((episode_id, action_id))
        return row

    @staticmethod
    def _require_expected_version(expected: int, actual: int) -> None:
        if expected != actual:
            raise StaleReviewError(
                f"expected review version {expected}, found {actual}"
            )

    def _reviewed_record(
        self,
        connection,
        *,
        command: ProposalReviewCommand,
        proposal,
        logical_record_id: str | None,
        selected_record_version_id: str | None,
        last_record_version_id: str | None,
        now,
    ) -> Tuple[RecordVersion | None, str | None, str | None]:
        previous_id = (
            command.supersedes_record_version_id
            or selected_record_version_id
            or last_record_version_id
        )
        previous = None
        if previous_id is not None:
            previous = connection.execute(
                "SELECT * FROM record_versions WHERE id=?",
                (previous_id,),
            ).fetchone()
            if previous is None:
                raise KeyError(previous_id)
            if (
                command.supersedes_record_version_id is not None
                and not previous["is_current"]
            ):
                raise StaleReviewError(
                    f"record version {previous_id} is no longer current"
                )
            if logical_record_id is not None and (
                previous["logical_record_id"] != logical_record_id
            ):
                raise ValueError("superseded record belongs to another review record")
            logical_record_id = previous["logical_record_id"]

        if command.logical_record_id is not None:
            if logical_record_id not in {None, command.logical_record_id}:
                raise ValueError("logical record does not match selected predecessor")
            logical_record_id = command.logical_record_id

        if command.judgment is ProposalJudgment.DEFER:
            return None, selected_record_version_id, last_record_version_id

        if command.judgment is ProposalJudgment.REJECT:
            if selected_record_version_id is not None:
                connection.execute(
                    "UPDATE record_versions SET status='superseded',is_current=0 "
                    "WHERE id=? AND is_current=1",
                    (selected_record_version_id,),
                )
            return None, None, selected_record_version_id or last_record_version_id

        if previous is not None and previous["is_current"]:
            changed = connection.execute(
                "UPDATE record_versions SET status='superseded',is_current=0 "
                "WHERE id=? AND is_current=1",
                (previous_id,),
            ).rowcount
            if changed != 1:
                raise StaleReviewError("selected record changed before review")

        payload = self._validated_payload(command, proposal)
        record = RecordVersion(
            id=new_research_id("version"),
            logical_record_id=logical_record_id or new_research_id("record"),
            kind=proposal["action_type"],
            payload=payload,
            supersedes_id=previous_id,
            actor_id=command.actor_id,
            status="accepted",
            is_current=True,
            created_at=now,
        )
        connection.execute(
            "INSERT INTO record_versions(id,logical_record_id,kind,payload_json,"
            "supersedes_id,actor_id,status,is_current,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
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
        return record, record.id, record.id

    def _validated_payload(
        self, command: ProposalReviewCommand, proposal
    ) -> Dict[str, Any]:
        batch_payload = json.loads(proposal["batch_json"])
        original = json.loads(proposal["action_json"])
        if command.judgment is ProposalJudgment.EDIT:
            edited = {**original, **(command.edited_payload or {})}
            batch_payload["actions"] = [
                edited if action["action_id"] == command.action_id else action
                for action in batch_payload["actions"]
            ]
            accepted_payload = edited
        else:
            accepted_payload = original
        batch = ProposalBatch.model_validate(batch_payload)
        observations = validate_proposal_batch(
            batch,
            self.quote_retriever,
            representation_id=proposal["representation_id"],
        )
        if not acceptance_eligible(observations):
            codes = sorted(
                {
                    observation.code
                    for observation in observations
                    if observation.deterministic and observation.severity == "severe"
                }
            )
            detail = ", ".join(codes) or "unknown invariant"
            raise UnacceptableProposalError(
                "proposal cannot become accepted evidence; "
                f"deterministic validation failed: {detail}"
            )
        return accepted_payload

    @staticmethod
    def _insert_inclusion_decision(
        connection,
        *,
        subject_id: str,
        subject_version: int,
        inclusion: WorkspaceInclusion,
        reason: str | None,
        actor_id: str,
        now: str,
    ) -> None:
        connection.execute(
            "INSERT INTO proposal_workspace_decisions("
            "id,subject_id,subject_version,inclusion,reason,actor_id,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                new_research_id("inclusion_decision"),
                subject_id,
                subject_version,
                inclusion.value,
                reason,
                actor_id,
                now,
            ),
        )

    @staticmethod
    def _json_or_none(value: Dict[str, Any] | None) -> str | None:
        return json.dumps(value, sort_keys=True) if value is not None else None

    @staticmethod
    def _decision_export(row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "version": row["subject_version"],
            "judgment": row["judgment"],
            "correctness": row["correctness"],
            "relevance": row["relevance"],
            "reason": row["reason"],
            "human_edit": (
                json.loads(row["edited_payload_json"])
                if row["edited_payload_json"] is not None
                else None
            ),
            "record_version_id": row["resulting_record_version_id"],
            "evaluation_eligible": bool(row["evaluation_eligible"]),
            "training_eligible": bool(row["training_eligible"]),
            "actor_id": row["actor_id"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _inclusion_export(row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "version": row["subject_version"],
            "inclusion": row["inclusion"],
            "reason": row["reason"],
            "actor_id": row["actor_id"],
            "created_at": row["created_at"],
        }
