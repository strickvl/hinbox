CREATE TABLE proposal_review_subjects (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES research_episodes(id) ON DELETE RESTRICT,
    batch_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    current_decision_id TEXT REFERENCES proposal_review_decisions(id) ON DELETE RESTRICT,
    logical_record_id TEXT,
    selected_record_version_id TEXT REFERENCES record_versions(id) ON DELETE RESTRICT,
    last_record_version_id TEXT REFERENCES record_versions(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(episode_id, action_id),
    FOREIGN KEY(batch_id, action_id)
        REFERENCES proposal_actions(batch_id, id) ON DELETE RESTRICT
) STRICT;

CREATE TABLE proposal_review_decisions (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES proposal_review_subjects(id) ON DELETE RESTRICT,
    subject_version INTEGER NOT NULL CHECK (subject_version > 0),
    judgment TEXT NOT NULL CHECK (judgment IN ('accept', 'edit', 'reject', 'defer')),
    correctness TEXT NOT NULL CHECK (correctness IN ('correct', 'incorrect', 'uncertain', 'not_assessed')),
    relevance TEXT NOT NULL CHECK (relevance IN ('relevant', 'irrelevant', 'uncertain', 'not_assessed')),
    reason TEXT,
    edited_payload_json TEXT CHECK (edited_payload_json IS NULL OR json_valid(edited_payload_json)),
    actor_id TEXT NOT NULL REFERENCES audit_actors(id) ON DELETE RESTRICT,
    resulting_record_version_id TEXT REFERENCES record_versions(id) ON DELETE RESTRICT,
    evaluation_eligible INTEGER NOT NULL CHECK (evaluation_eligible IN (0, 1)),
    training_eligible INTEGER NOT NULL CHECK (training_eligible IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE(subject_id, subject_version),
    CHECK ((judgment = 'edit') = (edited_payload_json IS NOT NULL)),
    CHECK (judgment IN ('accept', 'edit') OR resulting_record_version_id IS NULL)
) STRICT;

CREATE TABLE proposal_workspace_decisions (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES proposal_review_subjects(id) ON DELETE RESTRICT,
    subject_version INTEGER NOT NULL CHECK (subject_version > 0),
    inclusion TEXT NOT NULL CHECK (inclusion IN ('not_decided', 'included', 'excluded')),
    reason TEXT,
    actor_id TEXT NOT NULL REFERENCES audit_actors(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(subject_id, subject_version)
) STRICT;

CREATE INDEX proposal_review_decisions_subject_history
ON proposal_review_decisions(subject_id, subject_version);

CREATE INDEX proposal_workspace_decisions_subject_history
ON proposal_workspace_decisions(subject_id, subject_version);

CREATE TRIGGER proposal_review_decisions_human_only_insert
BEFORE INSERT ON proposal_review_decisions
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM audit_actors
        WHERE id = NEW.actor_id AND actor_type = 'human'
    ) THEN RAISE(ABORT, 'proposal review requires a human actor') END;
END;

CREATE TRIGGER proposal_workspace_decisions_human_only_insert
BEFORE INSERT ON proposal_workspace_decisions
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM audit_actors
        WHERE id = NEW.actor_id AND actor_type = 'human'
    ) THEN RAISE(ABORT, 'workspace inclusion review requires a human actor') END;
END;

CREATE TRIGGER accepted_record_versions_human_only_insert
BEFORE INSERT ON record_versions
WHEN NEW.status = 'accepted' OR NEW.is_current = 1
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM audit_actors
        WHERE id = NEW.actor_id AND actor_type = 'human'
    ) THEN RAISE(ABORT, 'accepted research evidence requires a human actor') END;
END;

CREATE TRIGGER proposal_review_subjects_guarded_update
BEFORE UPDATE ON proposal_review_subjects
BEGIN
    SELECT CASE WHEN
        NEW.id <> OLD.id
        OR NEW.episode_id <> OLD.episode_id
        OR NEW.batch_id <> OLD.batch_id
        OR NEW.action_id <> OLD.action_id
        OR NEW.created_at <> OLD.created_at
        OR NEW.version <> OLD.version + 1
    THEN RAISE(ABORT, 'invalid proposal review state transition') END;
    SELECT CASE WHEN NEW.current_decision_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM proposal_review_decisions decision
        WHERE decision.id = NEW.current_decision_id
          AND decision.subject_id = NEW.id
          AND decision.subject_version <= NEW.version
    ) THEN RAISE(ABORT, 'current review decision does not match subject version') END;
    SELECT CASE WHEN NEW.selected_record_version_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM record_versions record
        WHERE record.id = NEW.selected_record_version_id
          AND record.logical_record_id = NEW.logical_record_id
          AND record.status = 'accepted'
          AND record.is_current = 1
    ) THEN RAISE(ABORT, 'selected review record must be the current accepted version') END;
    SELECT CASE WHEN NEW.last_record_version_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM record_versions record
        WHERE record.id = NEW.last_record_version_id
          AND record.logical_record_id = NEW.logical_record_id
    ) THEN RAISE(ABORT, 'last review record must belong to the review record') END;
END;

CREATE TRIGGER proposal_review_decisions_immutable
BEFORE UPDATE ON proposal_review_decisions
BEGIN SELECT RAISE(ABORT, 'proposal review decisions are immutable'); END;

CREATE TRIGGER proposal_workspace_decisions_immutable
BEFORE UPDATE ON proposal_workspace_decisions
BEGIN SELECT RAISE(ABORT, 'workspace decisions are immutable'); END;

CREATE TRIGGER proposal_review_decisions_no_delete
BEFORE DELETE ON proposal_review_decisions
BEGIN SELECT RAISE(ABORT, 'proposal review decisions are append-only'); END;

CREATE TRIGGER proposal_workspace_decisions_no_delete
BEFORE DELETE ON proposal_workspace_decisions
BEGIN SELECT RAISE(ABORT, 'workspace decisions are append-only'); END;

CREATE TRIGGER proposal_review_subjects_no_delete
BEFORE DELETE ON proposal_review_subjects
BEGIN SELECT RAISE(ABORT, 'proposal review subjects preserve history'); END;
