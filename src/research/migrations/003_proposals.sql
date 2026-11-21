CREATE TABLE research_episodes (
    id TEXT PRIMARY KEY,
    request_key TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    workspace_revision_id TEXT NOT NULL REFERENCES workspace_revisions(id) ON DELETE RESTRICT,
    representation_id TEXT NOT NULL REFERENCES representations(id) ON DELETE RESTRICT,
    source_unit_id TEXT NOT NULL REFERENCES source_units(id) ON DELETE RESTRICT,
    observation_manifest_json TEXT NOT NULL CHECK (json_valid(observation_manifest_json)),
    accepted_state_after_json TEXT CHECK (accepted_state_after_json IS NULL OR json_valid(accepted_state_after_json)),
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    task_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    code_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'review_required', 'failed')),
    created_at TEXT NOT NULL,
    completed_at TEXT
) STRICT;

CREATE TABLE proposal_attempts (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES research_episodes(id) ON DELETE RESTRICT,
    attempt_number INTEGER NOT NULL CHECK (attempt_number IN (1, 2)),
    idempotency_key TEXT NOT NULL UNIQUE,
    repair_hint TEXT,
    status TEXT NOT NULL CHECK (status IN ('eligible', 'validation_failed', 'generation_failed')),
    error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    UNIQUE(episode_id, attempt_number)
) STRICT;

CREATE TABLE proposal_persistence_failures (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES research_episodes(id) ON DELETE RESTRICT,
    attempt_number INTEGER NOT NULL CHECK (attempt_number IN (1, 2)),
    failed_after_record_type TEXT NOT NULL,
    error_json TEXT NOT NULL CHECK (json_valid(error_json)),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE proposal_batches (
    id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL UNIQUE REFERENCES proposal_attempts(id) ON DELETE RESTRICT,
    state TEXT NOT NULL CHECK (state = 'provisional'),
    acceptance_eligible INTEGER NOT NULL CHECK (acceptance_eligible IN (0, 1)),
    batch_json TEXT NOT NULL CHECK (json_valid(batch_json)),
    transition_json TEXT NOT NULL CHECK (json_valid(transition_json)),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE proposal_actions (
    id TEXT NOT NULL,
    batch_id TEXT NOT NULL REFERENCES proposal_batches(id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    action_type TEXT NOT NULL CHECK (action_type IN ('passage', 'source_assertion', 'normalized_claim', 'mention', 'relationship', 'knowledge_position')),
    action_json TEXT NOT NULL CHECK (json_valid(action_json)),
    created_at TEXT NOT NULL,
    PRIMARY KEY(batch_id, id),
    UNIQUE(batch_id, ordinal)
) STRICT;

CREATE TABLE proposal_tool_results (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES proposal_batches(id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL CHECK (json_valid(arguments_json)),
    result_json TEXT NOT NULL CHECK (json_valid(result_json)),
    created_at TEXT NOT NULL,
    UNIQUE(batch_id, ordinal)
) STRICT;

CREATE TABLE proposal_validations (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES proposal_batches(id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    action_id TEXT,
    code TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'severe')),
    deterministic INTEGER NOT NULL CHECK (deterministic IN (0, 1)),
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(batch_id, ordinal)
) STRICT;

CREATE TRIGGER research_episodes_append_only_update
BEFORE UPDATE ON research_episodes
BEGIN
    SELECT CASE WHEN
        NEW.id <> OLD.id OR NEW.request_key <> OLD.request_key
        OR NEW.task_id <> OLD.task_id
        OR NEW.workspace_revision_id <> OLD.workspace_revision_id
        OR NEW.representation_id <> OLD.representation_id
        OR NEW.source_unit_id <> OLD.source_unit_id
        OR NEW.observation_manifest_json <> OLD.observation_manifest_json
        OR NEW.model <> OLD.model OR NEW.prompt_version <> OLD.prompt_version
        OR NEW.task_version <> OLD.task_version OR NEW.schema_version <> OLD.schema_version
        OR NEW.code_version <> OLD.code_version OR NEW.created_at <> OLD.created_at
        OR OLD.status <> 'running' OR NEW.status NOT IN ('completed', 'review_required', 'failed')
        OR NEW.completed_at IS NULL OR NEW.accepted_state_after_json IS NULL
    THEN RAISE(ABORT, 'research episode is append-only except finalization') END;
END;

CREATE TRIGGER proposal_attempts_immutable
BEFORE UPDATE ON proposal_attempts
BEGIN SELECT RAISE(ABORT, 'proposal attempts are immutable'); END;

CREATE TRIGGER proposal_batches_immutable
BEFORE UPDATE ON proposal_batches
BEGIN SELECT RAISE(ABORT, 'proposal batches are immutable'); END;

CREATE TRIGGER proposal_actions_immutable
BEFORE UPDATE ON proposal_actions
BEGIN SELECT RAISE(ABORT, 'proposal actions are immutable'); END;

CREATE TRIGGER proposal_persistence_failures_immutable
BEFORE UPDATE ON proposal_persistence_failures
BEGIN SELECT RAISE(ABORT, 'proposal persistence failures are immutable'); END;

CREATE TRIGGER proposal_tool_results_immutable
BEFORE UPDATE ON proposal_tool_results
BEGIN SELECT RAISE(ABORT, 'proposal tool results are immutable'); END;

CREATE TRIGGER proposal_validations_immutable
BEFORE UPDATE ON proposal_validations
BEGIN SELECT RAISE(ABORT, 'proposal validations are immutable'); END;

CREATE TRIGGER research_episodes_no_delete
BEFORE DELETE ON research_episodes
BEGIN SELECT RAISE(ABORT, 'research episodes are append-only'); END;

CREATE TRIGGER proposal_attempts_no_delete
BEFORE DELETE ON proposal_attempts
BEGIN SELECT RAISE(ABORT, 'proposal attempts are append-only'); END;

CREATE TRIGGER proposal_persistence_failures_no_delete
BEFORE DELETE ON proposal_persistence_failures
BEGIN SELECT RAISE(ABORT, 'proposal persistence failures are append-only'); END;

CREATE TRIGGER proposal_batches_no_delete
BEFORE DELETE ON proposal_batches
BEGIN SELECT RAISE(ABORT, 'proposal batches are append-only'); END;

CREATE TRIGGER proposal_actions_no_delete
BEFORE DELETE ON proposal_actions
BEGIN SELECT RAISE(ABORT, 'proposal actions are append-only'); END;

CREATE TRIGGER proposal_tool_results_no_delete
BEFORE DELETE ON proposal_tool_results
BEGIN SELECT RAISE(ABORT, 'proposal tool results are append-only'); END;

CREATE TRIGGER proposal_validations_no_delete
BEFORE DELETE ON proposal_validations
BEGIN SELECT RAISE(ABORT, 'proposal validations are append-only'); END;
