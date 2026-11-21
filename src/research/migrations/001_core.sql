CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
) STRICT;

CREATE TABLE audit_actors (
    id TEXT PRIMARY KEY,
    actor_type TEXT NOT NULL CHECK (actor_type IN ('human', 'model', 'system')),
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
) STRICT;

INSERT INTO audit_actors(id, actor_type, display_name)
VALUES ('historian', 'human', 'Configured historian');

CREATE TABLE anchors (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('person', 'organization', 'place', 'event', 'source')),
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE anchor_aliases (
    id TEXT PRIMARY KEY,
    anchor_id TEXT NOT NULL REFERENCES anchors(id) ON DELETE RESTRICT,
    alias TEXT NOT NULL CHECK (length(trim(alias)) > 0),
    UNIQUE(anchor_id, alias)
) STRICT;

CREATE TABLE record_versions (
    id TEXT PRIMARY KEY,
    logical_record_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
    supersedes_id TEXT REFERENCES record_versions(id) ON DELETE RESTRICT,
    actor_id TEXT NOT NULL REFERENCES audit_actors(id) ON DELETE RESTRICT,
    status TEXT NOT NULL CHECK (status IN ('provisional', 'accepted', 'superseded', 'rejected')),
    is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
    created_at TEXT NOT NULL,
    CHECK (is_current = 0 OR status = 'accepted'),
    CHECK (supersedes_id IS NULL OR supersedes_id <> id),
    UNIQUE(logical_record_id, id)
) STRICT;

CREATE UNIQUE INDEX one_current_record_version
ON record_versions(logical_record_id) WHERE is_current = 1;

CREATE TRIGGER record_version_supersession_same_record_insert
BEFORE INSERT ON record_versions
WHEN NEW.supersedes_id IS NOT NULL
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM record_versions previous
        WHERE previous.id = NEW.supersedes_id
          AND previous.logical_record_id = NEW.logical_record_id
    ) THEN RAISE(ABORT, 'superseded version must belong to the same record') END;
END;

CREATE TRIGGER record_version_immutable_history
BEFORE UPDATE ON record_versions
BEGIN
    SELECT CASE WHEN
        NEW.id <> OLD.id
        OR NEW.logical_record_id <> OLD.logical_record_id
        OR NEW.kind <> OLD.kind
        OR NEW.payload_json <> OLD.payload_json
        OR NEW.supersedes_id IS NOT OLD.supersedes_id
        OR NEW.actor_id <> OLD.actor_id
        OR NEW.created_at <> OLD.created_at
        OR NOT (
            NEW.status = 'superseded'
            AND NEW.is_current = 0
            AND OLD.status = 'accepted'
            AND OLD.is_current = 1
        )
    THEN RAISE(ABORT, 'record versions are immutable except accepted-to-superseded lifecycle transition') END;
END;

CREATE TRIGGER record_version_no_cycle_update
BEFORE UPDATE OF supersedes_id ON record_versions
WHEN NEW.supersedes_id IS NOT NULL
BEGIN
    WITH RECURSIVE ancestors(id) AS (
        SELECT NEW.supersedes_id
        UNION ALL
        SELECT record_versions.supersedes_id
        FROM record_versions JOIN ancestors ON record_versions.id = ancestors.id
        WHERE record_versions.supersedes_id IS NOT NULL
    )
    SELECT CASE WHEN EXISTS (SELECT 1 FROM ancestors WHERE id = NEW.id)
        THEN RAISE(ABORT, 'record version supersession cycle') END;
END;
