CREATE TABLE corpora (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    version TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE manifest_imports (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpora(id) ON DELETE RESTRICT,
    manifest_checksum TEXT NOT NULL,
    manifest_version TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    snapshot_json TEXT NOT NULL CHECK (json_valid(snapshot_json)),
    imported_at TEXT NOT NULL,
    UNIQUE(corpus_id, manifest_checksum)
) STRICT;

CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpora(id) ON DELETE RESTRICT,
    import_key TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    processing_state TEXT NOT NULL CHECK (processing_state IN
        ('included', 'excluded', 'unprocessed', 'unavailable', 'failed', 'partially_usable', 'corrected', 'superseded')),
    metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL,
    UNIQUE(corpus_id, import_key)
) STRICT;

CREATE TABLE representation_blobs (
    checksum TEXT PRIMARY KEY,
    byte_length INTEGER NOT NULL CHECK (byte_length >= 0),
    managed_path TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE representations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL,
    checksum TEXT NOT NULL REFERENCES representation_blobs(checksum) ON DELETE RESTRICT,
    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),
    review_state TEXT NOT NULL,
    textual_form TEXT NOT NULL,
    metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
    predecessor_id TEXT REFERENCES representations(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL,
    UNIQUE(source_id, kind, checksum)
) STRICT;

CREATE UNIQUE INDEX one_selected_representation_kind
ON representations(source_id, kind) WHERE selected = 1;

CREATE TABLE source_units (
    id TEXT PRIMARY KEY,
    representation_id TEXT NOT NULL REFERENCES representations(id) ON DELETE RESTRICT,
    unit_key TEXT NOT NULL,
    locator_json TEXT NOT NULL CHECK (json_valid(locator_json)),
    processing_state TEXT NOT NULL CHECK (processing_state IN
        ('processing', 'partially_usable', 'failed', 'corrected', 'superseded', 'usable')),
    created_at TEXT NOT NULL,
    UNIQUE(representation_id, unit_key)
) STRICT;

CREATE TABLE research_workspaces (
    id TEXT PRIMARY KEY,
    workspace_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE workspace_revisions (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL REFERENCES research_workspaces(id) ON DELETE RESTRICT,
    content_hash TEXT NOT NULL,
    reviewed_context_json TEXT NOT NULL CHECK (json_valid(reviewed_context_json)),
    created_at TEXT NOT NULL,
    UNIQUE(workspace_id, content_hash)
) STRICT;

CREATE TABLE workspace_memberships (
    revision_id TEXT NOT NULL REFERENCES workspace_revisions(id) ON DELETE RESTRICT,
    record_kind TEXT NOT NULL,
    record_id TEXT NOT NULL,
    inclusion_state TEXT NOT NULL CHECK (inclusion_state IN ('included', 'excluded')),
    reason TEXT,
    PRIMARY KEY(revision_id, record_kind, record_id)
) STRICT;
