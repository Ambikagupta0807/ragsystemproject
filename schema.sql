-- The whole database. Four tables.
--
-- Everything lives in one Postgres: the people, the documents and their
-- versions, the chunks with their vectors and keyword index, and the audit
-- trail. One connection string, one backup, one place to look.

CREATE EXTENSION IF NOT EXISTS vector;

-- Who is asking. In a real company this comes from your identity provider,
-- and the rest of the system does not change.
CREATE TABLE IF NOT EXISTS users (
    username    TEXT PRIMARY KEY,
    display_name TEXT    NOT NULL,
    department  TEXT     NOT NULL,
    is_employee BOOLEAN  NOT NULL
);

-- One row per uploaded document. A document is identified by its title and
-- version, never by its file name, because the same policy can arrive as
-- policy.md, policy-final.md or policy(2).md. The content hash is unique, so
-- uploading the same text twice is a no-op whatever the file was called.
CREATE TABLE IF NOT EXISTS documents (
    id             BIGSERIAL PRIMARY KEY,
    source         TEXT        NOT NULL,
    title          TEXT        NOT NULL,
    department     TEXT        NOT NULL,
    visibility     TEXT        NOT NULL,
    version        INT         NOT NULL DEFAULT 1,
    status         TEXT        NOT NULL DEFAULT 'active',
    effective_from DATE        NOT NULL,
    updated_on     DATE        NOT NULL,
    content_hash   TEXT        NOT NULL UNIQUE,
    indexed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (title, version)
);

-- One row per chunk. The vector and the keyword index sit next to the text.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT       NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    section     TEXT         NOT NULL,
    content     TEXT         NOT NULL,
    embedding   VECTOR(1536) NOT NULL,
    -- Postgres builds and maintains this itself, so the keyword index can
    -- never drift away from the text it indexes.
    search_text TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Who asked what, what they were shown, and whether it was answered. An
-- assistant that reads private documents needs this to be auditable.
CREATE TABLE IF NOT EXISTS audit_events (
    id        BIGSERIAL PRIMARY KEY,
    at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    username  TEXT        NOT NULL,
    question  TEXT        NOT NULL,
    answered  BOOLEAN     NOT NULL,
    grounded  BOOLEAN     NOT NULL,
    sources   TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_search_idx
    ON chunks USING gin (search_text);

CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS audit_events_at_idx
    ON audit_events (at DESC);

-- The people in the demo. Alice cannot read HR documents, Bob can.
INSERT INTO users (username, display_name, department, is_employee) VALUES
    ('alice', 'Alice',  'engineering', true),
    ('bob',   'Bob',    'hr',          true),
    ('carol', 'Carol',  'finance',     true),
    ('guest', 'Guest',  'none',        false)
ON CONFLICT (username) DO NOTHING;
