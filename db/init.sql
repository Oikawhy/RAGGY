CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE knowledge_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    logical_source_key TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    hash_sha256 TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    total_sections INTEGER,
    total_chunks INTEGER,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    UNIQUE (logical_source_key, version)
);

CREATE TABLE knowledge_sections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    section_num INTEGER NOT NULL,
    title TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('uk', 'en', 'mixed')),
    is_meta BOOLEAN DEFAULT FALSE,
    raw_content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_id, section_num),
    UNIQUE (id, source_id)
);

CREATE TABLE knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    section_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_tokens INTEGER NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('uk', 'en')),
    block_type TEXT DEFAULT 'paragraph',
    embedding vector(1024),
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', content)
    ) STORED,
    has_numbers BOOLEAN DEFAULT FALSE,
    has_disclaimer BOOLEAN DEFAULT FALSE,
    is_rule BOOLEAN DEFAULT FALSE,
    section_title TEXT NOT NULL,
    source_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (section_id, chunk_index),
    CONSTRAINT fk_chunks_section_source
        FOREIGN KEY (section_id, source_id)
        REFERENCES knowledge_sections(id, source_id)
        ON DELETE CASCADE
);

CREATE UNIQUE INDEX idx_chunks_content_dedup ON knowledge_chunks(section_id, content_hash);
CREATE INDEX idx_sources_hash ON knowledge_sources(hash_sha256);
CREATE INDEX idx_chunks_source ON knowledge_chunks(source_id);

CREATE TABLE active_corpus_sources (
    corpus_hash TEXT NOT NULL,
    source_id UUID NOT NULL REFERENCES knowledge_sources(id) ON DELETE CASCADE,
    activated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (corpus_hash, source_id)
);

CREATE INDEX idx_active_corpus_sources_source
    ON active_corpus_sources(source_id);

CREATE INDEX idx_chunks_embedding_hnsw
    ON knowledge_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200)
    WHERE embedding IS NOT NULL;

CREATE INDEX idx_chunks_content_fts
    ON knowledge_chunks USING gin (content_tsv);

CREATE INDEX idx_chunks_section ON knowledge_chunks(section_id);
CREATE INDEX idx_chunks_language ON knowledge_chunks(language);
CREATE INDEX idx_chunks_non_meta
    ON knowledge_chunks(id, section_id)
    WHERE is_rule = FALSE;
