-- ============================================================
-- Supabase pgvector Setup for Intelligent Document Analysis Agent
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor)
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector
WITH SCHEMA extensions;

-- ============================================================
-- 2. Create document_chunks table
--    Stores chunked content with 384-dim embeddings (all-MiniLM-L6-v2)
--    and full-text search column for hybrid search
-- ============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id              BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    content         TEXT NOT NULL,
    document_id     TEXT NOT NULL,
    file_name       TEXT DEFAULT '',
    page_number     INTEGER DEFAULT 0,
    section_title   TEXT DEFAULT '',
    chunk_index     INTEGER DEFAULT 0,
    content_type    TEXT DEFAULT 'text' CHECK (content_type IN ('text', 'table', 'image')),
    char_count      INTEGER DEFAULT 0,
    embedding       vector(384),
    fts             TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 3. Create indexes for fast search
-- ============================================================

-- HNSW index for semantic search (cosine similarity)
-- Parameters match the previous ChromaDB config: M=16, ef_construction=200
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_fts
ON document_chunks USING gin (fts);

-- B-tree index on document_id for fast filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON document_chunks (document_id);

-- B-tree index on content_type for filtered searches
CREATE INDEX IF NOT EXISTS idx_chunks_content_type
ON document_chunks (content_type);

-- ============================================================
-- 4. Semantic search function (match_documents)
--    Called via supabase.rpc("match_documents", {...})
-- ============================================================
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5,
    filter_document_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    id              BIGINT,
    content         TEXT,
    document_id     TEXT,
    file_name       TEXT,
    page_number     INTEGER,
    section_title   TEXT,
    chunk_index     INTEGER,
    content_type    TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.content,
        dc.document_id,
        dc.file_name,
        dc.page_number,
        dc.section_title,
        dc.chunk_index,
        dc.content_type,
        (1 - (dc.embedding <=> query_embedding))::FLOAT AS similarity
    FROM document_chunks dc
    WHERE
        (filter_document_id IS NULL OR dc.document_id = filter_document_id)
        AND (1 - (dc.embedding <=> query_embedding)) > match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT LEAST(match_count, 200);
END;
$$;

-- ============================================================
-- 5. Hybrid search function (semantic + full-text with RRF)
--    Combines pgvector cosine similarity with tsvector keyword search
--    using Reciprocal Rank Fusion for best results
-- ============================================================
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(384),
    match_count INT DEFAULT 5,
    semantic_weight FLOAT DEFAULT 1.0,
    full_text_weight FLOAT DEFAULT 1.0,
    rrf_k INT DEFAULT 50,
    filter_document_id TEXT DEFAULT NULL
)
RETURNS TABLE (
    id              BIGINT,
    content         TEXT,
    document_id     TEXT,
    file_name       TEXT,
    page_number     INTEGER,
    section_title   TEXT,
    chunk_index     INTEGER,
    content_type    TEXT,
    score           FLOAT
)
LANGUAGE sql STABLE
AS $$
    WITH full_text AS (
        SELECT
            dc.id,
            ROW_NUMBER() OVER (
                ORDER BY ts_rank_cd(dc.fts, websearch_to_tsquery(query_text)) DESC
            ) AS rank_ix
        FROM document_chunks dc
        WHERE
            dc.fts @@ websearch_to_tsquery(query_text)
            AND (filter_document_id IS NULL OR dc.document_id = filter_document_id)
        ORDER BY rank_ix
        LIMIT LEAST(match_count, 30) * 2
    ),
    semantic AS (
        SELECT
            dc.id,
            ROW_NUMBER() OVER (
                ORDER BY dc.embedding <=> query_embedding
            ) AS rank_ix
        FROM document_chunks dc
        WHERE (filter_document_id IS NULL OR dc.document_id = filter_document_id)
        ORDER BY rank_ix
        LIMIT LEAST(match_count, 30) * 2
    )
    SELECT
        dc.id,
        dc.content,
        dc.document_id,
        dc.file_name,
        dc.page_number,
        dc.section_title,
        dc.chunk_index,
        dc.content_type,
        (
            COALESCE(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight +
            COALESCE(1.0 / (rrf_k + sem.rank_ix), 0.0) * semantic_weight
        )::FLOAT AS score
    FROM full_text ft
    FULL OUTER JOIN semantic sem ON ft.id = sem.id
    JOIN document_chunks dc ON COALESCE(ft.id, sem.id) = dc.id
    ORDER BY score DESC
    LIMIT LEAST(match_count, 30);
$$;

-- ============================================================
-- 6. Set HNSW search ef parameter for better recall at query time
-- ============================================================
ALTER DATABASE postgres SET hnsw.ef_search = 100;

-- ============================================================
-- 7. Create storage bucket for original uploaded files
--    (Run this or create via Supabase Dashboard > Storage)
-- ============================================================
-- INSERT INTO storage.buckets (id, name, public)
-- VALUES ('documents', 'documents', false)
-- ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- DONE! Your Supabase project is ready for the RAG pipeline.
-- Set these in your .env file:
--   SUPABASE_URL=https://your-project.supabase.co
--   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
--   VECTOR_DB_TYPE=supabase
-- ============================================================
