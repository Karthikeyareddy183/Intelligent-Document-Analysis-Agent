-- ============================================================
-- Migration 005: Auth & Persistence Tables
--
-- Creates documents, conversations, messages tables with RLS.
-- Adds user_id to document_chunks.
-- Updates match_documents and hybrid_search RPCs with user filter.
--
-- Run this in the Supabase SQL Editor AFTER enabling
-- Supabase Auth (Email provider) in Dashboard > Authentication.
-- ============================================================

-- ─── 1. Documents table (replaces in-memory dict) ───────────

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document_id     TEXT UNIQUE NOT NULL,
    filename        TEXT NOT NULL,
    file_type       TEXT,
    file_size       BIGINT,
    storage_path    TEXT,
    status          TEXT DEFAULT 'processing' CHECK (status IN ('processing', 'completed', 'failed')),
    pages           INT DEFAULT 0,
    chunks_created  INT DEFAULT 0,
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);

-- ─── 2. Conversations table ─────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    document_id     TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    title           TEXT DEFAULT 'New Chat',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_document ON conversations(user_id, document_id);

-- ─── 3. Messages table ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources         JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);

-- ─── 4. Add user_id to document_chunks ──────────────────────

ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON document_chunks(user_id);

-- ─── 5. Enable RLS on all tables ────────────────────────────

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

-- Documents policies
CREATE POLICY "Users can view own documents"
    ON documents FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own documents"
    ON documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own documents"
    ON documents FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own documents"
    ON documents FOR DELETE
    USING (auth.uid() = user_id);

-- Conversations policies
CREATE POLICY "Users can view own conversations"
    ON conversations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own conversations"
    ON conversations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own conversations"
    ON conversations FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own conversations"
    ON conversations FOR DELETE
    USING (auth.uid() = user_id);

-- Messages policies
CREATE POLICY "Users can view own messages"
    ON messages FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own messages"
    ON messages FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Document chunks policies (service role bypasses RLS, but add for safety)
CREATE POLICY "Users can view own chunks"
    ON document_chunks FOR SELECT
    USING (user_id IS NULL OR user_id = auth.uid()::TEXT);

CREATE POLICY "Service can manage chunks"
    ON document_chunks FOR ALL
    USING (true)
    WITH CHECK (true);

-- ─── 6. Updated RPCs with user_id filter ────────────────────

-- Drop existing functions to recreate with new parameter
DROP FUNCTION IF EXISTS match_documents(vector(1536), FLOAT, INT, TEXT, INT);
DROP FUNCTION IF EXISTS hybrid_search(TEXT, vector(1536), INT, FLOAT, FLOAT, INT, TEXT, INT);

CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(1536),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5,
    filter_document_id TEXT DEFAULT NULL,
    filter_page_number INT DEFAULT NULL,
    filter_user_id TEXT DEFAULT NULL
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
        AND (filter_page_number IS NULL OR dc.page_number = filter_page_number)
        AND (filter_user_id IS NULL OR dc.user_id = filter_user_id)
        AND (1 - (dc.embedding <=> query_embedding)) > match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT LEAST(match_count, 200);
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1536),
    match_count INT DEFAULT 5,
    semantic_weight FLOAT DEFAULT 1.0,
    full_text_weight FLOAT DEFAULT 1.0,
    rrf_k INT DEFAULT 50,
    filter_document_id TEXT DEFAULT NULL,
    filter_page_number INT DEFAULT NULL,
    filter_user_id TEXT DEFAULT NULL
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
            AND (filter_page_number IS NULL OR dc.page_number = filter_page_number)
            AND (filter_user_id IS NULL OR dc.user_id = filter_user_id)
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
        WHERE
            (filter_document_id IS NULL OR dc.document_id = filter_document_id)
            AND (filter_page_number IS NULL OR dc.page_number = filter_page_number)
            AND (filter_user_id IS NULL OR dc.user_id = filter_user_id)
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

-- ─── 7. Auto-update updated_at trigger ──────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- DONE! Enable Supabase Auth (Email provider) in Dashboard,
-- then add JWT_SECRET and ANON_KEY to your .env file.
-- ============================================================
