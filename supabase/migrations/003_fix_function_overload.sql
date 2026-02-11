-- ============================================================
-- Migration 003: Fix function overload — drop old signatures
-- Run this in the Supabase SQL Editor
-- ============================================================

-- Drop the OLD match_documents (without filter_page_number)
DROP FUNCTION IF EXISTS match_documents(
    vector(384), FLOAT, INT, TEXT
);

-- Drop the OLD hybrid_search (without filter_page_number)
DROP FUNCTION IF EXISTS hybrid_search(
    TEXT, vector(384), INT, FLOAT, FLOAT, INT, TEXT
);

-- Now recreate both WITH the page filter (from 002)
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5,
    filter_document_id TEXT DEFAULT NULL,
    filter_page_number INT DEFAULT NULL
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
        AND (1 - (dc.embedding <=> query_embedding)) > match_threshold
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT LEAST(match_count, 200);
END;
$$;

CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(384),
    match_count INT DEFAULT 5,
    semantic_weight FLOAT DEFAULT 1.0,
    full_text_weight FLOAT DEFAULT 1.0,
    rrf_k INT DEFAULT 50,
    filter_document_id TEXT DEFAULT NULL,
    filter_page_number INT DEFAULT NULL
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
