"""Unit tests for the embedding service."""

import pytest
from services.embedding_service import EmbeddingService


@pytest.fixture
def embedding_svc(test_config):
    return EmbeddingService(test_config)


class TestChunking:
    def test_chunk_text_splits_long_text(self, embedding_svc, sample_text):
        chunks = embedding_svc.chunk_text(
            sample_text, {"document_id": "test-doc", "page_number": 1}
        )
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert chunk["metadata"]["document_id"] == "test-doc"

    def test_chunk_text_preserves_metadata(self, embedding_svc):
        text = "Short text that fits in one chunk."
        chunks = embedding_svc.chunk_text(
            text, {"document_id": "abc", "page_number": 3, "filename": "test.pdf"}
        )
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["document_id"] == "abc"
        assert chunks[0]["metadata"]["page_number"] == 3
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_chunk_text_empty_input(self, embedding_svc):
        chunks = embedding_svc.chunk_text("", {"document_id": "test"})
        assert chunks == []

    def test_chunk_text_whitespace_only(self, embedding_svc):
        chunks = embedding_svc.chunk_text("   \n\n  ", {"document_id": "test"})
        assert chunks == []


class TestEmbeddings:
    def test_generate_embeddings_correct_dimension(self, embedding_svc):
        texts = ["Hello world", "Test document content"]
        embeddings = embedding_svc.generate_embeddings(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384

    def test_generate_embeddings_empty_list(self, embedding_svc):
        embeddings = embedding_svc.generate_embeddings([])
        assert embeddings == []

    def test_generate_query_embedding(self, embedding_svc):
        embedding = embedding_svc.generate_query_embedding("test query")
        assert len(embedding) == 384
        assert isinstance(embedding[0], float)

    def test_embeddings_are_normalized(self, embedding_svc):
        """Normalized embeddings should have magnitude close to 1.0."""
        import math
        embedding = embedding_svc.generate_query_embedding("test query")
        magnitude = math.sqrt(sum(x**2 for x in embedding))
        assert abs(magnitude - 1.0) < 0.01

    def test_similar_texts_have_similar_embeddings(self, embedding_svc):
        """Semantically similar texts should produce similar embeddings."""
        emb1 = embedding_svc.generate_query_embedding("What was the revenue?")
        emb2 = embedding_svc.generate_query_embedding("How much money was earned?")
        emb3 = embedding_svc.generate_query_embedding("The cat sat on the mat")

        # Cosine similarity (embeddings are normalized, so dot product = cosine)
        sim_related = sum(a * b for a, b in zip(emb1, emb2))
        sim_unrelated = sum(a * b for a, b in zip(emb1, emb3))

        assert sim_related > sim_unrelated
