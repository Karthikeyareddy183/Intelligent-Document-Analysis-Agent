"""Unit tests for the vector database service."""

import pytest
from services.vector_db_service import ChromaDBService


@pytest.fixture
def chroma_svc(test_config):
    return ChromaDBService(test_config)


class TestAddDocuments:
    def test_add_single_document(self, chroma_svc):
        docs = [{
            "id": "chunk-1",
            "embedding": [0.1] * 384,
            "text": "Q3 revenue was $4.2 billion",
            "document_id": "doc-1",
            "filename": "report.pdf",
            "page_number": 5,
            "chunk_index": 0,
            "content_type": "text",
        }]
        count = chroma_svc.add_documents(docs)
        assert count == 1
        assert chroma_svc.get_document_count() == 1

    def test_add_multiple_documents(self, chroma_svc):
        docs = [
            {
                "id": f"chunk-{i}",
                "embedding": [0.1 * (i + 1)] * 384,
                "text": f"Content for chunk {i}",
                "document_id": "doc-1",
                "filename": "test.pdf",
                "page_number": i + 1,
                "chunk_index": i,
                "content_type": "text",
            }
            for i in range(5)
        ]
        count = chroma_svc.add_documents(docs)
        assert count == 5
        assert chroma_svc.get_document_count() == 5

    def test_add_empty_list(self, chroma_svc):
        count = chroma_svc.add_documents([])
        assert count == 0


class TestSearch:
    def test_search_returns_results(self, chroma_svc):
        chroma_svc.add_documents([{
            "id": "c1",
            "embedding": [0.5] * 384,
            "text": "Revenue data for Q3",
            "document_id": "doc-1",
            "filename": "report.pdf",
            "page_number": 1,
            "chunk_index": 0,
            "content_type": "text",
        }])
        results = chroma_svc.search([0.5] * 384, top_k=1)
        assert len(results) == 1
        assert "Revenue" in results[0]["text"]
        assert results[0]["relevance_score"] > 0

    def test_search_respects_top_k(self, chroma_svc):
        for i in range(10):
            chroma_svc.add_documents([{
                "id": f"c-{i}",
                "embedding": [0.1 * (i + 1)] * 384,
                "text": f"Chunk {i}",
                "document_id": "doc-1",
                "filename": "test.pdf",
                "page_number": 1,
                "chunk_index": i,
                "content_type": "text",
            }])
        results = chroma_svc.search([0.5] * 384, top_k=3)
        assert len(results) == 3

    def test_search_with_document_filter(self, chroma_svc):
        chroma_svc.add_documents([
            {
                "id": "c1",
                "embedding": [0.1] * 384,
                "text": "Doc 1 content",
                "document_id": "doc-1",
                "filename": "a.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "content_type": "text",
            },
            {
                "id": "c2",
                "embedding": [0.2] * 384,
                "text": "Doc 2 content",
                "document_id": "doc-2",
                "filename": "b.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "content_type": "text",
            },
        ])
        results = chroma_svc.search(
            [0.1] * 384, top_k=5, filters={"document_id": "doc-1"}
        )
        assert all(r["metadata"]["document_id"] == "doc-1" for r in results)

    def test_search_empty_collection(self, chroma_svc):
        results = chroma_svc.search([0.5] * 384, top_k=5)
        assert results == []


class TestDeleteDocument:
    def test_delete_removes_all_chunks(self, chroma_svc):
        for i in range(3):
            chroma_svc.add_documents([{
                "id": f"c-{i}",
                "embedding": [0.1] * 384,
                "text": f"Chunk {i}",
                "document_id": "doc-to-delete",
                "filename": "test.pdf",
                "page_number": 1,
                "chunk_index": i,
                "content_type": "text",
            }])
        assert chroma_svc.get_document_count() == 3

        chroma_svc.delete_document("doc-to-delete")
        assert chroma_svc.get_document_count() == 0
