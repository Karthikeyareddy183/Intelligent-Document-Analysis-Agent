"""Unit tests for API endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport

from api.main import create_app


@pytest.fixture
def mock_services():
    """Mock all services for API testing."""
    services = {
        "document_service": MagicMock(),
        "embedding_service": MagicMock(),
        "vector_db": MagicMock(),
        "llm_provider": MagicMock(),
        "orchestrator": MagicMock(),
        "tools": {},
    }
    services["vector_db"].get_document_count.return_value = 42
    services["document_service"].list_documents.return_value = []
    return services


@pytest.fixture
async def client(mock_services, test_config):
    """Create test client with mocked services."""
    app = create_app()
    app.state.config = test_config
    app.state.services = mock_services

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "agents" in data
        assert data["document_count"] == 42


class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_rejects_invalid_type(self, client):
        response = await client.post(
            "/api/v1/upload",
            files={"file": ("test.exe", b"fake content", "application/octet-stream")},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_accepts_pdf(self, client, mock_services):
        mock_services["document_service"].save_upload.return_value = (
            "test-uuid",
            "/tmp/test.pdf",
        )

        response = await client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["document_id"] == "test-uuid"


class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_with_valid_request(self, client, mock_services):
        mock_services["orchestrator"].answer_query.return_value = {
            "answer": "Q3 revenue was $4.2 billion",
            "confidence": 0.9,
            "sources": [
                {
                    "document_id": "doc-1",
                    "page": 5,
                    "section": "Financials",
                    "chunk_text": "Revenue...",
                    "relevance_score": 0.95,
                }
            ],
            "processing_time_ms": 1200,
        }

        response = await client.post(
            "/api/v1/query",
            json={"query": "What was Q3 revenue?", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "revenue" in data["answer"].lower()
        assert data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_query_rejects_short_query(self, client):
        response = await client.post(
            "/api/v1/query",
            json={"query": "hi"},
        )
        assert response.status_code == 422  # Validation error


class TestDocumentEndpoint:
    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, client, mock_services):
        mock_services["document_service"].get_document_info.return_value = None
        response = await client.get("/api/v1/documents/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_documents(self, client, mock_services):
        mock_services["document_service"].list_documents.return_value = [
            {
                "document_id": "doc-1",
                "filename": "test.pdf",
                "status": "completed",
                "pages": 10,
                "chunks_created": 25,
            }
        ]
        response = await client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
