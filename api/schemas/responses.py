"""Pydantic response models for API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SourceCitation(BaseModel):
    """A single source citation from the document."""

    document_id: str
    page: int
    section: str = ""
    chunk_text: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response for the /query endpoint."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[SourceCitation]
    processing_time_ms: int


class UploadResponse(BaseModel):
    """Response for the /upload endpoint."""

    document_id: str
    filename: str
    status: str
    pages: int = 0
    chunks_created: int = 0
    uploaded_at: str


class DocumentInfoResponse(BaseModel):
    """Response for the /documents/{id} endpoint."""

    document_id: str
    filename: str
    status: str
    pages: int = 0
    chunks_created: int = 0
    error: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response for listing all documents."""

    documents: list[DocumentInfoResponse]
    total: int


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str
    version: str
    agents: dict[str, str]
    vector_db: str
    document_count: int
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
