"""Shared test fixtures."""

import os
import pytest
from pathlib import Path

from core.config import Settings


@pytest.fixture
def test_config(tmp_path):
    """Config pointing to temp directories."""
    return Settings(
        OPENAI_API_KEY="sk-test-key",
        ANTHROPIC_API_KEY="sk-ant-test-key",
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o",
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        VECTOR_DB_TYPE="chromadb",
        CHROMA_PERSIST_DIR=str(tmp_path / "chroma_db"),
        CHROMA_COLLECTION_NAME="test_documents",
        CHUNK_SIZE=200,
        CHUNK_OVERLAP=50,
        MAX_FILE_SIZE_MB=10,
        LOG_LEVEL="DEBUG",
    )


@pytest.fixture
def sample_text():
    """Sample document text for testing."""
    return (
        "Quarterly Financial Report Q3 2025\n\n"
        "Revenue for Q3 was $4.2 billion, representing a 15% increase year-over-year. "
        "The growth was primarily driven by cloud services, which saw a 25% increase. "
        "Operating expenses totaled $3.1 billion, resulting in an operating margin of 26%.\n\n"
        "Key Highlights:\n"
        "- Cloud revenue: $1.8 billion (up 25%)\n"
        "- Enterprise licenses: $1.2 billion (up 8%)\n"
        "- Professional services: $1.2 billion (up 12%)\n\n"
        "The company expects Q4 revenue to be between $4.5 and $4.7 billion."
    )


@pytest.fixture
def sample_pdf_path():
    """Path to test PDF fixture."""
    path = Path(__file__).parent / "fixtures" / "sample.pdf"
    if path.exists():
        return str(path)
    return None


@pytest.fixture
def sample_image_path():
    """Path to test image fixture."""
    path = Path(__file__).parent / "fixtures" / "sample_scan.png"
    if path.exists():
        return str(path)
    return None
