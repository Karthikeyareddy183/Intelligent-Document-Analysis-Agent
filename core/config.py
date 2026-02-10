"""Application configuration using Pydantic Settings."""

import json
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration — reads from .env file and environment variables."""

    # ─── LLM Providers ────────────────────────────
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_PROVIDER: str = Field(default="openrouter", description="openai | anthropic | openrouter")
    LLM_MODEL: str = Field(default="google/gemini-2.0-flash-001", description="Model identifier")

    # ─── Embeddings ───────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ─── Vector Database ──────────────────────────
    VECTOR_DB_TYPE: str = Field(default="chromadb", description="chromadb | pinecone")
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = ""
    PINECONE_INDEX_NAME: str = "doc-analysis"

    # ─── Document Processing ─────────────────────
    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 750
    CHUNK_OVERLAP: int = 150
    OCR_LANGUAGE: str = "eng"

    # ─── API Configuration ───────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    RATE_LIMIT_PER_MINUTE: int = 30

    # ─── Logging ─────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once per process."""
    return Settings()
