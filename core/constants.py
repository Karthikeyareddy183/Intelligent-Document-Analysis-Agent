"""Application-wide constants and enumerations."""

from enum import Enum


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorDBType(str, Enum):
    CHROMADB = "chromadb"
    PINECONE = "pinecone"


# Supported file types for upload
ALLOWED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png", "webp"}

# API versioning
API_V1_PREFIX = "/api/v1"

# Defaults
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
DEFAULT_CONFIDENCE = 0.5
