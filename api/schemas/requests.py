"""Pydantic request models for API validation."""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request body for document querying."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question about document content",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Filter to a specific document (UUID). Searches all if not provided.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve",
    )
    include_images: bool = Field(
        default=True,
        description="Include visual content in analysis",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What was the Q3 revenue?",
                    "top_k": 5,
                    "include_images": True,
                }
            ]
        }
    }
