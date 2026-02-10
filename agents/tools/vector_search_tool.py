"""CrewAI tool for semantic search in the vector database."""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

from core.logger import setup_logger

logger = setup_logger(__name__)


class VectorSearchInput(BaseModel):
    query: str = Field(description="Search query text")
    document_id: Optional[str] = Field(
        default=None, description="Optional document ID to filter results"
    )
    top_k: int = Field(default=5, description="Number of results to return")


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = (
        "Search the vector database for document chunks most relevant to a query. "
        "Returns top-k results with text content, metadata, and relevance scores."
    )
    args_schema: type[BaseModel] = VectorSearchInput

    _embedding_service: object = None
    _vector_db_service: object = None

    def __init__(self, vector_db_service, embedding_service, **kwargs):
        super().__init__(**kwargs)
        self._vector_db_service = vector_db_service
        self._embedding_service = embedding_service

    def _run(
        self, query: str, document_id: Optional[str] = None, top_k: int = 5
    ) -> dict:
        logger.info(
            "Vector search",
            extra={"query": query[:100], "top_k": top_k, "filter": document_id},
        )

        try:
            # Generate query embedding
            query_embedding = self._embedding_service.generate_query_embedding(query)

            # Search vector DB
            filters = {"document_id": document_id} if document_id else None
            results = self._vector_db_service.search(query_embedding, top_k, filters)

            logger.info("Search complete", extra={"results": len(results)})

            return {
                "status": "success",
                "query": query,
                "results": results,
                "result_count": len(results),
            }

        except Exception as e:
            logger.error("Vector search failed", extra={"error": str(e)})
            return {"status": "error", "error": str(e), "results": []}
