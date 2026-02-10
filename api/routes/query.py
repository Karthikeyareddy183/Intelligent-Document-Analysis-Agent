"""Query endpoint — handles document question answering."""

import time
from fastapi import APIRouter, HTTPException, Request

from api.schemas.requests import QueryRequest
from api.schemas.responses import QueryResponse, ErrorResponse
from core.logger import setup_logger
from core.exceptions import AgentOrchestrationError

logger = setup_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Ask a question about uploaded documents",
    description="Submit a natural language question. The system retrieves relevant context and generates an answer.",
)
async def query_documents(request: Request, body: QueryRequest):
    services = request.app.state.services
    orchestrator = services["orchestrator"]

    # Validate document exists if document_id provided
    if body.document_id:
        doc_service = services["document_service"]
        doc_info = doc_service.get_document_info(body.document_id)
        if doc_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {body.document_id}",
            )

    logger.info(
        "Query received",
        extra={"query": body.query[:100], "document_id": body.document_id},
    )

    try:
        result = orchestrator.answer_query(
            query=body.query,
            document_id=body.document_id,
            top_k=body.top_k,
        )

        sources = []
        for src in result.get("sources", []):
            sources.append({
                "document_id": src.get("document_id", ""),
                "page": src.get("page", 0),
                "section": src.get("section", ""),
                "chunk_text": src.get("chunk_text", ""),
                "relevance_score": src.get("relevance_score", 0.0),
            })

        return QueryResponse(
            answer=result["answer"],
            confidence=result.get("confidence", 0.5),
            sources=sources,
            processing_time_ms=result.get("processing_time_ms", 0),
        )

    except AgentOrchestrationError as e:
        logger.error("Query failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e.message}")
    except Exception as e:
        logger.error("Unexpected query error", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Internal error processing query")
