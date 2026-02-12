"""Query endpoints — document question answering with optional SSE streaming."""

import json
from fastapi import APIRouter, HTTPException, Request, Depends

from sse_starlette.sse import EventSourceResponse

from api.auth import get_current_user
from api.schemas.requests import QueryRequest, StreamQueryRequest
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
async def query_documents(
    request: Request,
    body: QueryRequest,
    user_id: str = Depends(get_current_user),
):
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
            user_id=user_id,
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


@router.post(
    "/query/stream",
    summary="Ask a question with streaming SSE response",
    description="Submit a question with optional chat history. Returns Server-Sent Events with tokens, sources, and done.",
)
async def query_stream(
    request: Request,
    body: StreamQueryRequest,
    user_id: str = Depends(get_current_user),
):
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
        "Streaming query received",
        extra={
            "query": body.query[:100],
            "document_id": body.document_id,
            "history_len": len(body.chat_history),
            "conversation_id": body.conversation_id,
        },
    )

    chat_history = [{"role": msg.role, "content": msg.content} for msg in body.chat_history]
    conversation_id = body.conversation_id
    conv_service = services.get("conversation_service")

    # Save user message to conversation if conversation_id provided
    if conversation_id and conv_service:
        try:
            conv_service.add_message(conversation_id, user_id, "user", body.query)
        except Exception as e:
            logger.warning(f"Failed to save user message: {e}")

    async def event_generator():
        full_response = []
        sources_data = []

        try:
            async for event in orchestrator.answer_query_stream(
                query=body.query,
                document_id=body.document_id,
                top_k=body.top_k,
                chat_history=chat_history,
                user_id=user_id,
            ):
                event_type = event["event"]
                data = event["data"]

                if event_type == "sources":
                    sources_data = data
                    yield {"event": "sources", "data": json.dumps(data)}
                elif event_type == "done":
                    # Save assistant message to conversation
                    if conversation_id and conv_service and full_response:
                        try:
                            conv_service.add_message(
                                conversation_id,
                                user_id,
                                "assistant",
                                "".join(full_response),
                                sources=sources_data or None,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save assistant message: {e}")
                    yield {"event": "done", "data": ""}
                elif event_type == "error":
                    yield {"event": "error", "data": data}
                else:
                    full_response.append(data)
                    yield {"event": "token", "data": data}
        except Exception as e:
            logger.error("Stream error", extra={"error": str(e)})
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())
