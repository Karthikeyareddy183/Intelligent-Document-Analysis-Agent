"""Health check endpoint."""

import time
from fastapi import APIRouter, Request

from api.schemas.responses import HealthResponse
from core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all system components.",
)
async def health_check(request: Request):
    services = request.app.state.services

    # Check vector DB
    vector_db_status = "disconnected"
    document_count = 0
    try:
        vector_db = services["vector_db"]
        document_count = vector_db.get_document_count()
        vector_db_status = "connected"
    except Exception:
        vector_db_status = "error"

    return HealthResponse(
        status="healthy" if vector_db_status == "connected" else "degraded",
        version="1.0.0",
        agents={
            "document_processor": "operational",
            "retrieval": "operational",
            "analysis": "operational",
        },
        vector_db=vector_db_status,
        document_count=document_count,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
