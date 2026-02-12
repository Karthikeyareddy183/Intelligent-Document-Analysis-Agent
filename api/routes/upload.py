"""Upload endpoint — handles document file uploads."""

import time
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends

from api.auth import get_current_user
from api.schemas.responses import UploadResponse, ErrorResponse
from core.constants import ALLOWED_CONTENT_TYPES
from core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


def _process_in_thread(orchestrator, file_path: str, document_id: str, filename: str, user_id: str = None):
    """Run document processing in a separate thread so it doesn't block the event loop."""
    try:
        orchestrator.process_document(
            file_path=file_path,
            document_id=document_id,
            filename=filename,
            user_id=user_id,
        )
    except Exception as e:
        logger.error("Background processing failed", extra={"document_id": document_id, "error": str(e)})


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
    summary="Upload a document for processing",
    description="Upload a PDF or image file. The document will be processed in the background.",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF, JPG, PNG, or WEBP file"),
    user_id: str = Depends(get_current_user),
):
    services = request.app.state.services
    config = request.app.state.config

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unsupported file type: {file.content_type}",
                "supported_formats": list(ALLOWED_CONTENT_TYPES.values()),
            },
        )

    # Read and validate file size
    contents = await file.read()
    max_size = config.max_file_size_bytes
    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {config.MAX_FILE_SIZE_MB}MB",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Save file with user_id
    doc_service = services["document_service"]
    document_id, file_path = doc_service.save_upload(
        contents,
        file.filename,
        user_id=user_id,
        file_type=file.content_type,
        file_size=len(contents),
    )

    # Process document in a separate thread
    orchestrator = services["orchestrator"]
    thread = threading.Thread(
        target=_process_in_thread,
        args=(orchestrator, file_path, document_id, file.filename, user_id),
        daemon=True,
    )
    thread.start()

    logger.info(
        "Upload accepted",
        extra={"document_id": document_id, "file_name": file.filename, "size": len(contents)},
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        pages=0,
        chunks_created=0,
        uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
