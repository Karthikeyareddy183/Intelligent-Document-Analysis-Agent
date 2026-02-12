"""Document info endpoints."""

import os
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import FileResponse

from api.auth import get_current_user
from api.schemas.responses import DocumentInfoResponse, DocumentListResponse, ErrorResponse
from core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get(
    "/documents/{document_id}",
    response_model=DocumentInfoResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get document information",
    description="Retrieve processing status and metadata for a specific document.",
)
async def get_document(
    request: Request,
    document_id: str,
    user_id: str = Depends(get_current_user),
):
    doc_service = request.app.state.services["document_service"]
    doc_info = doc_service.get_document_info(document_id)

    if doc_info is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    # Verify ownership
    if doc_info.get("user_id") and str(doc_info["user_id"]) != user_id:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    return DocumentInfoResponse(
        document_id=doc_info["document_id"],
        filename=doc_info["filename"],
        status=doc_info["status"],
        pages=doc_info.get("pages", 0),
        chunks_created=doc_info.get("chunks_created", 0),
        error=doc_info.get("error"),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="Retrieve a list of all uploaded documents for the current user.",
)
async def list_documents(
    request: Request,
    user_id: str = Depends(get_current_user),
):
    doc_service = request.app.state.services["document_service"]
    documents = doc_service.list_documents(user_id=user_id)

    items = [
        DocumentInfoResponse(
            document_id=doc["document_id"],
            filename=doc["filename"],
            status=doc["status"],
            pages=doc.get("pages", 0),
            chunks_created=doc.get("chunks_created", 0),
            error=doc.get("error"),
        )
        for doc in documents
    ]

    return DocumentListResponse(documents=items, total=len(items))


@router.delete(
    "/documents/{document_id}",
    summary="Delete a document",
    description="Delete a document, its chunks, and associated conversations.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_document(
    request: Request,
    document_id: str,
    user_id: str = Depends(get_current_user),
):
    services = request.app.state.services
    doc_service = services["document_service"]

    # Verify document exists and belongs to user
    doc_info = doc_service.get_document_info(document_id)
    if doc_info is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
    if doc_info.get("user_id") and str(doc_info["user_id"]) != user_id:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    # Delete chunks from vector DB
    vector_db = services["vector_db"]
    try:
        vector_db.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Failed to delete chunks from vector DB: {e}")

    # Delete document metadata + local file
    deleted = doc_service.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete document")

    return {"status": "deleted", "document_id": document_id}


@router.get(
    "/documents/{document_id}/file",
    summary="Download document file",
    description="Serve the original uploaded file for PDF viewing.",
    responses={404: {"model": ErrorResponse}},
)
async def get_document_file(
    request: Request,
    document_id: str,
    user_id: str = Depends(get_current_user),
):
    doc_service = request.app.state.services["document_service"]
    doc_info = doc_service.get_document_info(document_id)

    if doc_info is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc_info.get("user_id") and str(doc_info["user_id"]) != user_id:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = doc_info.get("storage_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    media_type = doc_info.get("file_type", "application/octet-stream")
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=doc_info.get("filename", "document"),
    )
