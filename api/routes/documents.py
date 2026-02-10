"""Document info endpoints."""

from fastapi import APIRouter, HTTPException, Request

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
async def get_document(request: Request, document_id: str):
    doc_service = request.app.state.services["document_service"]
    doc_info = doc_service.get_document_info(document_id)

    if doc_info is None:
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
    description="Retrieve a list of all uploaded documents with their processing status.",
)
async def list_documents(request: Request):
    doc_service = request.app.state.services["document_service"]
    documents = doc_service.list_documents()

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
