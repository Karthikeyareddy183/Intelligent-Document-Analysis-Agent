"""Conversation endpoints — manage chat sessions and messages."""

from fastapi import APIRouter, HTTPException, Request, Depends, Query
from typing import Optional

from api.auth import get_current_user
from api.schemas.responses import (
    ConversationResponse,
    ConversationListResponse,
    ConversationListItem,
    MessageResponse,
    ErrorResponse,
)
from core.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List conversations",
    description="List all conversations for the current user, optionally filtered by document.",
)
async def list_conversations(
    request: Request,
    document_id: Optional[str] = Query(default=None),
    user_id: str = Depends(get_current_user),
):
    conv_service = request.app.state.services.get("conversation_service")
    if not conv_service:
        raise HTTPException(status_code=503, detail="Conversation service not available")

    conversations = conv_service.list_conversations(user_id, document_id=document_id)

    items = [
        ConversationListItem(
            id=c["id"],
            document_id=c["document_id"],
            title=c["title"],
            preview=c.get("preview", ""),
            created_at=c["created_at"],
            updated_at=c["updated_at"],
        )
        for c in conversations
    ]

    return ConversationListResponse(conversations=items, total=len(items))


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get conversation with messages",
    description="Retrieve a conversation and all its messages.",
)
async def get_conversation(
    request: Request,
    conversation_id: str,
    user_id: str = Depends(get_current_user),
):
    conv_service = request.app.state.services.get("conversation_service")
    if not conv_service:
        raise HTTPException(status_code=503, detail="Conversation service not available")

    conversation = conv_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Verify ownership
    if conversation["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conversation["id"],
        user_id=conversation["user_id"],
        document_id=conversation["document_id"],
        title=conversation["title"],
        messages=[
            MessageResponse(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                sources=m.get("sources"),
                created_at=m["created_at"],
            )
            for m in conversation.get("messages", [])
        ],
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
    )


@router.post(
    "/conversations",
    response_model=ConversationListItem,
    summary="Create a new conversation",
    description="Create a new chat session for a document.",
)
async def create_conversation(
    request: Request,
    body: dict,
    user_id: str = Depends(get_current_user),
):
    conv_service = request.app.state.services.get("conversation_service")
    if not conv_service:
        raise HTTPException(status_code=503, detail="Conversation service not available")

    document_id = body.get("document_id")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")

    title = body.get("title", "New Chat")

    conversation = conv_service.create_conversation(user_id, document_id, title)

    return ConversationListItem(
        id=conversation["id"],
        document_id=conversation["document_id"],
        title=conversation["title"],
        preview="",
        created_at=conversation["created_at"],
        updated_at=conversation["updated_at"],
    )


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete a conversation",
    description="Delete a conversation and all its messages.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_conversation(
    request: Request,
    conversation_id: str,
    user_id: str = Depends(get_current_user),
):
    conv_service = request.app.state.services.get("conversation_service")
    if not conv_service:
        raise HTTPException(status_code=503, detail="Conversation service not available")

    # Verify ownership
    conversation = conv_service.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conversation["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv_service.delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}
