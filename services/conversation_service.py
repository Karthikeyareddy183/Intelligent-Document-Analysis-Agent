"""Conversation service — manages chat sessions and messages via Supabase."""

from typing import Optional

from core.config import Settings
from core.logger import setup_logger

logger = setup_logger(__name__)


class ConversationService:
    """CRUD operations for conversations and messages stored in Supabase."""

    def __init__(self, config: Settings):
        from supabase import create_client
        self._supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)

    def create_conversation(
        self, user_id: str, document_id: str, title: str = "New Chat"
    ) -> dict:
        """Create a new conversation. Returns the conversation dict."""
        result = self._supabase.table("conversations").insert({
            "user_id": user_id,
            "document_id": document_id,
            "title": title,
        }).execute()

        row = result.data[0]
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_conversations(
        self, user_id: str, document_id: Optional[str] = None
    ) -> list[dict]:
        """List conversations for a user, optionally filtered by document.

        Includes a preview from the first user message (truncated to 100 chars).
        """
        query = (
            self._supabase.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
        )
        if document_id:
            query = query.eq("document_id", document_id)

        result = query.execute()

        conversations = []
        for row in result.data:
            # Fetch first user message as preview
            preview = ""
            try:
                msg_result = (
                    self._supabase.table("messages")
                    .select("content")
                    .eq("conversation_id", row["id"])
                    .eq("role", "user")
                    .order("created_at")
                    .limit(1)
                    .execute()
                )
                if msg_result.data:
                    preview = msg_result.data[0]["content"][:100]
            except Exception:
                pass

            conversations.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "document_id": row["document_id"],
                "title": row["title"],
                "preview": preview,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })

        return conversations

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a conversation with all its messages."""
        conv_result = (
            self._supabase.table("conversations")
            .select("*")
            .eq("id", conversation_id)
            .limit(1)
            .execute()
        )

        if not conv_result.data:
            return None

        row = conv_result.data[0]

        # Fetch all messages ordered by created_at
        msg_result = (
            self._supabase.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .execute()
        )

        messages = [
            {
                "id": msg["id"],
                "role": msg["role"],
                "content": msg["content"],
                "sources": msg.get("sources"),
                "created_at": msg["created_at"],
            }
            for msg in msg_result.data
        ]

        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "document_id": row["document_id"],
            "title": row["title"],
            "messages": messages,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        sources: Optional[list] = None,
    ) -> dict:
        """Add a message to a conversation and update conversation.updated_at."""
        insert_data = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "content": content,
        }
        if sources is not None:
            insert_data["sources"] = sources

        result = self._supabase.table("messages").insert(insert_data).execute()

        # Touch conversation updated_at
        self._supabase.table("conversations").update(
            {"updated_at": "now()"}
        ).eq("id", conversation_id).execute()

        msg = result.data[0]
        return {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "sources": msg.get("sources"),
            "created_at": msg["created_at"],
        }

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        """Update a conversation's title."""
        self._supabase.table("conversations").update(
            {"title": title}
        ).eq("id", conversation_id).execute()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation (messages cascade automatically)."""
        self._supabase.table("conversations").delete().eq(
            "id", conversation_id
        ).execute()
