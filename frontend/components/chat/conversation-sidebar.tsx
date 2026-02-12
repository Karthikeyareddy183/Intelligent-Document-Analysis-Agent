"use client";

import { Plus, MessageSquare, Trash2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  useConversations,
  useCreateConversation,
  useDeleteConversation,
} from "@/lib/hooks/use-conversations";
import { cn, formatDate, truncate } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

interface ConversationSidebarProps {
  documentId: string;
  activeConversationId: string | null;
  onSelect: (conversationId: string) => void;
  onNew: (conversationId: string) => void;
}

export function ConversationSidebar({
  documentId,
  activeConversationId,
  onSelect,
  onNew,
}: ConversationSidebarProps) {
  const { data, isLoading } = useConversations(documentId);
  const createConversation = useCreateConversation();
  const deleteConversation = useDeleteConversation();

  const conversations = data?.conversations || [];

  async function handleNewChat() {
    const conv = await createConversation.mutateAsync({
      documentId,
    });
    onNew(conv.id);
  }

  async function handleDelete(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    await deleteConversation.mutateAsync(id);
    if (activeConversationId === id) {
      onSelect("");
    }
  }

  return (
    <div className="flex h-full flex-col bg-muted/20">
      {/* New chat button */}
      <div className="p-3">
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start gap-2 h-9 font-medium bg-background/50 hover:bg-background"
          onClick={handleNewChat}
          disabled={createConversation.isPending}
        >
          {createConversation.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Plus className="h-4 w-4" />
          )}
          New Chat
        </Button>
      </div>

      {/* Divider */}
      <div className="px-3">
        <div className="h-px bg-border" />
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto p-2">
        {isLoading ? (
          <div className="space-y-2 p-1">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full rounded-lg" />
            ))}
          </div>
        ) : conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 px-3 text-center">
            <div className="rounded-xl bg-muted p-3 mb-3">
              <MessageSquare className="h-5 w-5 text-muted-foreground" />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Start a new chat to begin asking questions about this document
            </p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {conversations.map((conv: any) => (
              <button
                key={conv.id}
                onClick={() => onSelect(conv.id)}
                className={cn(
                  "group flex w-full items-start gap-2.5 rounded-lg p-2.5 text-left text-sm transition-all duration-150 cursor-pointer",
                  activeConversationId === conv.id
                    ? "bg-primary/8 border border-primary/15 shadow-sm"
                    : "hover:bg-accent/60 border border-transparent"
                )}
              >
                <MessageSquare
                  className={cn(
                    "mt-0.5 h-4 w-4 shrink-0 transition-colors",
                    activeConversationId === conv.id
                      ? "text-primary"
                      : "text-muted-foreground"
                  )}
                />
                <div className="flex-1 min-w-0">
                  <p
                    className={cn(
                      "font-medium truncate text-xs leading-tight",
                      activeConversationId === conv.id && "text-primary"
                    )}
                  >
                    {conv.title || "New Chat"}
                  </p>
                  {conv.preview && (
                    <p className="text-[11px] text-muted-foreground truncate mt-0.5">
                      {truncate(conv.preview, 60)}
                    </p>
                  )}
                  {conv.created_at && (
                    <p className="text-[10px] text-muted-foreground/60 mt-1">
                      {formatDate(conv.created_at)}
                    </p>
                  )}
                </div>
                <button
                  onClick={(e) => handleDelete(e, conv.id)}
                  className="hidden group-hover:flex h-6 w-6 items-center justify-center rounded-md hover:bg-destructive/10 transition-colors shrink-0"
                  aria-label="Delete conversation"
                >
                  <Trash2 className="h-3 w-3 text-muted-foreground hover:text-destructive transition-colors" />
                </button>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
