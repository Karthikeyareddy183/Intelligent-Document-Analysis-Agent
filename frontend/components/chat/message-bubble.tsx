"use client";

import { cn } from "@/lib/utils";
import { SourceCitation } from "@/components/chat/source-citation";
import { User, Sparkles } from "lucide-react";

interface MessageBubbleProps {
  role: "user" | "assistant";
  content: string;
  sources?: any[];
  isStreaming?: boolean;
  onPageClick?: (page: number) => void;
}

export function MessageBubble({
  role,
  content,
  sources,
  isStreaming,
  onPageClick,
}: MessageBubbleProps) {
  const isUser = role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 py-5 animate-fade-in",
        isUser && "flex-row-reverse"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg shadow-sm",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-primary/10"
        )}
      >
        {isUser ? (
          <User className="h-4 w-4" />
        ) : (
          <Sparkles className="h-4 w-4 text-primary" />
        )}
      </div>

      {/* Content */}
      <div
        className={cn(
          "max-w-[80%] space-y-2",
          isUser && "flex flex-col items-end"
        )}
      >
        {/* Label */}
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider px-1">
          {isUser ? "You" : "DocAI"}
        </span>

        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed",
            isUser
              ? "bg-primary text-primary-foreground rounded-tr-sm"
              : "bg-muted/70 border border-border/50 rounded-tl-sm"
          )}
        >
          {content ? (
            <div className="whitespace-pre-wrap break-words">{content}</div>
          ) : isStreaming ? (
            <div className="flex items-center gap-1.5 py-1">
              <span className="typing-dot h-2 w-2 rounded-full bg-primary/60" />
              <span className="typing-dot h-2 w-2 rounded-full bg-primary/60" />
              <span className="typing-dot h-2 w-2 rounded-full bg-primary/60" />
            </div>
          ) : null}

          {/* Streaming cursor */}
          {isStreaming && content && (
            <span className="inline-block w-0.5 h-4 bg-primary/60 animate-pulse ml-0.5 align-middle" />
          )}
        </div>

        {/* Sources */}
        {sources && sources.length > 0 && !isStreaming && (
          <SourceCitation sources={sources} onPageClick={onPageClick} />
        )}
      </div>
    </div>
  );
}
