"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Sparkles, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MessageBubble } from "@/components/chat/message-bubble";
import { useSSEStream } from "@/lib/hooks/use-sse-stream";

interface ChatPanelProps {
  documentId: string;
  conversationId: string | null;
  onPageClick?: (page: number) => void;
  initialMessages?: any[];
}

export function ChatPanel({
  documentId,
  conversationId,
  onPageClick,
  initialMessages,
}: ChatPanelProps) {
  const { messages, isStreaming, sendMessage, setMessages } = useSSEStream(
    documentId,
    conversationId
  );
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load initial messages when conversation changes
  useEffect(() => {
    if (initialMessages && initialMessages.length > 0) {
      setMessages(
        initialMessages.map((m: any) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          sources: m.sources,
        }))
      );
    } else {
      setMessages([]);
    }
  }, [conversationId, initialMessages, setMessages]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const query = input.trim();
    if (!query || isStreaming) return;
    setInput("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    await sendMessage(query);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 150) + "px";
    }
  }, [input]);

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          /* Empty chat state */
          <div className="flex h-full items-center justify-center p-8">
            <div className="text-center space-y-4 max-w-md animate-fade-in">
              <div className="flex justify-center">
                <div className="relative">
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/10 to-blue-500/10 border border-primary/10">
                    <MessageSquare className="h-8 w-8 text-primary/60" />
                  </div>
                  <div className="absolute -bottom-1 -right-1 flex h-7 w-7 items-center justify-center rounded-full bg-primary shadow-lg">
                    <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
                  </div>
                </div>
              </div>
              <div>
                <h3 className="font-heading font-semibold text-lg mb-1.5">
                  Ask about this document
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Ask any question and get AI-powered answers with precise
                  source citations from the document.
                </p>
              </div>

              {/* Suggestion chips */}
              <div className="flex flex-wrap gap-2 justify-center pt-2">
                {[
                  "Summarize this document",
                  "What are the key findings?",
                  "List the main topics",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => {
                      setInput(suggestion);
                      textareaRef.current?.focus();
                    }}
                    className="text-xs px-3 py-1.5 rounded-full border bg-background hover:bg-accent hover:border-primary/20 transition-all duration-200 text-muted-foreground hover:text-foreground cursor-pointer"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-4">
            {messages.map((msg, i) => (
              <MessageBubble
                key={msg.id}
                role={msg.role}
                content={msg.content}
                sources={msg.sources}
                isStreaming={
                  isStreaming &&
                  i === messages.length - 1 &&
                  msg.role === "assistant"
                }
                onPageClick={onPageClick}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="border-t bg-background/80 backdrop-blur-sm p-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2 rounded-2xl border bg-background shadow-sm p-1.5 focus-within:ring-2 focus-within:ring-ring/20 focus-within:border-primary/30 transition-all duration-200">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about this document..."
              className="flex-1 resize-none bg-transparent px-3 py-2 text-sm focus:outline-none min-h-[40px] max-h-[150px] placeholder:text-muted-foreground/60"
              rows={1}
              disabled={isStreaming}
            />
            <Button
              size="icon"
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              className="shrink-0 rounded-xl h-9 w-9 transition-all duration-200"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-[11px] text-muted-foreground/50 text-center mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
