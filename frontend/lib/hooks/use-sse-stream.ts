"use client";

import { useState, useCallback, useRef } from "react";
import { streamQuerySSE, type StreamCallbacks } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: any[];
}

interface UseSSEStreamReturn {
  messages: Message[];
  isStreaming: boolean;
  sendMessage: (query: string) => Promise<void>;
  setMessages: (messages: Message[]) => void;
}

export function useSSEStream(
  documentId: string,
  conversationId: string | null
): UseSSEStreamReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const streamingContent = useRef("");

  const sendMessage = useCallback(
    async (query: string) => {
      if (isStreaming) return;

      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: query,
      };

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: "",
        sources: [],
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);
      streamingContent.current = "";

      // Build chat history from existing messages (exclude the new ones)
      const chatHistory = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const callbacks: StreamCallbacks = {
        onToken: (token: string) => {
          streamingContent.current += token;
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              updated[updated.length - 1] = {
                ...last,
                content: streamingContent.current,
              };
            }
            return updated;
          });
        },
        onSources: (sources: any[]) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              updated[updated.length - 1] = { ...last, sources };
            }
            return updated;
          });
        },
        onDone: () => {
          setIsStreaming(false);
        },
        onError: (error: string) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              updated[updated.length - 1] = {
                ...last,
                content:
                  streamingContent.current || `Error: ${error}`,
              };
            }
            return updated;
          });
          setIsStreaming(false);
        },
      };

      try {
        await streamQuerySSE(
          query,
          documentId,
          chatHistory,
          conversationId,
          callbacks
        );
      } catch (err: any) {
        callbacks.onError(err.message || "Stream failed");
      }
    },
    [documentId, conversationId, messages, isStreaming]
  );

  return { messages, isStreaming, sendMessage, setMessages };
}
