import { createClient } from "@/lib/supabase/client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getAuthHeaders(): Promise<Record<string, string>> {
  const supabase = createClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  if (!session?.access_token) {
    throw new Error("Not authenticated");
  }

  return {
    Authorization: `Bearer ${session.access_token}`,
  };
}

async function apiFetch(path: string, options: RequestInit = {}) {
  const authHeaders = await getAuthHeaders();
  const res = await fetch(`${API_URL}/api/v1${path}`, {
    ...options,
    headers: {
      ...authHeaders,
      ...options.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || body.error || `API error ${res.status}`);
  }

  return res.json();
}

// ─── Documents ────────────────────────────

export async function uploadDocument(file: File) {
  const authHeaders = await getAuthHeaders();
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/api/v1/upload`, {
    method: "POST",
    headers: authHeaders,
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Upload failed: ${res.status}`);
  }

  return res.json();
}

export async function listDocuments() {
  return apiFetch("/documents");
}

export async function getDocument(documentId: string) {
  return apiFetch(`/documents/${documentId}`);
}

export async function deleteDocument(documentId: string) {
  return apiFetch(`/documents/${documentId}`, { method: "DELETE" });
}

// ─── Conversations ────────────────────────

export async function listConversations(documentId?: string) {
  const params = documentId ? `?document_id=${documentId}` : "";
  return apiFetch(`/conversations${params}`);
}

export async function getConversation(conversationId: string) {
  return apiFetch(`/conversations/${conversationId}`);
}

export async function createConversation(
  documentId: string,
  title?: string
) {
  return apiFetch("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId, title }),
  });
}

export async function deleteConversation(conversationId: string) {
  return apiFetch(`/conversations/${conversationId}`, { method: "DELETE" });
}

// ─── Streaming Query ──────────────────────

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onSources: (sources: any[]) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

export async function streamQuerySSE(
  query: string,
  documentId: string,
  chatHistory: { role: string; content: string }[],
  conversationId: string | null,
  callbacks: StreamCallbacks
): Promise<void> {
  const authHeaders = await getAuthHeaders();

  const res = await fetch(`${API_URL}/api/v1/query/stream`, {
    method: "POST",
    headers: {
      ...authHeaders,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query,
      document_id: documentId,
      top_k: 5,
      chat_history: chatHistory,
      conversation_id: conversationId,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    callbacks.onError(body.detail || `Query failed: ${res.status}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "token";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete SSE messages (double newline delimited)
    while (buffer.includes("\n\n")) {
      const idx = buffer.indexOf("\n\n");
      const message = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      let eventType = "token";
      let data = "";

      for (const line of message.split("\n")) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          data = line.slice(5);
          // Don't trim leading space from data — it may be intentional
          if (data.startsWith(" ")) data = data.slice(1);
        }
      }

      switch (eventType) {
        case "token":
          callbacks.onToken(data);
          break;
        case "sources":
          try {
            callbacks.onSources(JSON.parse(data));
          } catch {
            callbacks.onSources([]);
          }
          break;
        case "done":
          callbacks.onDone();
          return;
        case "error":
          callbacks.onError(data);
          return;
      }
    }
  }

  // If we exit the loop without a done event
  callbacks.onDone();
}
