"use client";

import { useState, useCallback } from "react";
import { useParams } from "next/navigation";
import {
  PanelResizeHandle as ResizableHandle,
  Panel as ResizablePanel,
  PanelGroup as ResizablePanelGroup,
} from "react-resizable-panels";
import {
  ArrowLeft,
  PanelLeftClose,
  PanelRightClose,
  FileText,
  Loader2,
  AlertCircle,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ChatPanel } from "@/components/chat/chat-panel";
import { ConversationSidebar } from "@/components/chat/conversation-sidebar";
import { PDFViewer } from "@/components/pdf/pdf-viewer";
import { useDocument } from "@/lib/hooks/use-documents";
import { useConversation } from "@/lib/hooks/use-conversations";
import { Skeleton } from "@/components/ui/skeleton";
import Link from "next/link";

export default function ChatPage() {
  const params = useParams();
  const documentId = params.documentId as string;

  const { data: doc, isLoading: docLoading } = useDocument(documentId);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const [targetPage, setTargetPage] = useState<number | undefined>();
  const [showSidebar, setShowSidebar] = useState(true);
  const [showPdf, setShowPdf] = useState(true);

  const { data: conversation } = useConversation(activeConversationId);

  const isProcessing = doc?.status === "processing";
  const isFailed = doc?.status === "failed";

  const handlePageClick = useCallback((page: number) => {
    setTargetPage(page);
    setShowPdf(true);
  }, []);

  const pdfUrl = doc?.storage_path
    ? `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/documents/${documentId}/file`
    : "";

  // Loading state
  if (docLoading) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <div className="flex flex-col items-center gap-3 animate-pulse">
          <Skeleton className="h-12 w-12 rounded-xl" />
          <Skeleton className="h-4 w-48" />
        </div>
      </div>
    );
  }

  // Not found
  if (!doc) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <div className="text-center space-y-3 animate-fade-in">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted mx-auto">
            <FileText className="h-7 w-7 text-muted-foreground" />
          </div>
          <h2 className="text-xl font-heading font-semibold">
            Document not found
          </h2>
          <Link
            href="/dashboard"
            className="text-sm text-primary hover:text-primary/80 transition-colors"
          >
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  // Processing
  if (isProcessing) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <div className="text-center space-y-4 animate-fade-in">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-100 dark:bg-amber-900/30 mx-auto">
            <Loader2 className="h-8 w-8 text-amber-600 dark:text-amber-400 animate-spin" />
          </div>
          <div>
            <h2 className="text-xl font-heading font-semibold">
              Processing document
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {doc.filename}
            </p>
            <p className="text-xs text-muted-foreground mt-2">
              This may take a moment depending on the file size
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Failed
  if (isFailed) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <div className="text-center space-y-4 animate-fade-in">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-destructive/10 mx-auto">
            <AlertCircle className="h-8 w-8 text-destructive" />
          </div>
          <div>
            <h2 className="text-xl font-heading font-semibold text-destructive">
              Processing failed
            </h2>
            <p className="text-sm text-muted-foreground mt-1 max-w-sm">
              {doc.error || "An error occurred while processing the document"}
            </p>
          </div>
          <Link href="/dashboard">
            <Button variant="outline" size="sm">
              Back to dashboard
            </Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-3.5rem)] flex flex-col">
      {/* Top bar */}
      <div className="flex items-center gap-3 border-b px-4 py-2 bg-background/80 backdrop-blur-sm shrink-0">
        <Link href="/dashboard">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>

        <div className="flex items-center gap-2 min-w-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10 shrink-0">
            <FileText className="h-3.5 w-3.5 text-primary" />
          </div>
          <span className="font-medium text-sm truncate">{doc.filename}</span>
        </div>

        <div className="hidden sm:flex items-center gap-2">
          <Badge variant="secondary" className="text-[10px] h-5">
            {doc.pages} pages
          </Badge>
          <Badge variant="secondary" className="text-[10px] h-5">
            {doc.chunks_created} chunks
          </Badge>
        </div>

        <div className="ml-auto flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 hidden md:flex"
            onClick={() => setShowSidebar(!showSidebar)}
            aria-label="Toggle sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 hidden md:flex"
            onClick={() => setShowPdf(!showPdf)}
            aria-label="Toggle PDF viewer"
          >
            <PanelRightClose className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup direction="horizontal">
          {/* Conversation sidebar */}
          {showSidebar && (
            <>
              <ResizablePanel
                defaultSize={20}
                minSize={15}
                maxSize={30}
                className="hidden md:block"
              >
                <ConversationSidebar
                  documentId={documentId}
                  activeConversationId={activeConversationId}
                  onSelect={setActiveConversationId}
                  onNew={setActiveConversationId}
                />
              </ResizablePanel>
              <ResizableHandle />
            </>
          )}

          {/* Chat panel */}
          <ResizablePanel defaultSize={showPdf ? 45 : 80} minSize={30}>
            <ChatPanel
              documentId={documentId}
              conversationId={activeConversationId}
              onPageClick={handlePageClick}
              initialMessages={conversation?.messages}
            />
          </ResizablePanel>

          {/* PDF viewer */}
          {showPdf && (
            <>
              <ResizableHandle />
              <ResizablePanel
                defaultSize={35}
                minSize={20}
                className="hidden md:block"
              >
                <PDFViewer url={pdfUrl} targetPage={targetPage} />
              </ResizablePanel>
            </>
          )}
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
