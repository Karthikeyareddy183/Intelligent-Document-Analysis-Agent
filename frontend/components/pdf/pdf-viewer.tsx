"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import "react-pdf/dist/esm/Page/TextLayer.css";
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize2,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { createClient } from "@/lib/supabase/client";

// Set up PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PDFViewerProps {
  url: string;
  targetPage?: number;
}

export function PDFViewer({ url, targetPage }: PDFViewerProps) {
  const [numPages, setNumPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [pdfData, setPdfData] = useState<ArrayBuffer | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch PDF with auth headers
  useEffect(() => {
    if (!url) return;

    async function fetchPdf() {
      try {
        setLoading(true);
        setError(false);
        const supabase = createClient();
        const {
          data: { session },
        } = await supabase.auth.getSession();

        const res = await fetch(url, {
          headers: session?.access_token
            ? { Authorization: `Bearer ${session.access_token}` }
            : {},
        });

        if (!res.ok) throw new Error("Failed to fetch PDF");

        const buffer = await res.arrayBuffer();
        setPdfData(buffer);
      } catch {
        setError(true);
      } finally {
        setLoading(false);
      }
    }

    fetchPdf();
  }, [url]);

  useEffect(() => {
    if (targetPage && targetPage > 0 && targetPage <= numPages) {
      setCurrentPage(targetPage);
    }
  }, [targetPage, numPages]);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
  }

  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= numPages) {
        setCurrentPage(page);
      }
    },
    [numPages]
  );

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center bg-muted/20">
        <div className="flex flex-col items-center gap-3 text-muted-foreground animate-pulse">
          <FileText className="h-8 w-8" />
          <p className="text-sm">Loading PDF...</p>
        </div>
      </div>
    );
  }

  if (error || !pdfData) {
    return (
      <div className="flex h-full items-center justify-center bg-muted/20">
        <div className="flex flex-col items-center gap-3 text-center px-6">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted">
            <FileText className="h-6 w-6 text-muted-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium">Unable to load preview</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              The PDF viewer requires the file to be accessible
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-muted/10">
      {/* Controls */}
      <div className="flex items-center justify-between border-b bg-background/80 backdrop-blur-sm px-3 py-2">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage <= 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-xs tabular-nums px-2 font-medium text-muted-foreground">
            {currentPage}
            <span className="mx-1 text-muted-foreground/40">/</span>
            {numPages}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage >= numPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale((s) => Math.max(0.5, s - 0.25))}
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs tabular-nums w-12 text-center font-medium text-muted-foreground">
            {Math.round(scale * 100)}%
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale((s) => Math.min(3, s + 0.25))}
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => setScale(1.0)}
            aria-label="Reset zoom"
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* PDF Content */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto flex justify-center p-4"
      >
        <Document
          file={{ data: pdfData }}
          onLoadSuccess={onDocumentLoadSuccess}
          loading={
            <div className="flex items-center justify-center h-64">
              <div className="text-sm text-muted-foreground animate-pulse">
                Rendering page...
              </div>
            </div>
          }
          error={
            <div className="flex items-center justify-center h-64">
              <div className="text-sm text-muted-foreground">
                Unable to render PDF
              </div>
            </div>
          }
        >
          <Page
            pageNumber={currentPage}
            scale={scale}
            renderAnnotationLayer={true}
            renderTextLayer={true}
            className="shadow-lg rounded-lg overflow-hidden"
          />
        </Document>
      </div>
    </div>
  );
}
