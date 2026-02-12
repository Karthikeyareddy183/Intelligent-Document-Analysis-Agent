"use client";

import { useDocuments } from "@/lib/hooks/use-documents";
import { UploadZone } from "@/components/dashboard/upload-zone";
import { DocumentCard } from "@/components/dashboard/document-card";
import { Skeleton } from "@/components/ui/skeleton";
import { FileText, Sparkles } from "lucide-react";

export default function DashboardPage() {
  const { data, isLoading } = useDocuments();
  const documents = data?.documents || [];

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-8 space-y-8 animate-fade-in">
      {/* Page header */}
      <div className="space-y-1">
        <h1 className="text-3xl font-heading font-bold tracking-tight">
          Documents
        </h1>
        <p className="text-muted-foreground">
          Upload and analyze your documents with AI-powered insights
        </p>
      </div>

      {/* Upload zone */}
      <UploadZone />

      {/* Document grid */}
      {isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-40 rounded-xl" />
          ))}
        </div>
      ) : documents.length === 0 ? (
        /* Empty state */
        <div className="flex flex-col items-center justify-center py-20 text-center animate-slide-up">
          <div className="relative mb-6">
            <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-gradient-to-br from-primary/10 to-blue-500/10 border border-primary/10">
              <FileText className="h-10 w-10 text-primary/60" />
            </div>
            <div className="absolute -top-1 -right-1 flex h-7 w-7 items-center justify-center rounded-full bg-primary shadow-lg">
              <Sparkles className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
          </div>
          <h3 className="font-heading font-semibold text-xl mb-2">
            No documents yet
          </h3>
          <p className="text-muted-foreground max-w-md leading-relaxed">
            Upload your first document to get started. Drop a PDF or image above
            and our AI will analyze it, extract content, and let you ask questions
            with precise source citations.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {documents.map((doc: any, i: number) => (
            <div key={doc.document_id} className="animate-slide-up" style={{ animationDelay: `${i * 50}ms` }}>
              <DocumentCard document={doc} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
