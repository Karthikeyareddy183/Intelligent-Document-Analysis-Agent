"use client";

import { useRouter } from "next/navigation";
import {
  FileText,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  Image as ImageIcon,
  MessageSquare,
  MoreHorizontal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useDeleteDocument } from "@/lib/hooks/use-documents";
import { formatDate, formatFileSize } from "@/lib/utils";
import { toast } from "sonner";
import { useState } from "react";

interface DocumentCardProps {
  document: {
    document_id: string;
    filename: string;
    file_type?: string;
    file_size?: number;
    status: string;
    pages: number;
    chunks_created: number;
    error?: string;
    created_at?: string;
  };
}

export function DocumentCard({ document: doc }: DocumentCardProps) {
  const router = useRouter();
  const deleteMutation = useDeleteDocument();
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const isProcessing = doc.status === "processing";
  const isFailed = doc.status === "failed";
  const isReady = doc.status === "completed";

  const isImage =
    doc.file_type?.startsWith("image/") ||
    /\.(jpg|jpeg|png|webp)$/i.test(doc.filename);

  function handleClick() {
    if (isReady) {
      router.push(`/chat/${doc.document_id}`);
    }
  }

  async function handleDelete() {
    toast.promise(deleteMutation.mutateAsync(doc.document_id), {
      loading: "Deleting document...",
      success: "Document deleted",
      error: "Failed to delete document",
    });
    setShowDeleteDialog(false);
  }

  return (
    <>
      <Card
        className={`group cursor-pointer overflow-hidden hover:shadow-lg hover:-translate-y-0.5 ${
          isReady ? "hover:border-primary/30" : ""
        }`}
        onClick={handleClick}
      >
        {/* Status bar at top */}
        <div
          className={`h-1 transition-all duration-500 ${
            isProcessing
              ? "bg-amber-400 animate-pulse"
              : isFailed
              ? "bg-destructive"
              : "bg-emerald-500"
          }`}
        />

        <div className="p-4 space-y-3">
          {/* File icon + name */}
          <div className="flex items-start gap-3">
            <div
              className={`flex h-10 w-10 items-center justify-center rounded-xl shrink-0 ${
                isImage
                  ? "bg-purple-100 dark:bg-purple-900/30"
                  : "bg-blue-100 dark:bg-blue-900/30"
              }`}
            >
              {isImage ? (
                <ImageIcon className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              ) : (
                <FileText className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-sm truncate leading-tight">
                {doc.filename}
              </p>
              {doc.file_size && (
                <p className="text-xs text-muted-foreground mt-0.5">
                  {formatFileSize(doc.file_size)}
                </p>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={(e) => {
                e.stopPropagation();
                setShowDeleteDialog(true);
              }}
              disabled={deleteMutation.isPending}
            >
              <Trash2 className="h-3.5 w-3.5 text-muted-foreground hover:text-destructive transition-colors" />
            </Button>
          </div>

          {/* Status + metadata */}
          <div className="flex items-center justify-between">
            <div>
              {isProcessing && (
                <Badge variant="warning" className="gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Processing
                </Badge>
              )}
              {isReady && (
                <Badge variant="success" className="gap-1">
                  <CheckCircle2 className="h-3 w-3" />
                  Ready
                </Badge>
              )}
              {isFailed && (
                <Badge variant="destructive" className="gap-1">
                  <XCircle className="h-3 w-3" />
                  Failed
                </Badge>
              )}
            </div>
            {isReady && (
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span>{doc.pages} pages</span>
                <span>{doc.chunks_created} chunks</span>
              </div>
            )}
          </div>

          {/* Date */}
          {doc.created_at && (
            <p className="text-xs text-muted-foreground pt-1 border-t">
              {formatDate(doc.created_at)}
            </p>
          )}
        </div>
      </Card>

      {/* Delete confirmation dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete document</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete &ldquo;{doc.filename}&rdquo;? This
              will permanently remove the document, all chunks, and
              conversation history.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
