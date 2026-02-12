"use client";

import { useCallback, useState } from "react";
import { Upload, FileText, Loader2, CheckCircle } from "lucide-react";
import { useUploadDocument } from "@/lib/hooks/use-documents";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";

const ACCEPTED_TYPES = [
  "application/pdf",
  "image/jpeg",
  "image/png",
  "image/webp",
];

export function UploadZone() {
  const [dragActive, setDragActive] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const upload = useUploadDocument();

  const handleFile = useCallback(
    (file: File) => {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        toast.error("Unsupported file type. Upload PDF, JPG, PNG, or WEBP.");
        return;
      }
      if (file.size > 250 * 1024 * 1024) {
        toast.error("File too large. Maximum size is 250MB.");
        return;
      }

      setUploadFile(file);
      setUploadProgress(0);

      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + Math.random() * 15;
        });
      }, 200);

      upload.mutateAsync(file).then(
        () => {
          clearInterval(interval);
          setUploadProgress(100);
          toast.success(`${file.name} uploaded! Processing started.`);
          setTimeout(() => {
            setUploadFile(null);
            setUploadProgress(0);
          }, 2000);
        },
        (err) => {
          clearInterval(interval);
          setUploadFile(null);
          setUploadProgress(0);
          toast.error(`Upload failed: ${err.message}`);
        }
      );
    },
    [upload]
  );

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(true);
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  const isUploading = upload.isPending;
  const isComplete = uploadProgress === 100;

  return (
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={() => setDragActive(false)}
      className={cn(
        "relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer group",
        dragActive
          ? "border-primary bg-primary/5 scale-[1.01]"
          : "border-muted-foreground/20 hover:border-primary/40 hover:bg-muted/50",
        isUploading && "pointer-events-none"
      )}
    >
      <input
        type="file"
        accept=".pdf,.jpg,.jpeg,.png,.webp"
        onChange={handleInputChange}
        className="absolute inset-0 cursor-pointer opacity-0 z-10"
        disabled={isUploading}
      />

      {uploadFile ? (
        /* Upload progress view */
        <div className="px-8 py-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
              {isComplete ? (
                <CheckCircle className="h-5 w-5 text-emerald-500" />
              ) : (
                <Loader2 className="h-5 w-5 text-primary animate-spin" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-sm truncate">{uploadFile.name}</p>
              <p className="text-xs text-muted-foreground">
                {isComplete
                  ? "Upload complete! Processing document..."
                  : `Uploading... ${Math.round(uploadProgress)}%`}
              </p>
            </div>
          </div>
          <Progress value={uploadProgress} className="h-1.5" />
        </div>
      ) : (
        /* Default drag-and-drop view */
        <div className="flex flex-col items-center gap-4 px-8 py-10">
          <div
            className={cn(
              "rounded-2xl p-4 transition-all duration-300",
              dragActive
                ? "bg-primary/10 scale-110"
                : "bg-muted group-hover:bg-primary/5"
            )}
          >
            <Upload
              className={cn(
                "h-8 w-8 transition-colors duration-300",
                dragActive
                  ? "text-primary"
                  : "text-muted-foreground group-hover:text-primary/70"
              )}
            />
          </div>
          <div className="text-center space-y-1.5">
            <p className="font-heading font-semibold text-base">
              Drop your document here
            </p>
            <p className="text-sm text-muted-foreground">
              or click to browse. Supports{" "}
              <span className="font-medium text-foreground/70">
                PDF, JPG, PNG, WEBP
              </span>{" "}
              up to 250MB
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
