"use client";

import { useState } from "react";
import { ChevronDown, FileText, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface Source {
  document_id: string;
  page: number;
  section: string;
  chunk_text: string;
  relevance_score: number;
}

interface SourceCitationProps {
  sources: Source[];
  onPageClick?: (page: number) => void;
}

export function SourceCitation({ sources, onPageClick }: SourceCitationProps) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="rounded-xl border bg-background/50 overflow-hidden animate-scale-in">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between px-3.5 py-2.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
      >
        <span className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5" />
          <span>
            {sources.length} source{sources.length !== 1 ? "s" : ""} found
          </span>
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 transition-transform duration-200",
            expanded && "rotate-180"
          )}
        />
      </button>

      {expanded && (
        <div className="border-t divide-y">
          {sources.map((src, i) => (
            <div
              key={i}
              className="px-3.5 py-3 space-y-1.5 hover:bg-muted/30 transition-colors"
            >
              <div className="flex items-center justify-between">
                <button
                  onClick={() => onPageClick?.(src.page)}
                  className="flex items-center gap-1.5 text-xs font-semibold text-primary hover:text-primary/80 transition-colors cursor-pointer"
                >
                  <ExternalLink className="h-3 w-3" />
                  Page {src.page}
                  {src.section ? ` - ${src.section}` : ""}
                </button>
                <Badge variant="secondary" className="text-[10px] h-5 px-1.5">
                  {Math.round(src.relevance_score * 100)}% match
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed line-clamp-3">
                {src.chunk_text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
