"use client";

import { useRef, useState } from "react";
import { FileUp, Loader2 } from "lucide-react";

import { uploadDocument, type DocumentResponse } from "@/lib/api";
import { cn } from "cn";

const ACCEPTED_EXTENSIONS = ".pdf,.docx,.txt";

// Lives in the sidebar rather than above the conversation: it is used when
// loading a document, and the rest of the time it was taking the top third of
// the screen away from the answers.
export function DocumentUpload({
  onIndexed,
}: {
  onIndexed: (filename: string) => void;
}) {
  const [pendingFilename, setPendingFilename] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<DocumentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDraggedOver, setIsDraggedOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Indexing is synchronous on the backend and takes seconds to minutes, so the
  // pending name is the only thing telling the user the app is still working.
  const isUploading = pendingFilename !== null;

  async function upload(file: File) {
    setError(null);
    setLastResult(null);
    setPendingFilename(file.name);
    try {
      const result = await uploadDocument(file);
      setLastResult(result);
      // The name the backend indexed it under, not the one the browser sent:
      // it is normalised there, and the scope has to match the stored rows.
      onIndexed(result.filename);
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Не удалось загрузить файл.",
      );
    } finally {
      setPendingFilename(null);
    }
  }

  return (
    <div className="space-y-2">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setIsDraggedOver(true);
        }}
        onDragLeave={() => setIsDraggedOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setIsDraggedOver(false);
          const file = event.dataTransfer.files?.[0];
          if (file && !isUploading) {
            void upload(file);
          }
        }}
        className={cn(
          "rounded-lg border border-dashed transition-colors",
          isDraggedOver ? "border-ring bg-sidebar-accent" : "border-sidebar-border",
        )}
      >
        <button
          type="button"
          disabled={isUploading}
          onClick={() => inputRef.current?.click()}
          className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-xs disabled:opacity-70"
        >
          {isUploading ? (
            <>
              <Loader2 className="size-4 shrink-0 animate-spin" />
              <span className="min-w-0">
                <span className="block truncate font-medium">{pendingFilename}</span>
                <span className="text-sidebar-foreground/60">
                  Индексация идёт синхронно, до нескольких минут
                </span>
              </span>
            </>
          ) : (
            <>
              <FileUp className="size-4 shrink-0 text-sidebar-foreground/60" />
              <span className="min-w-0">
                <span className="block font-medium">Загрузить документ</span>
                <span className="text-sidebar-foreground/60">
                  PDF, DOCX или TXT — можно перетащить
                </span>
              </span>
            </>
          )}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_EXTENSIONS}
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            // Reset so re-picking the same file fires onChange again.
            event.target.value = "";
            if (file) {
              void upload(file);
            }
          }}
        />
      </div>

      {error && (
        <p className="rounded-lg bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {error}
        </p>
      )}

      {lastResult && (
        <p className="px-1 text-xs text-sidebar-foreground/70">
          <span className="font-medium">{lastResult.filename}</span> —{" "}
          {lastResult.chunks_indexed} {pluralizeChunks(lastResult.chunks_indexed)}.
          {lastResult.warnings.length > 0 &&
            " Цитаты по этому документу будут содержать пункт, но не страницу."}
        </p>
      )}

    </div>
  );
}

function pluralizeChunks(count: number): string {
  return pluralize(count, "чанк", "чанка", "чанков");
}

function pluralize(count: number, one: string, few: string, many: string): string {
  const lastTwo = count % 100;
  const last = count % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return many;
  if (last === 1) return one;
  if (last >= 2 && last <= 4) return few;
  return many;
}
