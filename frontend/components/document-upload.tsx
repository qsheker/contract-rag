"use client";

import { useRef, useState } from "react";
import { FileUp, Loader2 } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { uploadDocument, type DocumentResponse } from "@/lib/api";
import { cn } from "cn";

const ACCEPTED_EXTENSIONS = ".pdf,.docx,.txt";

export function DocumentUpload() {
  const [pendingFilename, setPendingFilename] = useState<string | null>(null);
  const [indexed, setIndexed] = useState<DocumentResponse[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDraggedOver, setIsDraggedOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Indexing is synchronous on the backend and takes seconds to minutes, so the
  // pending name is the only thing telling the user the app is still working.
  const isUploading = pendingFilename !== null;

  async function upload(file: File) {
    setError(null);
    setPendingFilename(file.name);
    try {
      const result = await uploadDocument(file);
      setIndexed((previous) => [
        result,
        ...previous.filter((entry) => entry.filename !== result.filename),
      ]);
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
    <Card className="gap-3">
      <CardContent className="space-y-3">
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
            "flex flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-6 text-center transition-colors",
            isDraggedOver ? "border-ring bg-muted" : "border-border",
            isUploading && "opacity-70",
          )}
        >
          {isUploading ? (
            <>
              <Loader2 className="size-5 animate-spin text-muted-foreground" />
              <p className="text-sm font-medium">Обрабатываем {pendingFilename}</p>
              <p className="text-xs text-muted-foreground">
                Загрузка, разбиение на пункты и индексация идут синхронно — это
                может занять до нескольких минут.
              </p>
            </>
          ) : (
            <>
              <FileUp className="size-5 text-muted-foreground" />
              <p className="text-sm font-medium">
                Перетащите документ или выберите файл
              </p>
              <p className="text-xs text-muted-foreground">
                PDF, DOCX или TXT. После индексации документ сразу доступен для
                вопросов.
              </p>
              <Button
                size="sm"
                variant="outline"
                onClick={() => inputRef.current?.click()}
              >
                Выбрать файл
              </Button>
            </>
          )}
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
          <Alert variant="destructive">
            <AlertTitle>Файл не проиндексирован</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {indexed.map((document) => (
          <Alert key={document.filename}>
            <AlertTitle>
              {document.filename} — {document.chunks_indexed}{" "}
              {pluralizeChunks(document.chunks_indexed)}
            </AlertTitle>
            {document.warnings.length > 0 && (
              <AlertDescription>
                {document.warnings.join(" ")} Цитаты по этому документу будут
                содержать пункт, но не страницу.
              </AlertDescription>
            )}
          </Alert>
        ))}
      </CardContent>
    </Card>
  );
}

function pluralizeChunks(count: number): string {
  const lastTwo = count % 100;
  const last = count % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return "чанков";
  if (last === 1) return "чанк";
  if (last >= 2 && last <= 4) return "чанка";
  return "чанков";
}
