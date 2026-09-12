"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, RotateCcw, Send, X } from "lucide-react";

import { CitationBadges } from "@/components/citation-badges";
import { MessageText } from "@/components/message-text";
import { RetrievalDetails } from "@/components/retrieval-details";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ChatEntry } from "@/lib/chats";
import { cn } from "cn";

export function Chat({
  entries,
  isAnswering,
  onAsk,
  onRetry,
  onCancel,
}: {
  entries: ChatEntry[];
  isAnswering: boolean;
  onAsk: (message: string) => void;
  onRetry: (errorIndex: number) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, isAnswering]);

  function send() {
    const message = draft.trim();
    if (!message || isAnswering) {
      return;
    }
    setDraft("");
    onAsk(message);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        {/* One column, capped: a line of legal prose running the full width of
            a desktop window is the thing that made this unreadable. */}
        <div className="mx-auto w-full max-w-3xl px-4 py-8 sm:px-6">
          {entries.length === 0 && !isAnswering ? (
            <EmptyState />
          ) : (
            <div className="space-y-8">
              {entries.map((entry, index) => (
                <Message
                  key={index}
                  entry={entry}
                  question={index > 0 ? entries[index - 1] : undefined}
                  onRetry={() => onRetry(index)}
                  canRetry={!isAnswering}
                />
              ))}

              {isAnswering && (
                <div className="flex items-center gap-3 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" />
                  Ищем в договорах и собираем ответ…
                  <Button size="sm" variant="ghost" onClick={onCancel}>
                    <X className="size-3.5" />
                    Отменить
                  </Button>
                </div>
              )}
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="border-t bg-background/80 backdrop-blur">
        <div className="mx-auto w-full max-w-3xl px-4 py-4 sm:px-6">
          <div className="flex items-end gap-2 rounded-2xl border bg-card p-2 shadow-sm focus-within:ring-1 focus-within:ring-ring">
            <Textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  send();
                }
              }}
              placeholder="Вопрос по договору…"
              rows={1}
              disabled={isAnswering}
              className="max-h-40 min-h-9 resize-none border-0 bg-transparent px-2 py-1.5 shadow-none focus-visible:ring-0"
            />
            <Button
              size="icon"
              onClick={send}
              disabled={isAnswering || draft.trim().length === 0}
              aria-label="Отправить"
            >
              <Send className="size-4" />
            </Button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Enter — отправить, Shift+Enter — новая строка. Ответ строится только по
            загруженным документам.
          </p>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="py-16 text-center">
      <h2 className="text-lg font-semibold">Спросите что-нибудь по договору</h2>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
        Например: «Какой срок оплаты?» — а затем уточните: «а если просрочить?».
        Каждый ответ подкреплён пунктом договора, который можно раскрыть и
        прочитать целиком.
      </p>
    </div>
  );
}

function Message({
  entry,
  question,
  onRetry,
  canRetry,
}: {
  entry: ChatEntry;
  question: ChatEntry | undefined;
  onRetry: () => void;
  canRetry: boolean;
}) {
  if (entry.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground">
          <p className="whitespace-pre-wrap">{entry.content}</p>
        </div>
      </div>
    );
  }

  if (entry.role === "error") {
    return (
      <div className="space-y-3 rounded-xl border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm">
        <p className="whitespace-pre-wrap">{entry.content}</p>
        {question?.role === "user" && (
          <Button size="sm" variant="outline" onClick={onRetry} disabled={!canRetry}>
            <RotateCcw className="size-3.5" />
            Повторить
          </Button>
        )}
      </div>
    );
  }

  // Shown only when condensation actually rewrote the question: with an empty
  // history the backend returns the message unchanged, and repeating it back
  // word for word trains the eye to skip the line that matters.
  const wasCondensed =
    question?.role === "user" && entry.standaloneQuery !== question.content;

  return (
    // No bubble around the answer: it is the long text on the page, and a tinted
    // box around eight clauses is what made the old screen look like a dump.
    <div className="space-y-4 text-[15px] leading-7">
      <MessageText text={entry.content} />

      {wasCondensed && (
        <p className="text-xs text-muted-foreground">
          Поиск шёл по запросу: «{entry.standaloneQuery}»
        </p>
      )}

      <div className={cn("space-y-3 border-t pt-3", entry.citations.length === 0 && "pt-2")}>
        <CitationBadges citations={entry.citations} excerpts={entry.excerpts} />
        <RetrievalDetails excerpts={entry.excerpts} />
      </div>
    </div>
  );
}
