"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Send } from "lucide-react";

import { CitationBadges } from "@/components/citation-badges";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { sendChatMessage, type ChatTurn, type Citation } from "@/lib/api";
import { cn } from "cn";

type ChatEntry = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  // Kept per answer so a follow-up can show what retrieval actually searched
  // for; it differs from the user's words exactly when condensation did work.
  standaloneQuery?: string;
};

export function Chat() {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [isAnswering, setIsAnswering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, isAnswering]);

  async function send() {
    const message = draft.trim();
    if (!message || isAnswering) {
      return;
    }

    // The history sent to the backend is the conversation as it stood before
    // this message, and it carries the user's literal words - condensation is
    // the backend's job and must see what was actually typed.
    const history: ChatTurn[] = entries.map((entry) => ({
      role: entry.role,
      content: entry.content,
    }));

    setEntries((previous) => [...previous, { role: "user", content: message }]);
    setDraft("");
    setError(null);
    setIsAnswering(true);

    try {
      const response = await sendChatMessage(message, history);
      setEntries((previous) => [
        ...previous,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations,
          standaloneQuery: response.standalone_query,
        },
      ]);
    } catch (chatError) {
      setError(
        chatError instanceof Error
          ? chatError.message
          : "Не удалось получить ответ.",
      );
    } finally {
      setIsAnswering(false);
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
        {entries.length === 0 && !isAnswering && (
          <p className="py-8 text-center text-sm text-muted-foreground">
            Спросите что-нибудь по договору — например, «Какой срок оплаты?», а
            затем уточните: «а если просрочить?».
          </p>
        )}

        {entries.map((entry, index) => (
          <ChatBubble key={index} entry={entry} />
        ))}

        {isAnswering && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Ищем в договорах и собираем ответ…
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Ответ не получен</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="flex items-end gap-2">
        <Textarea
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void send();
            }
          }}
          placeholder="Вопрос по договору. Enter — отправить, Shift+Enter — новая строка."
          rows={2}
          disabled={isAnswering}
          className="min-h-0 resize-none"
        />
        <Button
          size="icon-lg"
          onClick={() => void send()}
          disabled={isAnswering || draft.trim().length === 0}
          aria-label="Отправить"
        >
          <Send />
        </Button>
      </div>
    </div>
  );
}

function ChatBubble({ entry }: { entry: ChatEntry }) {
  const isUser = entry.role === "user";
  const wasCondensed =
    entry.standaloneQuery !== undefined && entry.standaloneQuery.length > 0;

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] space-y-2 rounded-lg px-3 py-2 text-sm",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted",
        )}
      >
        <p className="whitespace-pre-wrap">{entry.content}</p>

        {!isUser && wasCondensed && (
          <p className="text-xs text-muted-foreground">
            Поиск шёл по запросу: «{entry.standaloneQuery}»
          </p>
        )}

        {!isUser && entry.citations && (
          <CitationBadges citations={entry.citations} />
        )}
      </div>
    </div>
  );
}
