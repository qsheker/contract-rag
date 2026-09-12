"use client";

import { useEffect, useRef, useState } from "react";
import { FileText, PanelLeft, X } from "lucide-react";

import { Chat } from "@/components/chat";
import { ChatSidebar } from "@/components/chat-sidebar";
import { Button } from "@/components/ui/button";
import { sendChatMessage, type ChatTurn } from "@/lib/api";
import {
  createChat,
  loadChats,
  saveChats,
  titleFor,
  type Chat as Conversation,
  type ChatEntry,
} from "@/lib/chats";
import { cn } from "cn";

export function AppShell() {
  const [chats, setChats] = useState<Conversation[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  // Held here rather than inside the transcript so switching chats mid-answer
  // keeps both the spinner and the cancel button attached to the right one.
  const [answeringChatId, setAnsweringChatId] = useState<string | null>(null);
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const stored = loadChats();
    const restored = stored.length > 0 ? stored : [createChat()];
    setChats(restored);
    setActiveChatId(restored[0].id);
  }, []);

  useEffect(() => {
    saveChats(chats);
  }, [chats]);

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? null;

  function updateEntries(
    chatId: string,
    update: (entries: ChatEntry[]) => ChatEntry[],
  ) {
    setChats((current) =>
      current.map((chat) =>
        chat.id === chatId ? { ...chat, entries: update(chat.entries) } : chat,
      ),
    );
  }

  async function ask(
    chatId: string,
    message: string,
    previous: ChatEntry[],
    sourceFiles: string[],
  ) {
    // The history sent to the backend carries the user's literal words -
    // condensation is the backend's job and must see what was actually typed.
    // Failures are left out: they are not turns of the conversation.
    const history: ChatTurn[] = previous
      .filter((entry) => entry.role !== "error")
      .map((entry) => ({ role: entry.role, content: entry.content }));

    const controller = new AbortController();
    abortRef.current = controller;
    updateEntries(chatId, () => [...previous, { role: "user", content: message }]);
    setAnsweringChatId(chatId);

    try {
      const response = await sendChatMessage(
        message,
        history,
        sourceFiles,
        controller.signal,
      );
      updateEntries(chatId, (current) => [
        ...current,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations,
          excerpts: response.excerpts,
          standaloneQuery: response.standalone_query,
        },
      ]);
    } catch (chatError) {
      updateEntries(chatId, (current) => [
        ...current,
        { role: "error", content: describeFailure(chatError) },
      ]);
    } finally {
      abortRef.current = null;
      setAnsweringChatId((current) => (current === chatId ? null : current));
    }
  }

  function createConversation() {
    const chat = createChat();
    setChats((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    setSidebarOpen(false);
  }

  function deleteConversation(chatId: string) {
    setChats((current) => {
      const remaining = current.filter((chat) => chat.id !== chatId);
      // The list is never empty: an empty sidebar with no way back to a chat
      // is a dead end, so deleting the last one leaves a fresh one.
      const next = remaining.length > 0 ? remaining : [createChat()];
      if (chatId === activeChatId) {
        setActiveChatId(next[0].id);
      }
      return next;
    });
  }

  function retry(chat: Conversation, errorIndex: number) {
    // The question sits immediately before its failure; replaying it drops both
    // and asks again from the state the conversation had at the time.
    const question = chat.entries[errorIndex - 1];
    if (!question || question.role !== "user") {
      return;
    }
    void ask(
      chat.id,
      question.content,
      chat.entries.slice(0, errorIndex - 1),
      chat.sourceFiles,
    );
  }

  // An upload belongs to the chat it was made from: that is what makes this
  // conversation about that document rather than about the whole index.
  function scopeToDocument(filename: string) {
    if (!activeChatId) {
      return;
    }
    setChats((current) =>
      current.map((chat) =>
        chat.id === activeChatId && !chat.sourceFiles.includes(filename)
          ? { ...chat, sourceFiles: [...chat.sourceFiles, filename] }
          : chat,
      ),
    );
  }

  function clearScope(chatId: string) {
    setChats((current) =>
      current.map((chat) =>
        chat.id === chatId ? { ...chat, sourceFiles: [] } : chat,
      ),
    );
  }

  return (
    <div className="flex h-full">
      <aside
        className={cn(
          "w-72 shrink-0 border-r border-sidebar-border",
          // Off-canvas below the tablet breakpoint: at that width the sidebar
          // would leave the answer a column too narrow to read.
          "max-lg:fixed max-lg:inset-y-0 max-lg:left-0 max-lg:z-40 max-lg:transition-transform",
          isSidebarOpen ? "max-lg:translate-x-0" : "max-lg:-translate-x-full",
        )}
      >
        <ChatSidebar
          chats={chats}
          activeChatId={activeChatId}
          onSelect={(chatId) => {
            setActiveChatId(chatId);
            setSidebarOpen(false);
          }}
          onCreate={createConversation}
          onDelete={deleteConversation}
          onIndexed={scopeToDocument}
        />
      </aside>

      {isSidebarOpen && (
        <button
          type="button"
          aria-label="Закрыть меню"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-30 bg-black/40 lg:hidden"
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
          <Button
            size="icon"
            variant="ghost"
            className="lg:hidden"
            aria-label={isSidebarOpen ? "Закрыть меню" : "Открыть меню"}
            onClick={() => setSidebarOpen((open) => !open)}
          >
            {isSidebarOpen ? <X className="size-4" /> : <PanelLeft className="size-4" />}
          </Button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-medium">
              {activeChat ? titleFor(activeChat) : "contract-rag"}
            </h1>
            <p className="truncate text-xs text-muted-foreground">
              Ответы с обязательной ссылкой на пункт договора
            </p>
          </div>
          {activeChat && activeChat.sourceFiles.length > 0 && (
            <button
              type="button"
              onClick={() => clearScope(activeChat.id)}
              title="Искать по всем документам"
              className="flex max-w-[45%] shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <FileText className="size-3.5 shrink-0" />
              <span className="truncate font-mono">
                {activeChat.sourceFiles.length === 1
                  ? activeChat.sourceFiles[0]
                  : `${activeChat.sourceFiles.length} документа`}
              </span>
              <X className="size-3 shrink-0" />
            </button>
          )}
        </header>

        {activeChat && (
          <Chat
            entries={activeChat.entries}
            isAnswering={answeringChatId === activeChat.id}
            onAsk={(message) =>
              void ask(activeChat.id, message, activeChat.entries, activeChat.sourceFiles)
            }
            onRetry={(errorIndex) => retry(activeChat, errorIndex)}
            onCancel={() => abortRef.current?.abort()}
          />
        )}
      </main>
    </div>
  );
}

function describeFailure(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") {
    return "Ожидание отменено. Ответ можно запросить снова.";
  }
  return error instanceof Error ? error.message : "Не удалось получить ответ.";
}
