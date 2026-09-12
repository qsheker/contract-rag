"use client";

import { MessageSquarePlus, Trash2 } from "lucide-react";

import { DocumentUpload } from "@/components/document-upload";
import { ThemeSwitch } from "@/components/theme-switch";
import { Button } from "@/components/ui/button";
import { titleFor, type Chat } from "@/lib/chats";
import { cn } from "cn";

export function ChatSidebar({
  chats,
  activeChatId,
  onSelect,
  onCreate,
  onDelete,
  onIndexed,
}: {
  chats: Chat[];
  activeChatId: string | null;
  onSelect: (chatId: string) => void;
  onCreate: () => void;
  onDelete: (chatId: string) => void;
  onIndexed: (filename: string) => void;
}) {
  return (
    <div className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 px-3 py-3">
        <span className="flex-1 truncate text-sm font-semibold">contract-rag</span>
      </div>

      <div className="px-3 pb-2">
        <Button
          variant="outline"
          className="w-full justify-start bg-transparent"
          onClick={onCreate}
        >
          <MessageSquarePlus className="size-4" />
          Новый чат
        </Button>
      </div>

      {/* The list scrolls on its own; the sections below it stay reachable
          however long the history gets. */}
      <nav className="min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 py-1">
        {chats.length === 0 && (
          <p className="px-2 py-3 text-xs text-sidebar-foreground/60">
            Пока ни одного чата.
          </p>
        )}
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={cn(
              "group flex items-center gap-1 rounded-lg pr-1 transition-colors",
              chat.id === activeChatId
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "hover:bg-sidebar-accent/60",
            )}
          >
            <button
              type="button"
              onClick={() => onSelect(chat.id)}
              className="min-w-0 flex-1 truncate px-2.5 py-2 text-left text-sm"
              aria-current={chat.id === activeChatId}
            >
              {titleFor(chat)}
            </button>
            <button
              type="button"
              onClick={() => onDelete(chat.id)}
              aria-label={`Удалить чат «${titleFor(chat)}»`}
              // Visible on hover, and always once focused by keyboard.
              className="rounded-md p-1.5 text-sidebar-foreground/50 opacity-0 transition hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        ))}
      </nav>

      <div className="space-y-3 border-t border-sidebar-border px-3 py-3">
        <DocumentUpload onIndexed={onIndexed} />
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-sidebar-foreground/60">Оформление</p>
          <ThemeSwitch />
        </div>
      </div>
    </div>
  );
}
