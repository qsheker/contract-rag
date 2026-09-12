// Several conversations, kept in the browser. The ticket keeps a database out
// of scope; what a chat list needs is that a conversation outlive the tab it
// was started in, and localStorage gives exactly that and nothing more.

import type { Citation, Excerpt } from "@/lib/api";

export type ChatEntry =
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string;
      citations: Citation[];
      excerpts: Excerpt[];
      // Kept per answer so a follow-up can show what retrieval actually
      // searched for; it differs from the user's words exactly when
      // condensation did work.
      standaloneQuery: string;
    }
  // A failure belongs in the transcript rather than in a banner underneath it:
  // a banner is cleared by the next question, and the history is then left with
  // an unanswered bubble and no trace of why.
  | { role: "error"; content: string };

export type Chat = {
  id: string;
  title: string;
  createdAt: number;
  entries: ChatEntry[];
  // Uploading a document into a chat makes the chat about that document:
  // search stays inside it, so an answer here cannot be built out of a
  // contract the reader never opened in this conversation.
  sourceFiles: string[];
};

const STORAGE_KEY = "contract-rag.chats";
// The single conversation this replaced, read once so a session in progress is
// not thrown away by the upgrade.
const LEGACY_STORAGE_KEY = "contract-rag.chat";
const UNTITLED = "Новый чат";
const TITLE_MAX_CHARS = 48;

export function createChat(): Chat {
  return {
    id: crypto.randomUUID(),
    title: UNTITLED,
    createdAt: Date.now(),
    entries: [],
    sourceFiles: [],
  };
}

// The first question is the title, the way a chat list everywhere else names a
// conversation: it is what the reader is scanning for.
export function titleFor(chat: Chat): string {
  const firstQuestion = chat.entries.find((entry) => entry.role === "user");
  if (!firstQuestion) {
    return UNTITLED;
  }
  const question = firstQuestion.content.trim().replace(/\s+/g, " ");
  return question.length > TITLE_MAX_CHARS
    ? `${question.slice(0, TITLE_MAX_CHARS).trimEnd()}…`
    : question;
}

export function loadChats(): Chat[] {
  const stored = read(STORAGE_KEY);
  if (stored) {
    return stored;
  }
  const legacy = read(LEGACY_STORAGE_KEY);
  if (legacy) {
    // The old key held a bare entry array rather than a list of chats.
    return legacy;
  }
  return [];
}

export function saveChats(chats: Chat[]): void {
  // An empty list is never written back: the first render happens before the
  // load below has applied, and in development React mounts twice, so saving
  // unconditionally would wipe the stored chats and then restore the emptiness.
  if (chats.length === 0) {
    return;
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch {
    // A full or blocked storage must not take the conversation down with it.
  }
}

function isChat(value: unknown): value is Chat {
  const candidate = value as Partial<Chat> | null;
  return (
    typeof candidate?.id === "string" &&
    Array.isArray(candidate.entries) &&
    typeof candidate.createdAt === "number"
  );
}

function read(key: string): Chat[] | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) {
      return null;
    }
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) {
      return null;
    }
    // The legacy key stored entries directly; wrap them into one chat.
    if (key === LEGACY_STORAGE_KEY) {
      const chat = { ...createChat(), entries: parsed as ChatEntry[] };
      return [{ ...chat, title: titleFor(chat) }];
    }
    // Storage outlives the code that wrote it, so its shape is checked rather
    // than asserted: one chat saved by an older build without `entries` used to
    // take the whole page down on render.
    // Chats saved before scoping existed have no sourceFiles; they searched
    // everything, and that is what an absent scope still means.
    const chats = parsed.filter(isChat).map((chat) => ({
      ...chat,
      sourceFiles: chat.sourceFiles ?? [],
    }));
    return chats.length > 0 ? chats : null;
  } catch {
    return null;
  }
}
