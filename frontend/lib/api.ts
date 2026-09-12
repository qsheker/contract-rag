// The typed edge of the FastAPI backend. Every shape here mirrors a pydantic
// model in api/main.py; when one changes, both have to.

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Role = "user" | "assistant";

export type ChatTurn = {
  role: Role;
  content: string;
};

export type Citation = {
  clause_id: string | null;
  // Null for DOCX and TXT sources: those formats carry no page boundaries.
  page_number: number | null;
  source_file: string;
};

// One clause that was in front of the model, verbatim. `cited` marks the ones
// the answer stands on; the rest explain a weak answer, which is otherwise
// only visible in the server log.
export type Excerpt = {
  clause_id: string | null;
  page_number: number | null;
  source_file: string;
  text: string;
  cited: boolean;
  // Null when the clause was pulled in beside a hit rather than found by search.
  similarity: number | null;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  standalone_query: string;
  excerpts: Excerpt[];
};

export type DocumentResponse = {
  filename: string;
  chunks_indexed: number;
  warnings: string[];
};

export async function sendChatMessage(
  message: string,
  history: ChatTurn[],
  // The documents this conversation is about. Empty searches the whole index,
  // which is what a chat that uploaded nothing of its own wants.
  sourceFiles: string[],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, source_files: sourceFiles }),
    // Aborting releases the reader, not the server: the endpoint is synchronous
    // and finishes its work either way. What is cancelled is the waiting.
    signal,
  });
  return readJson<ChatResponse>(response);
}

export async function uploadDocument(file: File): Promise<DocumentResponse> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents`, {
    method: "POST",
    body,
  });
  return readJson<DocumentResponse>(response);
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(describeError(payload, response.status));
  }
  return payload as T;
}

// FastAPI answers a raised HTTPException with a string `detail` and a failed
// request-body validation with a list of them, so both forms have to be read.
function describeError(payload: unknown, status: number): string {
  const detail = (payload as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) => (entry as { msg?: string }).msg)
      .filter(Boolean);
    if (messages.length > 0) {
      return messages.join("; ");
    }
  }
  return `Сервер ответил ошибкой ${status}.`;
}
