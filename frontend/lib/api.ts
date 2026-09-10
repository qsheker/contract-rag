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

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  standalone_query: string;
};

export type DocumentResponse = {
  filename: string;
  chunks_indexed: number;
  warnings: string[];
};

export async function sendChatMessage(
  message: string,
  history: ChatTurn[],
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
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
