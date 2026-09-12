"""FastAPI endpoints over the retrieval, generation and indexing pipeline."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from api.reformulate import reformulate_query
from contract_rag.chunker import UnsupportedNumberingError
from contract_rag.embeddings import get_default_embedder
from contract_rag.generator import (
    Citation,
    GenerationError,
    UnsupportedCitationError,
    generate_answer,
)
from contract_rag.ingestion import ingest_document, normalize_filename, pageless_warnings
from contract_rag.loader import LoaderError, LoaderErrorCode, UnsupportedFormatError
from contract_rag.retriever import (
    DEFAULT_MATCH_COUNT,
    ContextChunk,
    expand_with_related_clauses,
    retrieve,
)

# uvicorn configures only its own loggers, leaving the root logger on the
# WARNING-level fallback handler: without this every logger.info in the project
# - the lifespan progress, the indexed-chunk counts, the stale-row cleanup -
# goes nowhere. This is the application entry point, so configuring the root
# logger here is its job rather than a library's.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

# The Next.js dev server. Listed explicitly rather than via a wildcard: the API
# talks to Supabase and a generation provider on the developer's behalf.
ALLOWED_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")

# Uploads are read into memory before they reach the loaders, so the ceiling is
# what keeps one oversized file from taking the process down.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Composed from the error code instead of reusing str(error): loader messages
# name the temporary path the upload was spooled to, which means nothing here.
LOADER_ERROR_MESSAGES: dict[LoaderErrorCode, str] = {
    LoaderErrorCode.INVALID_PDF: "{filename} is not a readable PDF.",
    LoaderErrorCode.PASSWORD_PROTECTED: "{filename} is password protected.",
    LoaderErrorCode.NO_TEXT_LAYER: (
        "{filename} carries no text layer - it is most likely a scan, and OCR is "
        "not part of the pipeline."
    ),
    LoaderErrorCode.INVALID_DOCX: "{filename} is not a readable DOCX.",
    LoaderErrorCode.INVALID_ENCODING: "{filename} is not valid UTF-8 text.",
}


class ChatMessage(BaseModel):
    """One earlier turn of the conversation, as the client remembers it."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """A new question plus the history it may be a follow-up to."""

    message: str
    history: list[ChatMessage] = []
    # The documents this conversation is about, as the client remembers them.
    # Empty means every indexed document, which is what a chat that uploaded
    # nothing of its own still wants.
    source_files: list[str] = []

    @field_validator("message")
    @classmethod
    def _require_content(cls, message: str) -> str:
        stripped = message.strip()
        if not stripped:
            raise ValueError("message must not be empty")
        return stripped


class Excerpt(BaseModel):
    """One clause the answer was written from, verbatim.

    ``cited`` marks the ones the answer actually stands on; the rest are what
    was in front of the model and went unused. Both are returned because a
    citation the reader cannot open is only half a citation, and because a poor
    answer is explained by what search supplied - which is otherwise invisible
    outside the server log.
    """

    clause_id: str | None
    page_number: int | None
    source_file: str
    text: str
    cited: bool
    # None when the clause was pulled in as a neighbour of a hit rather than
    # found by search itself.
    similarity: float | None


class ChatResponse(BaseModel):
    """The answer, what it cites, and the query retrieval actually ran on."""

    answer: str
    citations: list[Citation]
    standalone_query: str
    excerpts: list[Excerpt]


class DocumentResponse(BaseModel):
    """What indexing one upload produced, and what the user should know about it."""

    filename: str
    chunks_indexed: int
    warnings: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the embedding model before the first request rather than during it.

    ru-en-RoSBERTa takes tens of seconds to load. Paying that on start-up keeps
    it out of whichever request happens to arrive first, where it would look
    like a hung chat.
    """

    logger.info("Loading the embedding model")
    get_default_embedder()
    logger.info("Embedding model ready")
    yield


async def report_unexpected_errors(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Turn an unhandled exception into a JSON 500 the browser can actually read.

    Starlette's own 500 is produced outside the middleware stack, so it carries
    no CORS headers and the browser rejects it before the page sees a status -
    every backend crash then reads as "Failed to fetch" in the UI. Answering
    here, inside CORS, keeps the reason visible where the user is looking.
    """

    try:
        return await call_next(request)
    except Exception as error:
        logger.exception("Unhandled error serving %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            # The message itself, not a placeholder: this runs on the
            # developer's own machine, and "see the server log" just moves the
            # answer to another window.
            content={"detail": f"{type(error).__name__}: {error}"},
        )


app = FastAPI(title="contract-rag", lifespan=lifespan)
# Added first so CORS ends up wrapping it: FastAPI treats the last-added
# middleware as the outermost one, and the 500 above needs CORS headers.
app.add_middleware(BaseHTTPMiddleware, dispatch=report_unexpected_errors)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(ALLOWED_ORIGINS),
    allow_methods=["POST"],
    allow_headers=["*"],
)


# Deliberately `def`, not `async def`: retrieval, generation and indexing all
# block for seconds, and Starlette runs a sync endpoint in a worker thread
# instead of stalling the event loop for every other request.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer one message in the context of its conversation."""

    history = [turn.model_dump() for turn in request.history]
    standalone_query = reformulate_query(history, request.message)

    hits = retrieve(
        standalone_query,
        k=DEFAULT_MATCH_COUNT,
        source_files=request.source_files or None,
    )
    # Exact clause boundaries leave a lead-in like "5.2. Работник обязан:" as a
    # chunk of its own: the best match for a question whose answer is entirely
    # in its sub-items. The expansion hands those over too.
    context = expand_with_related_clauses(hits)
    try:
        answer = generate_answer(standalone_query, [entry.chunk for entry in context])
    except UnsupportedCitationError as error:
        # The rejection is correct and the detail - a list of row ids - belongs
        # in the log, not on screen. What the reader needs is why they have no
        # answer and what to do about it.
        logger.warning("Answer withheld: %s", error)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Модель сослалась на пункты, которых нет в найденных фрагментах, "
                "поэтому ответ отклонён — цитата, которую нельзя проверить, хуже "
                "отсутствия ответа. Попробуйте задать вопрос конкретнее."
            ),
        ) from error
    except GenerationError as error:
        logger.warning("Generation failed: %s", error)
        # 502: the request was fine, the model behind it was not.
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="Модель генерации не ответила или вернула непригодный результат.",
        ) from error

    return ChatResponse(
        answer=answer.text,
        citations=answer.citations,
        standalone_query=standalone_query,
        excerpts=_describe_excerpts(context, answer.citations),
    )


@app.post("/documents", response_model=DocumentResponse)
def upload_document(file: UploadFile = File(...)) -> DocumentResponse:  # noqa: B008
    """Index one uploaded document so it can be asked about immediately.

    Synchronous on purpose: at the scale of a single-user project a queue would
    add a moving part without removing the wait.
    """

    filename = _require_filename(file.filename)
    file_bytes = _read_upload(file)

    try:
        chunks_indexed = ingest_document(file_bytes, filename)
    except UnsupportedFormatError as error:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{filename}: only PDF, DOCX and TXT files can be indexed.",
        ) from error
    except LoaderError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_describe_loader_error(error, filename),
        ) from error
    except UnsupportedNumberingError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"{filename}: no clause numbering the chunker recognises, so the "
                f"document cannot be split into citable chunks."
            ),
        ) from error

    return DocumentResponse(
        filename=filename,
        chunks_indexed=chunks_indexed,
        warnings=pageless_warnings(filename),
    )


def _describe_excerpts(
    context: list[ContextChunk],
    citations: list[Citation],
) -> list[Excerpt]:
    """Mark each supplied clause as cited or not, by the same key the check uses.

    Deliberately the triple ``generate_answer`` verifies against: if the answer
    passed that check, every citation matches an excerpt here, and the UI cannot
    end up showing a citation it has no text for.
    """

    cited_keys = {
        (citation.source_file, citation.clause_id, citation.page_number) for citation in citations
    }
    return [
        Excerpt(
            clause_id=entry.chunk.clause_id,
            page_number=entry.chunk.page_number,
            source_file=entry.chunk.source_file,
            text=entry.chunk.text,
            cited=(
                entry.chunk.source_file,
                entry.chunk.clause_id,
                entry.chunk.page_number,
            )
            in cited_keys,
            similarity=entry.similarity,
        )
        for entry in context
    ]


def _require_filename(filename: str | None) -> str:
    try:
        return normalize_filename(filename or "")
    except ValueError as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="The upload carries no usable file name.",
        ) from error


def _read_upload(file: UploadFile) -> bytes:
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"The uploaded file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return file_bytes


def _describe_loader_error(error: LoaderError, filename: str) -> str:
    template = LOADER_ERROR_MESSAGES.get(error.code, "{filename} could not be read.")
    return template.format(filename=filename)
