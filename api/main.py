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
from contract_rag.generator import Citation, GenerationError, generate_answer
from contract_rag.ingestion import ingest_document, normalize_filename, pageless_warnings
from contract_rag.loader import LoaderError, LoaderErrorCode, UnsupportedFormatError
from contract_rag.retriever import DEFAULT_MATCH_COUNT, retrieve

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

    @field_validator("message")
    @classmethod
    def _require_content(cls, message: str) -> str:
        stripped = message.strip()
        if not stripped:
            raise ValueError("message must not be empty")
        return stripped


class ChatResponse(BaseModel):
    """The answer, what it cites, and the query retrieval actually ran on."""

    answer: str
    citations: list[Citation]
    standalone_query: str


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
    allow_methods=["GET", "POST"],
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

    hits = retrieve(standalone_query, k=DEFAULT_MATCH_COUNT)
    try:
        answer = generate_answer(standalone_query, [hit.chunk for hit in hits])
    except GenerationError as error:
        # 502: the request was fine, the model behind it was not.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error

    return ChatResponse(
        answer=answer.text,
        citations=answer.citations,
        standalone_query=standalone_query,
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
