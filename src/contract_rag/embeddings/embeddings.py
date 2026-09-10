"""Generate retrieval embeddings and upsert chunks into Supabase."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import quote

from dotenv import load_dotenv

from contract_rag.chunker import Chunk

MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
MODEL_TOKEN_LIMIT = 512
EMBEDDING_DIMENSION = 1024
TABLE_NAME = "contract_chunks"
UPSERT_BATCH_SIZE = 50
# A chunk shorter than this is a fragment and gets its headings prepended
# before encoding; a longer one already carries its own framing. Measured,
# not guessed: prepending headings to every chunk broadened the long ones
# topically enough to push correct answers out of top-5 and cost the eval set
# one question (12/14 -> 11/14). The regression appears above ~600 characters,
# and 400 still frames 93% of the real sub-item fragments.
CONTEXT_MAX_CHARS = 400
# Room for one delete filter in the query string, well under the request-line
# limit a gateway will accept. Only stale ids ever go here, so it is rarely hit.
FILTER_BUDGET_BYTES = 3000

logger = logging.getLogger(__name__)


class _SentenceTransformerModel(Protocol):
    """The small portion of SentenceTransformer used by the wrapper."""

    max_seq_length: int
    tokenizer: Any

    def get_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk_id: str
    chunk: Chunk


class RoSBERTaEmbedder:
    """Lazy-compatible wrapper around the fixed ru-en-RoSBERTa model."""

    def __init__(self, model: _SentenceTransformerModel | None = None) -> None:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = cast(_SentenceTransformerModel, SentenceTransformer(MODEL_NAME))

        dimension = model.get_embedding_dimension()
        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"{MODEL_NAME} must produce {EMBEDDING_DIMENSION}-dimensional embeddings; "
                f"received {dimension}"
            )

        model.max_seq_length = min(model.max_seq_length, MODEL_TOKEN_LIMIT)
        self._model = model

    def embed_documents(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed chunk text, framing short chunks with their headings.

        What is encoded is not quite what is stored: a short chunk's ancestor
        headings are prepended so a sub-item is searchable by what it is about,
        while the stored ``text`` - and therefore every citation - stays exactly
        what the document says. See ``Chunk.context`` and ``CONTEXT_MAX_CHARS``.
        """

        prefixed_texts: list[str] = []
        for chunk in chunks:
            prefixed_text = f"{DOCUMENT_PREFIX}{_with_context(chunk)}"
            if self._token_count(prefixed_text) > self._model.max_seq_length:
                logger.warning(
                    "Embedding input truncated to %d tokens: source_file=%s clause_id=%s",
                    self._model.max_seq_length,
                    chunk.source_file,
                    chunk.clause_id,
                )
            prefixed_texts.append(prefixed_text)

        return self._encode(prefixed_texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query with the prefix asymmetric retrieval requires."""

        prefixed_query = f"{QUERY_PREFIX}{query}"
        if self._token_count(prefixed_query) > self._model.max_seq_length:
            logger.warning(
                "Query input truncated to %d tokens",
                self._model.max_seq_length,
            )
        return self._encode([prefixed_query])[0]

    def _encode(self, prefixed_texts: list[str]) -> list[list[float]]:
        if not prefixed_texts:
            return []

        encoded = self._model.encode(
            prefixed_texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        result = [[float(value) for value in vector] for vector in vectors]
        for vector in result:
            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"{MODEL_NAME} returned an embedding with {len(vector)} dimensions; "
                    f"expected {EMBEDDING_DIMENSION}"
                )
        return result

    def _token_count(self, text: str) -> int:
        tokenized = self._model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        input_ids = tokenized["input_ids"]
        return len(input_ids)


_default_embedder: RoSBERTaEmbedder | None = None


def get_default_embedder() -> RoSBERTaEmbedder:
    """Return the process-wide embedder, loading the model on first use."""

    global _default_embedder
    if _default_embedder is None:
        _default_embedder = RoSBERTaEmbedder()
    return _default_embedder


def create_supabase_client_from_env() -> Any:
    """Create a Supabase client from local environment variables or ``.env``."""

    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    missing = [name for name, value in (("SUPABASE_URL", url), ("SUPABASE_KEY", key)) if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    from supabase import create_client

    return create_client(url, key)


def embed_and_index(chunks: list[Chunk], supabase_client: Any) -> list[str]:
    """Embed every unique chunk, upsert it into Supabase and return its id.

    The returned ids are what a caller re-indexing a whole document needs in
    order to recognise the rows that are no longer part of it - see
    ``delete_stale_chunks``.
    """

    indexed_chunks = _assign_chunk_ids(chunks)
    if not indexed_chunks:
        return []

    embeddings = get_default_embedder().embed_documents(
        [indexed_chunk.chunk for indexed_chunk in indexed_chunks]
    )
    rows = [
        {
            "id": indexed_chunk.chunk_id,
            "text": indexed_chunk.chunk.text,
            "embedding": embedding,
            "clause_id": indexed_chunk.chunk.clause_id,
            "page_number": indexed_chunk.chunk.page_number,
            "source_file": indexed_chunk.chunk.source_file,
            "detected_strategy": indexed_chunk.chunk.detected_strategy,
        }
        for indexed_chunk, embedding in zip(indexed_chunks, embeddings, strict=True)
    ]

    for batch in _batched(rows, UPSERT_BATCH_SIZE):
        supabase_client.table(TABLE_NAME).upsert(batch, on_conflict="id").execute()

    return [indexed_chunk.chunk_id for indexed_chunk in indexed_chunks]


def delete_stale_chunks(
    source_file: str,
    keep_chunk_ids: Collection[str],
    supabase_client: Any,
) -> None:
    """Drop rows of ``source_file`` that the latest indexing run did not write.

    Upsert alone keeps a document free of duplicates but not free of leftovers:
    re-indexing an edited file leaves the rows of clauses it no longer contains,
    and a retrieved leftover would be cited as if it were still in the contract.
    Call this only after the upsert has succeeded, so a failure mid-run leaves
    the previous version of the document in place instead of nothing.

    The stale ids are worked out here rather than handed to PostgREST as a
    "not in (survivors)" filter. Every filter value travels in the query string,
    and ids are ``source_file::clause_id``: a long non-ASCII file name expands
    to nine URL-encoded bytes per character, so listing the survivors of a
    whole document overruns the request line and the gateway answers a bare
    ``Bad Request``. Naming only what has to go usually means naming nothing:
    a first upload has no stale rows and issues no delete at all.
    """

    kept = set(keep_chunk_ids)
    response = (
        supabase_client.table(TABLE_NAME).select("id").eq("source_file", source_file).execute()
    )
    stale_ids = [
        row["id"] for row in (getattr(response, "data", None) or []) if row["id"] not in kept
    ]
    if not stale_ids:
        return

    for batch in _batched_filter_values(stale_ids):
        supabase_client.table(TABLE_NAME).delete().in_("id", batch).execute()

    logger.info("Removed %d stale chunk(s) of %s", len(stale_ids), source_file)


def _batched_filter_values(values: Sequence[str]) -> Iterable[list[str]]:
    """Group ids into batches whose URL-encoded length stays within the budget.

    Batched by encoded size rather than by count: how many ids fit depends on
    the file name, and a fixed count that works for ``contract.pdf`` overruns
    the request line for a long Cyrillic one.
    """

    batch: list[str] = []
    batch_length = 0
    for value in values:
        # Every id contains "::", so PostgREST quotes it; the quotes are encoded
        # too. One extra byte covers the separating comma.
        encoded_length = len(quote(f'"{value}"')) + 1
        if batch and batch_length + encoded_length > FILTER_BUDGET_BYTES:
            yield batch
            batch = []
            batch_length = 0
        batch.append(value)
        batch_length += encoded_length
    if batch:
        yield batch


def _assign_chunk_ids(chunks: Sequence[Chunk]) -> list[_IndexedChunk]:
    unique_chunks = list(dict.fromkeys(chunks))
    chunks_by_base_id: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in unique_chunks:
        chunks_by_base_id[_base_chunk_id(chunk)].append(chunk)

    assigned_ids: dict[Chunk, str] = {}
    for base_id, colliding_chunks in chunks_by_base_id.items():
        ordered_chunks = sorted(
            colliding_chunks,
            key=lambda chunk: (
                chunk.page_number,
                chunk.detected_strategy,
                chunk.text,
            ),
        )
        for occurrence, chunk in enumerate(ordered_chunks, start=1):
            assigned_ids[chunk] = (
                base_id
                if occurrence == 1
                else f"{base_id}::page-{chunk.page_number}::occurrence-{occurrence}"
            )

    return [_IndexedChunk(chunk_id=assigned_ids[chunk], chunk=chunk) for chunk in unique_chunks]


def _with_context(chunk: Chunk) -> str:
    """Return the text to encode, framed by the chunk's headings when it is short."""

    if not chunk.context or len(chunk.text) >= CONTEXT_MAX_CHARS:
        return chunk.text
    return f"{chunk.context}\n{chunk.text}"


def _base_chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source_file}::{chunk.clause_id or 'preamble'}"


def _batched(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])
