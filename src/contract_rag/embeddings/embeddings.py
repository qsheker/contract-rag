"""Generate retrieval embeddings and upsert chunks into Supabase."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from dotenv import load_dotenv

from contract_rag.chunker import Chunk

MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
MODEL_TOKEN_LIMIT = 512
EMBEDDING_DIMENSION = 1024
TABLE_NAME = "contract_chunks"
UPSERT_BATCH_SIZE = 50

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
        """Embed clean chunk text with the required retrieval prefix."""

        prefixed_texts: list[str] = []
        for chunk in chunks:
            prefixed_text = f"{DOCUMENT_PREFIX}{chunk.text}"
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


def embed_and_index(chunks: list[Chunk], supabase_client: Any) -> None:
    """Embed every unique chunk and idempotently upsert it into Supabase."""

    indexed_chunks = _assign_chunk_ids(chunks)
    if not indexed_chunks:
        return

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


def _base_chunk_id(chunk: Chunk) -> str:
    return f"{chunk.source_file}::{chunk.clause_id or 'preamble'}"


def _batched(rows: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])
