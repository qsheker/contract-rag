"""Top-k vector retrieval over the Supabase pgvector index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contract_rag.chunker import Chunk
from contract_rag.embeddings import (
    RoSBERTaEmbedder,
    create_supabase_client_from_env,
    get_default_embedder,
)

MATCH_FUNCTION_NAME = "match_documents"
DEFAULT_MATCH_COUNT = 5


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One search hit: the stored chunk plus how it was ranked."""

    chunk: Chunk
    chunk_id: str
    similarity: float


class SupabaseRetriever:
    """Cosine top-k search over ``contract_chunks`` via the match_documents RPC."""

    def __init__(
        self,
        supabase_client: Any,
        embedder: RoSBERTaEmbedder | None = None,
    ) -> None:
        self._client = supabase_client
        self._embedder = embedder

    def retrieve(self, query: str, k: int = DEFAULT_MATCH_COUNT) -> list[RetrievedChunk]:
        """Return at most ``k`` chunks ordered by descending similarity."""

        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("k must be an integer greater than or equal to 1")
        if k <= 0:
            raise ValueError("k must be greater than or equal to 1")

        query_embedding = self._get_embedder().embed_query(query)
        response = self._client.rpc(
            MATCH_FUNCTION_NAME,
            {"query_embedding": query_embedding, "match_count": k},
        ).execute()

        # An empty table answers with an empty payload, not an error.
        rows = getattr(response, "data", None) or []
        hits = [_to_retrieved_chunk(row) for row in rows]
        # The RPC already orders by distance; re-sorting keeps the ordering
        # guarantee at this boundary, where callers and reranking observe it.
        hits.sort(key=lambda hit: hit.similarity, reverse=True)
        return hits

    def _get_embedder(self) -> RoSBERTaEmbedder:
        if self._embedder is None:
            self._embedder = get_default_embedder()
        return self._embedder


_default_retriever: SupabaseRetriever | None = None


def _get_default_retriever() -> SupabaseRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = SupabaseRetriever(create_supabase_client_from_env())
    return _default_retriever


def retrieve(
    query: str,
    k: int = DEFAULT_MATCH_COUNT,
    *,
    supabase_client: Any | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks, building a Supabase client from the environment."""

    retriever = (
        _get_default_retriever() if supabase_client is None else SupabaseRetriever(supabase_client)
    )
    return retriever.retrieve(query, k)


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    chunk = Chunk(
        text=row["text"],
        clause_id=row["clause_id"],
        page_number=row["page_number"],
        source_file=row["source_file"],
        detected_strategy=row["detected_strategy"],
    )
    return RetrievedChunk(
        chunk=chunk,
        chunk_id=row["id"],
        similarity=float(row["similarity"]),
    )
