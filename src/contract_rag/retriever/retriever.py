"""Top-k vector retrieval over the Supabase pgvector index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from contract_rag.chunker import Chunk
from contract_rag.embeddings import (
    TABLE_NAME,
    RoSBERTaEmbedder,
    create_supabase_client_from_env,
    get_default_embedder,
)

MATCH_FUNCTION_NAME = "match_documents"
DEFAULT_MATCH_COUNT = 5
# How much clause text the expansion may add around the hits. Ollama serves
# qwen2.5 with a 4096-token window by default regardless of what the model
# itself supports, and an over-long prompt is truncated silently - taking the
# system prompt with it. This leaves room for the rules, the schema and the
# answer.
CONTEXT_BUDGET_CHARS = 5000
# PostgREST answers at most 1000 rows per request.
_PAGE_SIZE = 1000
_ROW_COLUMNS = "id,text,clause_id,page_number,source_file,detected_strategy"


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

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_MATCH_COUNT,
        *,
        source_files: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return at most ``k`` chunks ordered by descending similarity.

        ``source_files`` narrows the search to those documents; ``None`` searches
        every indexed one. The filter is applied inside the RPC rather than to
        its result, so ``k`` means k within the scope instead of whatever
        survives filtering someone else's top-k.
        """

        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("k must be an integer greater than or equal to 1")
        if k <= 0:
            raise ValueError("k must be greater than or equal to 1")

        query_embedding = self._get_embedder().embed_query(query)
        response = self._client.rpc(
            MATCH_FUNCTION_NAME,
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "source_files": list(source_files) if source_files else None,
            },
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
    source_files: Sequence[str] | None = None,
    supabase_client: Any | None = None,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks, building a Supabase client from the environment."""

    retriever = (
        _get_default_retriever() if supabase_client is None else SupabaseRetriever(supabase_client)
    )
    return retriever.retrieve(query, k, source_files=source_files)


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """One chunk handed to generation, together with why it is there.

    ``similarity`` is the retrieval score when search found this chunk itself,
    and ``None`` when it was pulled in afterwards because a retrieved clause
    cannot be read without it.
    """

    chunk: Chunk
    chunk_id: str
    similarity: float | None

    @property
    def was_retrieved(self) -> bool:
        return self.similarity is not None


def expand_with_related_clauses(
    hits: Sequence[RetrievedChunk],
    *,
    supabase_client: Any | None = None,
    budget_chars: int = CONTEXT_BUDGET_CHARS,
) -> list[ContextChunk]:
    """Add the clauses a hit is unreadable without: its sub-items and headings.

    Exact clause boundaries make attribution right and make a lead-in useless on
    its own. "5.2. The employee shall:" is a correct hit for "what will I have
    to do at work" and answers nothing; the duties are in 5.2.1 to 5.2.5, and
    each of those alone matches the question too weakly to be retrieved. They
    are gathered by clause number rather than by similarity, which is
    deterministic and keeps every sub-item's own ``clause_id`` - so a citation
    still names the exact sub-item rather than the heading above it.

    This is deliberately not part of ``retrieve``: what search found and what
    generation was given are different facts, and the eval set measures the
    first. Kept separate, adding neighbours cannot flatter a retrieval score.
    """

    if not hits:
        return []

    client = create_supabase_client_from_env() if supabase_client is None else supabase_client
    rows_by_source = {
        source_file: _fetch_document_rows(source_file, client)
        for source_file in dict.fromkeys(hit.chunk.source_file for hit in hits)
    }

    context = [
        ContextChunk(chunk=hit.chunk, chunk_id=hit.chunk_id, similarity=hit.similarity)
        for hit in hits
    ]
    taken = {entry.chunk_id for entry in context}
    used_chars = sum(len(entry.chunk.text) for entry in context)

    for candidate, chunk_id in _related_candidates(hits, rows_by_source):
        if chunk_id in taken:
            continue
        if used_chars + len(candidate.text) > budget_chars:
            # Skip rather than stop: a later hit's short sub-item still fits
            # where one long clause did not.
            continue
        context.append(ContextChunk(chunk=candidate, chunk_id=chunk_id, similarity=None))
        taken.add(chunk_id)
        used_chars += len(candidate.text)

    return context


def _related_candidates(
    hits: Sequence[RetrievedChunk],
    rows_by_source: dict[str, list[dict[str, Any]]],
) -> list[tuple[Chunk, str]]:
    """Order every relative of every hit by how much it is needed.

    Sorted across all hits at once rather than hit by hit, so one hit whose
    section runs long cannot eat the budget a later hit's sub-item needed. The
    key is: the better-ranked the hit, the closer the relative to it, and then
    document order among equals.
    """

    scored: list[tuple[tuple[int, int, tuple[Any, ...]], Chunk, str]] = []
    for hit_rank, hit in enumerate(hits):
        clause_id = hit.chunk.clause_id
        if not clause_id:
            continue
        for row in rows_by_source.get(hit.chunk.source_file, []):
            distance = _relative_distance(clause_id, row["clause_id"])
            if distance is None:
                continue
            sort_key = (hit_rank, distance, _clause_sort_key(row["clause_id"]))
            scored.append((sort_key, _to_chunk(row), row["id"]))

    scored.sort(key=lambda entry: entry[0])
    return [(chunk, chunk_id) for _, chunk, chunk_id in scored]


def _relative_distance(clause_id: str, other_id: str | None) -> int | None:
    """Return how far ``other_id`` sits from ``clause_id``, or None if unrelated.

    Descendants come first because they carry the substance a lead-in only
    announces; ancestors follow, because a sub-item read alone needs the
    sentence it continues.
    """

    if not other_id or other_id == clause_id:
        return None
    if other_id.startswith(f"{clause_id}."):
        return other_id.count(".") - clause_id.count(".")
    if clause_id.startswith(f"{other_id}."):
        return 10 + clause_id.count(".") - other_id.count(".")
    return None


def _clause_sort_key(clause_id: str | None) -> tuple[Any, ...]:
    """Sort 4.9.10 after 4.9.2, which string ordering would not."""

    if not clause_id:
        return (1, ())
    parts = clause_id.split(".")
    if all(part.isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (1, tuple(parts))


def _fetch_document_rows(source_file: str, supabase_client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        response = (
            supabase_client.table(TABLE_NAME)
            .select(_ROW_COLUMNS)
            .eq("source_file", source_file)
            .range(start, start + _PAGE_SIZE - 1)
            .execute()
        )
        page = getattr(response, "data", None) or []
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            return rows
        start += _PAGE_SIZE


def _to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        text=row["text"],
        clause_id=row["clause_id"],
        page_number=row["page_number"],
        source_file=row["source_file"],
        detected_strategy=row["detected_strategy"],
    )


def _to_retrieved_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_to_chunk(row),
        chunk_id=row["id"],
        similarity=float(row["similarity"]),
    )
