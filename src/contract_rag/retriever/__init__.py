"""Public API for top-k retrieval over the vector index."""

from contract_rag.retriever.retriever import (
    DEFAULT_MATCH_COUNT,
    MATCH_FUNCTION_NAME,
    RetrievedChunk,
    SupabaseRetriever,
    retrieve,
)

__all__ = [
    "DEFAULT_MATCH_COUNT",
    "MATCH_FUNCTION_NAME",
    "RetrievedChunk",
    "SupabaseRetriever",
    "retrieve",
]
