"""Public API for top-k retrieval over the vector index."""

from contract_rag.retriever.retriever import (
    CONTEXT_BUDGET_CHARS,
    DEFAULT_MATCH_COUNT,
    MATCH_FUNCTION_NAME,
    ContextChunk,
    RetrievedChunk,
    SupabaseRetriever,
    expand_with_related_clauses,
    retrieve,
)

__all__ = [
    "CONTEXT_BUDGET_CHARS",
    "DEFAULT_MATCH_COUNT",
    "MATCH_FUNCTION_NAME",
    "ContextChunk",
    "RetrievedChunk",
    "SupabaseRetriever",
    "expand_with_related_clauses",
    "retrieve",
]
