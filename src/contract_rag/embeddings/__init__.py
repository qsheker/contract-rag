"""Public API for chunk embedding and Supabase indexing."""

from contract_rag.embeddings.embeddings import (
    CONTEXT_MAX_CHARS,
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    FILTER_BUDGET_BYTES,
    MODEL_NAME,
    MODEL_TOKEN_LIMIT,
    QUERY_PREFIX,
    RoSBERTaEmbedder,
    create_supabase_client_from_env,
    delete_stale_chunks,
    embed_and_index,
    get_default_embedder,
)

__all__ = [
    "CONTEXT_MAX_CHARS",
    "DOCUMENT_PREFIX",
    "EMBEDDING_DIMENSION",
    "FILTER_BUDGET_BYTES",
    "MODEL_NAME",
    "MODEL_TOKEN_LIMIT",
    "QUERY_PREFIX",
    "RoSBERTaEmbedder",
    "create_supabase_client_from_env",
    "delete_stale_chunks",
    "embed_and_index",
    "get_default_embedder",
]
