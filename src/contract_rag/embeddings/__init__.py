"""Public API for chunk embedding and Supabase indexing."""

from contract_rag.embeddings.embeddings import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    MODEL_TOKEN_LIMIT,
    RoSBERTaEmbedder,
    create_supabase_client_from_env,
    embed_and_index,
)

__all__ = [
    "DOCUMENT_PREFIX",
    "EMBEDDING_DIMENSION",
    "MODEL_NAME",
    "MODEL_TOKEN_LIMIT",
    "RoSBERTaEmbedder",
    "create_supabase_client_from_env",
    "embed_and_index",
]
