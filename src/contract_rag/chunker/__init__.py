"""Public API for generic clause chunking."""

from contract_rag.chunker.chunker import (
    Chunk,
    ChunkStrategy,
    DottedNumberingStrategy,
    HeadingOnlyStrategy,
    UnsupportedNumberingError,
    VerboseNumberingStrategy,
    chunk_by_clause,
)

__all__ = [
    "Chunk",
    "ChunkStrategy",
    "DottedNumberingStrategy",
    "HeadingOnlyStrategy",
    "UnsupportedNumberingError",
    "VerboseNumberingStrategy",
    "chunk_by_clause",
]
