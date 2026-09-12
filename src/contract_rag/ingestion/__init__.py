"""Public API for indexing uploaded documents through the whole pipeline."""

from contract_rag.ingestion.ingestion import (
    PAGELESS_WARNING,
    ingest_document,
    normalize_filename,
    pageless_warnings,
)

__all__ = [
    "PAGELESS_WARNING",
    "ingest_document",
    "normalize_filename",
    "pageless_warnings",
]
