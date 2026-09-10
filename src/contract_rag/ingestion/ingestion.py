"""Index an uploaded document end to end: load, chunk, embed, upsert."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from contract_rag.chunker import chunk_by_clause
from contract_rag.embeddings import (
    create_supabase_client_from_env,
    delete_stale_chunks,
    embed_and_index,
)
from contract_rag.loader import (
    PAGELESS_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    PageText,
    UnsupportedFormatError,
    load_document,
)

logger = logging.getLogger(__name__)

PAGELESS_WARNING = (
    "This format carries no page boundaries, so citations from {filename} will "
    "name a clause but no page number."
)


def ingest_document(
    file_bytes: bytes,
    filename: str,
    *,
    supabase_client: Any | None = None,
) -> int:
    """Index one uploaded document and return how many chunks it produced.

    The whole pipeline runs before anything is written: an unreadable file or a
    numbering the chunker does not support raises here, with nothing indexed.
    """

    source_file = normalize_filename(filename)
    extension = _require_supported_extension(source_file)

    pages = _load_from_bytes(file_bytes, source_file, extension)
    chunks = chunk_by_clause(pages)

    client = create_supabase_client_from_env() if supabase_client is None else supabase_client
    indexed_ids = embed_and_index(chunks, client)
    # Only after the upsert has landed: see delete_stale_chunks on why the order
    # matters for a re-upload that fails halfway.
    delete_stale_chunks(source_file, indexed_ids, client)

    logger.info("Indexed %d chunk(s) from %s", len(indexed_ids), source_file)
    return len(indexed_ids)


def normalize_filename(filename: str) -> str:
    """Reduce an uploaded name to the bare basename used as chunk provenance.

    The name ends up in ``Chunk.source_file``, in the row id and in citations
    shown to the user, so directory components - whether a browser's own path or
    a traversal attempt - are dropped rather than stored.
    """

    source_file = os.path.basename(filename.replace("\\", "/").strip())
    if not source_file or source_file in {".", ".."}:
        raise ValueError(f"Upload carries no usable file name: {filename!r}")
    return source_file


def pageless_warnings(filename: str) -> list[str]:
    """Return the warnings a caller must show for ``filename``, if any."""

    extension = os.path.splitext(filename)[1].lower()
    if extension not in PAGELESS_EXTENSIONS:
        return []
    return [PAGELESS_WARNING.format(filename=os.path.basename(filename))]


def _require_supported_extension(source_file: str) -> str:
    """Reject an unsupported format before a temporary file is even written."""

    extension = os.path.splitext(source_file)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(source_file, extension)
    return extension


def _load_from_bytes(file_bytes: bytes, source_file: str, extension: str) -> list[PageText]:
    """Load bytes through the path-based loaders, keeping the original name.

    The loaders open paths, so the upload is spooled to a temporary directory
    under its own basename: that keeps the name inside any loader error message
    recognisable. The loaded pages are then re-stamped with ``source_file``,
    because the temporary directory changes on every upload while the row id
    (``source_file::clause_id``) must not - otherwise re-uploading the same file
    would index a second copy instead of replacing the first.
    """

    with tempfile.TemporaryDirectory(prefix="contract-rag-upload-") as directory:
        temporary_path = os.path.join(directory, source_file)
        with open(temporary_path, "wb") as handle:
            handle.write(file_bytes)
        pages = load_document(temporary_path)

    return [
        PageText(page_number=page.page_number, text=page.text, source_file=source_file)
        for page in pages
    ]
