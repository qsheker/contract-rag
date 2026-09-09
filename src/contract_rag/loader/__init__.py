"""Public API for page-oriented document loading and cleanup."""

from contract_rag.loader.loader import (
    SUPPORTED_EXTENSIONS,
    LoaderError,
    LoaderErrorCode,
    NoTextLayerError,
    PageText,
    UnsupportedFormatError,
    join_hyphenation,
    load_document,
    load_docx,
    load_pdf,
    load_txt,
    strip_boilerplate,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "LoaderError",
    "LoaderErrorCode",
    "NoTextLayerError",
    "PageText",
    "UnsupportedFormatError",
    "join_hyphenation",
    "load_document",
    "load_docx",
    "load_pdf",
    "load_txt",
    "strip_boilerplate",
]
