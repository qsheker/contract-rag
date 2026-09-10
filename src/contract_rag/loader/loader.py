"""Page-oriented document text extraction and deterministic cleanup."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

import docx
import pymupdf
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one page together with its source provenance.

    ``page_number`` is ``None`` for formats that carry no page boundaries.
    """

    page_number: int | None
    text: str
    source_file: str

    def __post_init__(self) -> None:
        if self.page_number is None:
            return
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
            raise ValueError("page_number must be an integer greater than or equal to 1")
        if self.page_number < 1:
            raise ValueError("page_number must be greater than or equal to 1")


class LoaderErrorCode(StrEnum):
    """Stable machine-readable codes for expected document loading failures."""

    FILE_NOT_FOUND = "file_not_found"
    PATH_IS_DIRECTORY = "path_is_directory"
    INVALID_PDF = "invalid_pdf"
    PASSWORD_PROTECTED = "password_protected"
    NO_TEXT_LAYER = "no_text_layer"
    INVALID_DOCX = "invalid_docx"
    INVALID_ENCODING = "invalid_encoding"
    UNSUPPORTED_FORMAT = "unsupported_format"


class LoaderError(Exception):
    """Expected failure while opening or reading a document."""

    def __init__(
        self,
        code: LoaderErrorCode,
        source_file: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.source_file = source_file


class NoTextLayerError(LoaderError):
    """Raised when an openable document has no extractable text."""

    def __init__(self, source_file: str, document_format: str = "PDF") -> None:
        super().__init__(
            code=LoaderErrorCode.NO_TEXT_LAYER,
            source_file=source_file,
            message=f"{document_format} has no extractable text layer: {source_file}",
        )


class UnsupportedFormatError(LoaderError):
    """Raised when a path carries an extension no loader handles."""

    def __init__(self, source_file: str, extension: str) -> None:
        self.extension = extension
        described_extension = extension or "(no extension)"
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        super().__init__(
            code=LoaderErrorCode.UNSUPPORTED_FORMAT,
            source_file=source_file,
            message=(
                f"Unsupported document format {described_extension}: {source_file}. "
                f"Supported formats: {supported}"
            ),
        )


_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE_RUN = re.compile(r"\s+")
_JOINABLE_HYPHEN = re.compile(
    r"(?P<hyphen>[-\u00ad\u2010\u2011])\r?\n[ \t]*(?P<next>[^\W\d_])",
    flags=re.UNICODE,
)

# Outer edges take precedence on very short pages so one physical line never
# contributes to more than one positional candidate on the same page.
_BOUNDARY_POSITIONS = (
    ("header_1", 0),
    ("footer_1", -1),
    ("header_2", 1),
    ("footer_2", -2),
)


def load_pdf(path: str | os.PathLike[str]) -> list[PageText]:
    """Load and clean every page of a text-layer PDF.

    Page numbers are one-based. The original path representation is retained in
    every returned ``PageText.source_file`` value.
    """

    source_file = _require_readable_file(path, "PDF")

    try:
        document = pymupdf.open(source_file)
    except FileNotFoundError as error:
        raise LoaderError(
            LoaderErrorCode.FILE_NOT_FOUND,
            source_file,
            f"PDF file does not exist: {source_file}",
        ) from error
    except IsADirectoryError as error:
        raise LoaderError(
            LoaderErrorCode.PATH_IS_DIRECTORY,
            source_file,
            f"Expected a PDF file but received a directory: {source_file}",
        ) from error
    except (pymupdf.EmptyFileError, pymupdf.FileDataError) as error:
        raise LoaderError(
            LoaderErrorCode.INVALID_PDF,
            source_file,
            f"File is not a readable PDF: {source_file}",
        ) from error

    with document:
        if not document.is_pdf:
            raise LoaderError(
                LoaderErrorCode.INVALID_PDF,
                source_file,
                f"File is not a PDF: {source_file}",
            )
        if document.needs_pass:
            raise LoaderError(
                LoaderErrorCode.PASSWORD_PROTECTED,
                source_file,
                f"PDF requires a password: {source_file}",
            )

        raw_pages = [
            PageText(
                page_number=page_index + 1,
                text=page.get_text(),
                source_file=source_file,
            )
            for page_index, page in enumerate(document)
        ]

    if not raw_pages or not any(page.text.strip() for page in raw_pages):
        raise NoTextLayerError(source_file)

    pages_without_boilerplate = strip_boilerplate(raw_pages)
    return [
        PageText(
            page_number=page.page_number,
            text=join_hyphenation(page.text),
            source_file=page.source_file,
        )
        for page in pages_without_boilerplate
    ]


def load_docx(path: str | os.PathLike[str]) -> list[PageText]:
    """Load a whole DOCX as a single page.

    DOCX stores no page boundaries: they are produced when a renderer applies
    fonts and margins, not recorded in the file. ``page_number`` is therefore
    ``None`` rather than an invented number, and citations from DOCX sources
    can carry a clause but no page.
    """

    source_file = _require_readable_file(path, "DOCX")
    try:
        document = docx.Document(source_file)
    except (PackageNotFoundError, ValueError) as error:
        raise LoaderError(
            LoaderErrorCode.INVALID_DOCX,
            source_file,
            f"File is not a readable DOCX: {source_file}",
        ) from error

    text = join_hyphenation("\n".join(_iter_docx_blocks(document))).strip()
    if not text:
        raise NoTextLayerError(source_file, document_format="DOCX")

    return [PageText(page_number=None, text=text, source_file=source_file)]


def load_txt(path: str | os.PathLike[str]) -> list[PageText]:
    """Load a whole plain-text file as a single page.

    Plain text has no notion of a page at all, so ``page_number`` is ``None``.
    """

    source_file = _require_readable_file(path, "TXT")
    try:
        # utf-8-sig also strips the BOM that Windows editors prepend.
        with open(source_file, encoding="utf-8-sig") as handle:
            raw_text = handle.read()
    except UnicodeDecodeError as error:
        raise LoaderError(
            LoaderErrorCode.INVALID_ENCODING,
            source_file,
            f"TXT file is not valid UTF-8: {source_file}",
        ) from error

    text = join_hyphenation(raw_text).strip()
    if not text:
        raise NoTextLayerError(source_file, document_format="TXT")

    return [PageText(page_number=None, text=text, source_file=source_file)]


# Registry rather than an if-chain: a new format is one entry plus its loader.
_LOADERS: dict[str, Callable[[str | os.PathLike[str]], list[PageText]]] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)

# Formats whose loaders return page_number=None, so chunks from them can be
# cited by clause but never by page. Interfaces that accept uploads need to say
# so before the user asks a question and wonders where the page went.
PAGELESS_EXTENSIONS = frozenset({".docx", ".txt"})


def load_document(path: str | os.PathLike[str]) -> list[PageText]:
    """Load any supported document, dispatching on the file extension.

    Callers that already know the format may keep using ``load_pdf``,
    ``load_docx`` or ``load_txt`` directly.
    """

    source_file = os.fspath(path)
    extension = os.path.splitext(source_file)[1].lower()
    loader = _LOADERS.get(extension)
    if loader is None:
        raise UnsupportedFormatError(source_file, extension)
    return loader(path)


def _require_readable_file(path: str | os.PathLike[str], document_format: str) -> str:
    """Return the path as a string once it is known to be an existing file."""

    source_file = os.fspath(path)
    if not os.path.exists(source_file):
        raise LoaderError(
            LoaderErrorCode.FILE_NOT_FOUND,
            source_file,
            f"{document_format} file does not exist: {source_file}",
        )
    if os.path.isdir(source_file):
        raise LoaderError(
            LoaderErrorCode.PATH_IS_DIRECTORY,
            source_file,
            f"Expected a {document_format} file but received a directory: {source_file}",
        )
    return source_file


def _iter_docx_blocks(document: docx.document.Document) -> Iterator[str]:
    """Yield paragraph and table text in document order, one line per block.

    Tables are included because contract details - parties, bank details,
    payment schedules - routinely live in them, and dropping them would lose
    text the answer must be able to cite.
    """

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document).text
        elif child.tag == qn("w:tbl"):
            for row in Table(child, document).rows:
                # One row per line keeps the line-based clause strategies working.
                yield "\t".join(_WHITESPACE_RUN.sub(" ", cell.text).strip() for cell in row.cells)


def strip_boilerplate(pages: list[PageText]) -> list[PageText]:
    """Remove repeated lines from the first or last two text positions.

    Digits and whitespace are normalized for comparison only. A candidate must
    occur in the same position on at least half of all pages and on at least two
    pages in absolute terms.
    """

    if not pages:
        return []

    source_files = {page.source_file for page in pages}
    if len(source_files) > 1:
        raise ValueError("all pages must belong to the same source_file")

    candidates_by_page = [_boundary_candidates(page.text) for page in pages]
    counts: dict[str, Counter[str]] = {position: Counter() for position, _ in _BOUNDARY_POSITIONS}
    for candidates in candidates_by_page:
        for position, (_, normalized_line) in candidates.items():
            counts[position][normalized_line] += 1

    required_occurrences = max(2, math.ceil(len(pages) * 0.5))
    qualified = {
        (position, normalized_line)
        for position, position_counts in counts.items()
        for normalized_line, count in position_counts.items()
        if count >= required_occurrences
    }

    cleaned_pages: list[PageText] = []
    for page, candidates in zip(pages, candidates_by_page, strict=True):
        removed_line_indexes = {
            line_index
            for position, (line_index, normalized_line) in candidates.items()
            if (position, normalized_line) in qualified
        }
        lines = page.text.splitlines()
        cleaned_text = "\n".join(
            line for line_index, line in enumerate(lines) if line_index not in removed_line_indexes
        ).strip()
        cleaned_pages.append(
            PageText(
                page_number=page.page_number,
                text=cleaned_text,
                source_file=page.source_file,
            )
        )

    return cleaned_pages


def join_hyphenation(text: str) -> str:
    """Join words split by one line break according to the next letter's case.

    A line-ending hyphen is removed before a lowercase letter and preserved
    before an uppercase letter. The newline and horizontal indentation are
    removed in both cases.
    """

    def replace_hyphenation(match: re.Match[str]) -> str:
        hyphen = match.group("hyphen")
        next_character = match.group("next")
        if next_character.islower():
            return next_character
        if next_character.isupper() or next_character.istitle():
            return f"{hyphen}{next_character}"
        return match.group(0)

    return _JOINABLE_HYPHEN.sub(replace_hyphenation, text)


def _normalize_boundary_line(line: str) -> str:
    normalized_whitespace = _WHITESPACE_RUN.sub(" ", line.strip())
    return _DIGIT_RUN.sub("#", normalized_whitespace)


def _boundary_candidates(text: str) -> dict[str, tuple[int, str]]:
    non_empty_lines = [
        (line_index, line) for line_index, line in enumerate(text.splitlines()) if line.strip()
    ]
    candidates: dict[str, tuple[int, str]] = {}
    used_line_indexes: set[int] = set()

    for position, relative_index in _BOUNDARY_POSITIONS:
        try:
            line_index, line = non_empty_lines[relative_index]
        except IndexError:
            continue
        if line_index in used_line_indexes:
            continue
        used_line_indexes.add(line_index)
        candidates[position] = (line_index, _normalize_boundary_line(line))

    return candidates
