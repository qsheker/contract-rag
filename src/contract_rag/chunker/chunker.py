"""Generic clause chunking with ordered detection strategies."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from contract_rag.loader import PageText


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable document fragment anchored to its starting page.

    ``page_number`` is ``None`` when the source format has no pages (DOCX, TXT);
    such a chunk can still be cited by clause, but not by page.
    """

    text: str
    clause_id: str | None
    page_number: int | None
    source_file: str
    detected_strategy: str

    def __post_init__(self) -> None:
        if self.page_number is None:
            return
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
            raise ValueError("page_number must be an integer greater than or equal to 1")
        if self.page_number < 1:
            raise ValueError("page_number must be greater than or equal to 1")
        if not self.detected_strategy:
            raise ValueError("detected_strategy must not be empty")


class UnsupportedNumberingError(Exception):
    """Raised when no configured strategy finds enough structural boundaries."""

    def __init__(self, source_file: str | None) -> None:
        self.source_file = source_file
        source_description = source_file or "the supplied document"
        super().__init__(f"No supported numbering convention found in {source_description}")


@dataclass(frozen=True, slots=True)
class _LineRecord:
    text: str
    page_number: int | None


@dataclass(frozen=True, slots=True)
class _Boundary:
    line_index: int
    clause_id: str
    page_number: int | None


class ChunkStrategy(ABC):
    """Interface for one clause-boundary detection strategy."""

    name: ClassVar[str]
    minimum_matches: ClassVar[int] = 3

    def detect(self, pages: list[PageText]) -> list[Chunk] | None:
        """Return chunks when this strategy finds enough boundaries."""

        source_file = _validate_pages(pages)
        line_records = _flatten_pages(pages)
        boundaries = [
            _Boundary(
                line_index=line_index,
                clause_id=clause_id,
                page_number=line.page_number,
            )
            for line_index, line in enumerate(line_records)
            if (clause_id := self.match_clause_id(line.text)) is not None
        ]
        if len(boundaries) < self.minimum_matches:
            return None

        return _build_chunks(
            line_records=line_records,
            boundaries=boundaries,
            source_file=source_file,
            strategy_name=self.name,
        )

    @abstractmethod
    def match_clause_id(self, line: str) -> str | None:
        """Return a clause identifier when ``line`` is a boundary."""


class DottedNumberingStrategy(ChunkStrategy):
    """Detect dotted clause identifiers such as ``4.2`` and ``4.2.1``.

    Also detects the single-level section headings those clauses hang under
    (``6. RIGHTS AND OBLIGATIONS``), because without them a whole section lands
    inside the last clause of the previous one and every fact in it gets cited
    under that clause's number.
    """

    name = "dotted_numbering"
    _clause_pattern = re.compile(
        r"^\s*(?P<clause_id>\d+\.\d+(?:\.\d+)?)(?P<terminator>\.(?!\d))?(?=\s|$)"
    )
    _section_pattern = re.compile(r"^\s*(?P<clause_id>\d{1,2})\.(?=\s|$)")

    def match_clause_id(self, line: str) -> str | None:
        clause_match = self._clause_pattern.match(line)
        if clause_match:
            return (
                clause_match.group("clause_id")
                if self._starts_a_clause(line, clause_match)
                else None
            )

        # Tried second on purpose: "5.1." must be read as clause 5.1, not as
        # section 5 followed by a stray "1.".
        section_match = self._section_pattern.match(line)
        if section_match and _looks_like_heading(line[section_match.end() :]):
            return section_match.group("clause_id")
        return None

    @staticmethod
    def _starts_a_clause(line: str, match: re.Match[str]) -> bool:
        """Tell a numbered item apart from a cross-reference wrapped onto a line.

        A PDF line break can push a reference to the start of a line - "...in
        accordance with clause\n6.2.6 shall be applied" - where it looks exactly
        like a clause opening. Two things distinguish a real item:

        * it terminates its own number ("4.9.3. для возмещения..."), which prose
          quoting a clause number does not do;
        * or the text after it starts a new sentence with a capital.

        Requiring the capital alone, as this did before, threw away every Russian
        sub-item that continues the parent clause's sentence in lower case - 125
        of 231 boundaries in one real employment contract.
        """

        if match.group("terminator"):
            return True
        remainder = line[match.end() :].strip()
        first_following_letter = next(
            (character for character in remainder if character.isalpha()),
            None,
        )
        return first_following_letter is None or not first_following_letter.islower()


class VerboseNumberingStrategy(ChunkStrategy):
    """Detect explicit Article and Section labels."""

    name = "verbose_numbering"
    _combined_pattern = re.compile(
        r"^\s*Article\s+(?P<article>[IVXLCDM]+|\d+)\s*,\s*"
        r"Section\s+(?P<section>\d+(?:\.\d+)*)\.?(?=\s|$)",
        flags=re.IGNORECASE,
    )
    _article_pattern = re.compile(
        r"^\s*Article\s+(?P<article>[IVXLCDM]+|\d+)\.?(?=\s|$)",
        flags=re.IGNORECASE,
    )
    _section_pattern = re.compile(
        r"^\s*Section\s+(?P<section>\d+(?:\.\d+)*)\.?(?=\s|$)",
        flags=re.IGNORECASE,
    )

    def match_clause_id(self, line: str) -> str | None:
        combined_match = self._combined_pattern.match(line)
        if combined_match:
            article_id = combined_match.group("article").upper()
            return f"Article {article_id}, Section {combined_match.group('section')}"

        section_match = self._section_pattern.match(line)
        if section_match:
            return f"Section {section_match.group('section')}"

        article_match = self._article_pattern.match(line)
        if article_match:
            return f"Article {article_match.group('article').upper()}"

        return None


class HeadingOnlyStrategy(ChunkStrategy):
    """Detect short title-like lines without numbering markers."""

    name = "heading_only"
    _leading_marker = re.compile(r"^(?:\d|[•●▪◦]|[oO]\s+|\([^)]+\))")
    _word = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

    def match_clause_id(self, line: str) -> str | None:
        heading = " ".join(line.split())
        if self._leading_marker.match(heading):
            return None
        if (
            DottedNumberingStrategy._clause_pattern.match(heading)
            or VerboseNumberingStrategy().match_clause_id(heading) is not None
        ):
            return None
        if (heading.startswith("[") and heading.endswith("]")) or (
            heading.startswith("{") and heading.endswith("}")
        ):
            return None
        if not _looks_like_heading(heading):
            return None
        return heading


_HEADING_MAX_CHARS = 100
_HEADING_MAX_WORDS = 12
# A heading is mostly capitalised; prose that happens to be short is not.
_HEADING_TITLE_CASE_RATIO = 0.6
_HEADING_WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _looks_like_heading(text: str) -> bool:
    """Return whether ``text`` reads as a title rather than as a sentence.

    Shared by the two strategies that need the question answered so they cannot
    disagree: ``HeadingOnlyStrategy`` for unnumbered titles, and
    ``DottedNumberingStrategy`` for the text following a section number.
    """

    heading = " ".join(text.split())
    if not heading or len(heading) > _HEADING_MAX_CHARS:
        return False
    if len(heading.split()) > _HEADING_MAX_WORDS:
        return False
    # A title carries no sentence punctuation; a wrapped sentence fragment does.
    if heading[-1] in ".;,:!?" or ";" in heading:
        return False

    words = _HEADING_WORD.findall(heading)
    if not words:
        return False
    uppercase_initials = sum(word[0].isupper() for word in words)
    return heading.isupper() or uppercase_initials / len(words) >= _HEADING_TITLE_CASE_RATIO


_STRATEGIES: tuple[ChunkStrategy, ...] = (
    DottedNumberingStrategy(),
    VerboseNumberingStrategy(),
    HeadingOnlyStrategy(),
)


def chunk_by_clause(pages: list[PageText]) -> list[Chunk]:
    """Chunk one document using the first strategy that reaches its threshold."""

    source_file = _validate_pages(pages)
    for strategy in _STRATEGIES:
        chunks = strategy.detect(pages)
        if chunks is not None:
            return chunks
    raise UnsupportedNumberingError(source_file)


def _validate_pages(pages: list[PageText]) -> str | None:
    if not pages:
        return None

    source_files = {page.source_file for page in pages}
    if len(source_files) > 1:
        raise ValueError("all pages must belong to the same source_file")

    page_numbers = [page.page_number for page in pages]
    # Pageless formats (DOCX, TXT) produce exactly one page, so there is no
    # ordering to check - and comparing None against None would raise.
    if any(page_number is None for page_number in page_numbers):
        if len(page_numbers) > 1:
            raise ValueError("pages without a page_number must be a single page")
    elif any(
        current >= following
        for current, following in zip(page_numbers, page_numbers[1:], strict=False)
    ):
        raise ValueError("pages must be ordered by strictly increasing page_number")

    return pages[0].source_file


def _flatten_pages(pages: list[PageText]) -> list[_LineRecord]:
    return [
        _LineRecord(text=line, page_number=page.page_number)
        for page in pages
        for line in page.text.splitlines()
    ]


def _build_chunks(
    line_records: list[_LineRecord],
    boundaries: list[_Boundary],
    source_file: str | None,
    strategy_name: str,
) -> list[Chunk]:
    if source_file is None:
        return []

    chunks: list[Chunk] = []
    first_boundary = boundaries[0]
    preamble_records = line_records[: first_boundary.line_index]
    preamble_text = _join_records(preamble_records)
    if preamble_text:
        chunks.append(
            Chunk(
                text=preamble_text,
                clause_id=None,
                page_number=_first_content_page(preamble_records),
                source_file=source_file,
                detected_strategy=strategy_name,
            )
        )

    for boundary_index, boundary in enumerate(boundaries):
        next_line_index = (
            boundaries[boundary_index + 1].line_index
            if boundary_index + 1 < len(boundaries)
            else len(line_records)
        )
        chunk_text = _join_records(line_records[boundary.line_index : next_line_index])
        chunks.append(
            Chunk(
                text=chunk_text,
                clause_id=boundary.clause_id,
                page_number=boundary.page_number,
                source_file=source_file,
                detected_strategy=strategy_name,
            )
        )

    return chunks


def _join_records(records: list[_LineRecord]) -> str:
    return "\n".join(record.text for record in records).strip()


def _first_content_page(records: list[_LineRecord]) -> int | None:
    for record in records:
        if record.text.strip():
            return record.page_number
    raise ValueError("chunk content must contain at least one non-empty line")
