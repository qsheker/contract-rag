"""Generic clause chunking with ordered detection strategies."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from contract_rag.loader import PageText


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable document fragment anchored to its starting page."""

    text: str
    clause_id: str | None
    page_number: int
    source_file: str
    detected_strategy: str

    def __post_init__(self) -> None:
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
    page_number: int


@dataclass(frozen=True, slots=True)
class _Boundary:
    line_index: int
    clause_id: str
    page_number: int


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
    """Detect dotted clause identifiers such as ``4.2`` and ``4.2.1``."""

    name = "dotted_numbering"
    _pattern = re.compile(
        r"^\s*(?P<clause_id>\d+\.\d+(?:\.\d+)?)(?:\.(?!\d))?(?=\s|$)"
    )

    def match_clause_id(self, line: str) -> str | None:
        match = self._pattern.match(line)
        if not match:
            return None

        remainder = line[match.end() :].strip()
        first_following_letter = next(
            (character for character in remainder if character.isalpha()),
            None,
        )
        if first_following_letter is not None and first_following_letter.islower():
            return None
        return match.group("clause_id")


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
        if not heading or len(heading) > 100:
            return None
        if len(heading.split()) > 12:
            return None
        if self._leading_marker.match(heading):
            return None
        if heading[-1] in ".;,:!?" or ";" in heading:
            return None
        if (
            DottedNumberingStrategy._pattern.match(heading)
            or VerboseNumberingStrategy().match_clause_id(heading) is not None
        ):
            return None
        if (heading.startswith("[") and heading.endswith("]")) or (
            heading.startswith("{") and heading.endswith("}")
        ):
            return None

        words = self._word.findall(heading)
        if not words:
            return None
        uppercase_initials = sum(word[0].isupper() for word in words)
        is_title_like = uppercase_initials / len(words) >= 0.6
        if not heading.isupper() and not is_title_like:
            return None
        return heading


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
    if any(
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


def _first_content_page(records: list[_LineRecord]) -> int:
    for record in records:
        if record.text.strip():
            return record.page_number
    raise ValueError("chunk content must contain at least one non-empty line")
