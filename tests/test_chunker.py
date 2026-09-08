from pathlib import Path

import pytest

from contract_rag.chunker import (
    Chunk,
    DottedNumberingStrategy,
    HeadingOnlyStrategy,
    UnsupportedNumberingError,
    VerboseNumberingStrategy,
    chunk_by_clause,
)
from contract_rag.loader import PageText, load_pdf

CORPUS_DIR = Path(__file__).parents[1] / "corpus_raw" / "contracts"
EXPECTED_CORPUS_STRATEGIES = {
    "contract_01.pdf": "dotted_numbering",
    "contract_02.pdf": "verbose_numbering",
    "contract_03.pdf": "heading_only",
    "contract_04.pdf": None,
    "contract_05.pdf": "heading_only",
}


def test_dotted_numbering_creates_one_chunk_per_clause_with_preamble() -> None:
    pages = [
        PageText(
            page_number=1,
            text=(
                "Agreement preamble\n1.1 First clause\nFirst body\n1.2 Second clause\nPage one body"
            ),
            source_file="sample.pdf",
        ),
        PageText(
            page_number=2,
            text="Second clause continues\n1.3 Third clause\nThird body",
            source_file="sample.pdf",
        ),
    ]

    chunks = chunk_by_clause(pages)

    assert [chunk.clause_id for chunk in chunks] == [None, "1.1", "1.2", "1.3"]
    assert chunks[0].text == "Agreement preamble"
    assert chunks[0].page_number == 1
    assert chunks[2].text == "1.2 Second clause\nPage one body\nSecond clause continues"
    assert chunks[2].page_number == 1
    assert all(chunk.source_file == "sample.pdf" for chunk in chunks)
    assert all(chunk.detected_strategy == "dotted_numbering" for chunk in chunks)


def test_strategies_require_three_matches() -> None:
    pages = [
        PageText(1, "1.1 First\nbody.\n1.2 Second\nmore body.", "sample.pdf"),
    ]

    assert DottedNumberingStrategy().detect(pages) is None
    with pytest.raises(UnsupportedNumberingError) as exception_info:
        chunk_by_clause(pages)
    assert exception_info.value.source_file == "sample.pdf"


def test_dotted_numbering_supports_three_levels_without_partial_matches() -> None:
    pages = [
        PageText(
            1,
            "1.1 First\nBody.\n1.1.1 Nested\nBody.\n2.3.4 Another nested clause\nBody.",
            "sample.pdf",
        )
    ]

    chunks = DottedNumberingStrategy().detect(pages)

    assert chunks is not None
    assert [chunk.clause_id for chunk in chunks] == ["1.1", "1.1.1", "2.3.4"]


def test_dotted_numbering_rejects_line_wrapped_references() -> None:
    pages = [
        PageText(
            1,
            (
                "6.2.6 shall be applied as stated.\n"
                "6.2.6 of this section remains effective.\n"
                "2.16.1 or 2.16.3 applies.\n"
                "2.13."
            ),
            "sample.pdf",
        )
    ]

    assert DottedNumberingStrategy().detect(pages) is None


def test_dotted_numbering_has_priority_over_other_strategies() -> None:
    pages = [
        PageText(
            1,
            ("ARTICLE I\n1.1 First\nARTICLE II\n1.2 Second\nARTICLE III\n1.3 Third"),
            "sample.pdf",
        )
    ]

    chunks = chunk_by_clause(pages)

    assert all(chunk.detected_strategy == "dotted_numbering" for chunk in chunks)


def test_verbose_numbering_supports_articles_sections_and_combined_labels() -> None:
    pages = [
        PageText(
            1,
            (
                "Preamble\nArticle IV, Section 2 General Terms\nBody one.\n"
                "SECTION 2.1 Duties\nBody two.\nARTICLE V Closing\nBody three."
            ),
            "sample.pdf",
        )
    ]

    chunks = VerboseNumberingStrategy().detect(pages)

    assert chunks is not None
    assert [chunk.clause_id for chunk in chunks] == [
        None,
        "Article IV, Section 2",
        "Section 2.1",
        "Article V",
    ]
    assert all(chunk.detected_strategy == "verbose_numbering" for chunk in chunks)


def test_heading_only_uses_heading_text_as_clause_id() -> None:
    pages = [
        PageText(
            1,
            (
                "Cover text.\nIntroduction\nThe parties agree.\n"
                "Payment Terms\nPayment is monthly.\nLegal Matters\nThe law applies."
            ),
            "sample.pdf",
        )
    ]

    chunks = HeadingOnlyStrategy().detect(pages)

    assert chunks is not None
    assert [chunk.clause_id for chunk in chunks] == [
        None,
        "Introduction",
        "Payment Terms",
        "Legal Matters",
    ]
    assert all(chunk.detected_strategy == "heading_only" for chunk in chunks)


def test_chunk_and_input_validation() -> None:
    chunk = Chunk("text", "1.1", 1, "sample.pdf", "dotted_numbering")
    assert chunk.page_number == 1

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        Chunk("text", "1.1", 0, "sample.pdf", "dotted_numbering")
    with pytest.raises(ValueError, match="same source_file"):
        chunk_by_clause([PageText(1, "1.1", "a.pdf"), PageText(2, "1.2", "b.pdf")])
    with pytest.raises(ValueError, match="strictly increasing"):
        chunk_by_clause([PageText(2, "1.1", "a.pdf"), PageText(1, "1.2", "a.pdf")])


@pytest.mark.parametrize(
    ("filename", "expected_strategy"),
    EXPECTED_CORPUS_STRATEGIES.items(),
)
def test_corpus_strategy_outcome(filename: str, expected_strategy: str | None) -> None:
    pages = load_pdf(CORPUS_DIR / filename)

    if expected_strategy is None:
        with pytest.raises(UnsupportedNumberingError):
            chunk_by_clause(pages)
        return

    chunks = chunk_by_clause(pages)
    assert chunks
    assert all(chunk.detected_strategy == expected_strategy for chunk in chunks)
    assert all(chunk.source_file == str(CORPUS_DIR / filename) for chunk in chunks)
    assert all(chunk.page_number >= 1 for chunk in chunks)
