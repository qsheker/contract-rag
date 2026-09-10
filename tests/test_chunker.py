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
from contract_rag.loader import PageText, load_document, load_pdf

CORPUS_DIR = Path(__file__).parents[1] / "corpus_raw" / "contracts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
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


def test_dotted_numbering_accepts_sub_items_that_continue_in_lower_case() -> None:
    # Russian sub-items continue the parent clause's sentence, so they open in
    # lower case. Demanding a capital dropped every one of them.
    pages = [
        PageText(
            1,
            (
                "3.1. Удержания производятся в следующих случаях:\n"
                "3.1.1. для возмещения неотработанного аванса;\n"
                "3.1.2. в случаях возмещения затрат на обучение;\n"
                "3.1.3. в иных случаях при наличии согласия."
            ),
            "sample.docx",
        )
    ]

    chunks = DottedNumberingStrategy().detect(pages)

    assert chunks is not None
    assert [chunk.clause_id for chunk in chunks] == ["3.1", "3.1.1", "3.1.2", "3.1.3"]


def test_a_terminated_number_opens_a_clause_and_a_bare_one_does_not() -> None:
    # The whole distinction in one place: an enumerator terminates its own
    # number, prose quoting a clause number does not.
    strategy = DottedNumberingStrategy()

    assert strategy.match_clause_id("4.9.3. для возмещения аванса;") == "4.9.3"
    assert strategy.match_clause_id("6.2.6 shall be applied as stated.") is None


def test_section_headings_open_a_chunk_of_their_own() -> None:
    pages = [
        PageText(
            1,
            (
                "5. ПРАВА И ОБЯЗАННОСТИ РАБОТНИКА\n"
                "5.1. Работник обязан соблюдать стандарты.\n"
                "6. ПРАВА И ОБЯЗАННОСТИ РАБОТОДАТЕЛЯ\n"
                "6.1. Работодатель обязан выплачивать вознаграждение."
            ),
            "sample.docx",
        )
    ]

    chunks = chunk_by_clause(pages)

    assert [chunk.clause_id for chunk in chunks] == ["5", "5.1", "6", "6.1"]
    # Without the heading boundary, section 6 would sit inside clause 5.1 and
    # every fact in it would be cited as 5.1.
    clause_5_1 = next(chunk for chunk in chunks if chunk.clause_id == "5.1")
    assert "РАБОТОДАТЕЛЯ" not in clause_5_1.text


def test_a_numbered_line_of_prose_is_not_a_section_heading() -> None:
    strategy = DottedNumberingStrategy()

    assert strategy.match_clause_id("5. ПРАВА И ОБЯЗАННОСТИ РАБОТНИКА") == "5"
    # Reads as a sentence, not a title: ends in a full stop and is mostly
    # lower case.
    assert strategy.match_clause_id("5. работник обязан оплатить услуги в срок.") is None


def test_realistic_numbering_is_split_without_losing_text() -> None:
    """Cover the shapes of a real employment contract in one document.

    Three-level sub-items opening in lower case, single-level section headings
    and an annex that restarts numbering - the combination that produced 106
    boundaries out of 231 before this fix.
    """

    pages = load_document(FIXTURES_DIR / "clause_numbering.txt")
    chunks = chunk_by_clause(pages)

    clause_ids = [chunk.clause_id for chunk in chunks]
    assert clause_ids.count("2.1.2") == 1
    assert "3" in clause_ids and "4" in clause_ids
    # The annex restarts at 1, so those identifiers occur twice.
    assert clause_ids.count("1.1") == 2

    def significant(text: str) -> str:
        return "".join(text.split())

    # Nothing may fall between the loader and the chunks.
    assert significant("".join(chunk.text for chunk in chunks)) == significant(pages[0].text)


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


def test_pageless_document_yields_chunks_without_a_page_number() -> None:
    pages = [
        PageText(
            page_number=None,
            text=(
                "Agreement preamble\n1.1 First clause\nBody\n"
                "1.2 Second clause\nMore body\n1.3 Third clause\nTail"
            ),
            source_file="agreement.docx",
        )
    ]

    chunks = chunk_by_clause(pages)

    assert [chunk.clause_id for chunk in chunks] == [None, "1.1", "1.2", "1.3"]
    assert all(chunk.page_number is None for chunk in chunks)
    assert all(chunk.source_file == "agreement.docx" for chunk in chunks)


def test_several_pageless_pages_are_rejected() -> None:
    pages = [
        PageText(page_number=None, text="1.1 First\n1.2 Second", source_file="agreement.docx"),
        PageText(page_number=None, text="1.3 Third\n1.4 Fourth", source_file="agreement.docx"),
    ]

    with pytest.raises(ValueError, match="single page"):
        chunk_by_clause(pages)
