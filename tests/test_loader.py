from dataclasses import FrozenInstanceError
from pathlib import Path

import pymupdf
import pytest

from contract_rag.loader import (
    LoaderError,
    LoaderErrorCode,
    NoTextLayerError,
    PageText,
    join_hyphenation,
    load_pdf,
    strip_boilerplate,
)

CORPUS_DIR = Path(__file__).parents[1] / "corpus_raw" / "contracts"
CORPUS_FILES = sorted(CORPUS_DIR.glob("contract_*.pdf"))


def _create_pdf(path: Path, page_texts: list[str]) -> None:
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)


def test_page_text_is_immutable_and_requires_positive_page_number() -> None:
    page = PageText(page_number=1, text="body", source_file="sample.pdf")

    with pytest.raises(FrozenInstanceError):
        page.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        PageText(page_number=0, text="body", source_file="sample.pdf")


def test_load_pdf_returns_one_based_page_objects(tmp_path: Path) -> None:
    pdf_path = tmp_path / "three-pages.pdf"
    _create_pdf(pdf_path, ["Alpha", "Beta", "Gamma"])

    pages = load_pdf(pdf_path)

    assert [page.page_number for page in pages] == [1, 2, 3]
    assert [page.text for page in pages] == ["Alpha", "Beta", "Gamma"]
    assert all(page.source_file == str(pdf_path) for page in pages)


def test_strip_boilerplate_normalizes_digits_and_whitespace_at_exact_position() -> None:
    pages = [
        PageText(1, "  Agreement   1\nBody alpha\nPage 1", "sample.pdf"),
        PageText(2, "Agreement 22\nBody beta\nPage   2", "sample.pdf"),
        PageText(3, "Third heading\nShared\nThird footer", "sample.pdf"),
        PageText(4, "Fourth heading\nBody delta\nFourth footer", "sample.pdf"),
    ]

    cleaned = strip_boilerplate(pages)

    assert [page.text for page in cleaned] == [
        "Body alpha",
        "Body beta",
        "Third heading\nShared\nThird footer",
        "Fourth heading\nBody delta\nFourth footer",
    ]
    assert [page.text for page in pages] == [
        "  Agreement   1\nBody alpha\nPage 1",
        "Agreement 22\nBody beta\nPage   2",
        "Third heading\nShared\nThird footer",
        "Fourth heading\nBody delta\nFourth footer",
    ]


def test_strip_boilerplate_requires_two_occurrences_and_one_source() -> None:
    single_page = [PageText(1, "\nHeading\nBody\nFooter\n", "sample.pdf")]

    assert strip_boilerplate(single_page) == [PageText(1, "Heading\nBody\nFooter", "sample.pdf")]
    assert strip_boilerplate([]) == []
    with pytest.raises(ValueError, match="same source_file"):
        strip_boilerplate(
            [
                PageText(1, "First", "first.pdf"),
                PageText(1, "Second", "second.pdf"),
            ]
        )


def test_strip_boilerplate_keeps_matches_from_different_positions() -> None:
    pages = [
        PageText(1, "Shared\nBody alpha\nFooter alpha", "sample.pdf"),
        PageText(2, "Header beta\nShared\nFooter beta", "sample.pdf"),
        PageText(3, "Header gamma\nBody gamma\nFooter gamma", "sample.pdf"),
        PageText(4, "Header delta\nBody delta\nFooter delta", "sample.pdf"),
    ]

    assert strip_boilerplate(pages) == pages


def test_strip_boilerplate_counts_empty_pages_in_threshold() -> None:
    pages = [
        PageText(1, "Marker\nBody alpha", "sample.pdf"),
        PageText(2, "Marker\nBody beta", "sample.pdf"),
        PageText(3, "", "sample.pdf"),
        PageText(4, "Header delta\nBody delta", "sample.pdf"),
        PageText(5, "Header epsilon\nBody epsilon", "sample.pdf"),
    ]

    assert strip_boilerplate(pages) == pages


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "...policies and prac-\ntices have pushed...",
            "...policies and practices have pushed...",
        ),
        ("...Auch andere EU-\nStaaten, wie...", "...Auch andere EU-Staaten, wie..."),
        ("обяза-\n    тельства сторон", "обязательства сторон"),
        ("келіс-\n    ім талаптары", "келісім талаптары"),
        ("multi\u2010\nlingual", "multilingual"),
        ("EU-\r\n\tStaaten", "EU-Staaten"),
        ("end-\n\nnew paragraph", "end-\n\nnew paragraph"),
        ("section-\n2", "section-\n2"),
    ],
)
def test_join_hyphenation(source: str, expected: str) -> None:
    assert join_hyphenation(source) == expected


def test_load_pdf_raises_no_text_layer_error_for_blank_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    _create_pdf(pdf_path, ["", ""])

    with pytest.raises(NoTextLayerError) as exception_info:
        load_pdf(pdf_path)

    assert exception_info.value.code is LoaderErrorCode.NO_TEXT_LAYER
    assert exception_info.value.source_file == str(pdf_path)


def test_load_pdf_reports_expected_file_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.pdf"
    with pytest.raises(LoaderError) as missing_error:
        load_pdf(missing_path)
    assert missing_error.value.code is LoaderErrorCode.FILE_NOT_FOUND

    with pytest.raises(LoaderError) as directory_error:
        load_pdf(tmp_path)
    assert directory_error.value.code is LoaderErrorCode.PATH_IS_DIRECTORY

    invalid_path = tmp_path / "invalid.pdf"
    invalid_path.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(LoaderError) as invalid_error:
        load_pdf(invalid_path)
    assert invalid_error.value.code is LoaderErrorCode.INVALID_PDF


def test_load_pdf_rejects_password_protected_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "protected.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Protected text")
        document.save(
            pdf_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-secret",
            user_pw="user-secret",
        )

    with pytest.raises(LoaderError) as exception_info:
        load_pdf(pdf_path)

    assert exception_info.value.code is LoaderErrorCode.PASSWORD_PROTECTED


@pytest.mark.parametrize("pdf_path", CORPUS_FILES, ids=lambda path: path.name)
def test_load_pdf_handles_every_contract_in_corpus(pdf_path: Path) -> None:
    with pymupdf.open(pdf_path) as document:
        expected_page_count = document.page_count

    pages = load_pdf(pdf_path)

    assert len(pages) == expected_page_count
    assert [page.page_number for page in pages] == list(range(1, expected_page_count + 1))
    assert all(page.source_file == str(pdf_path) for page in pages)
    assert any(page.text for page in pages)
