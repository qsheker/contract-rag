from pathlib import Path
from typing import Any

import pytest

import contract_rag.embeddings.embeddings as embeddings_module
from contract_rag.chunker import UnsupportedNumberingError
from contract_rag.embeddings import EMBEDDING_DIMENSION, MODEL_TOKEN_LIMIT, RoSBERTaEmbedder
from contract_rag.ingestion import (
    PAGELESS_WARNING,
    ingest_document,
    normalize_filename,
    pageless_warnings,
)
from contract_rag.loader import LoaderError, NoTextLayerError, UnsupportedFormatError

CORPUS_DIR = Path(__file__).parents[1] / "corpus_raw" / "contracts"
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeModel:
    def __init__(self) -> None:
        self.max_seq_length = MODEL_TOKEN_LIMIT
        self.encoded_texts: list[str] = []
        self.tokenizer = self._tokenize

    def get_embedding_dimension(self) -> int:
        return EMBEDDING_DIMENSION

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.encoded_texts.extend(sentences)
        return [[float(index)] * EMBEDDING_DIMENSION for index, _ in enumerate(sentences)]

    @staticmethod
    def _tokenize(text: str, **kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split()) + 2))}


class _NotFilter:
    """The `not_.in_(...)` half of the PostgREST builder the fake understands."""

    def __init__(self, client: "FakeSupabaseClient") -> None:
        self._client = client

    def in_(self, column: str, values: list[str]) -> "FakeSupabaseClient":
        assert column == "id"
        self._client.pending_keep_ids = set(values)
        return self._client


class FakeSupabaseClient:
    """An in-memory contract_chunks table that records upserts and deletes."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[tuple[str, list[dict[str, Any]], str | None]] = []
        self.delete_calls: list[tuple[str, set[str] | None]] = []
        self.pending_keep_ids: set[str] | None = None
        self._table_name = ""
        self._operation = ""
        self._pending_rows: list[dict[str, Any]] = []
        self._on_conflict: str | None = None
        self._pending_source_file: str | None = None

    def table(self, table_name: str) -> "FakeSupabaseClient":
        self._table_name = table_name
        return self

    def upsert(
        self,
        rows: list[dict[str, Any]],
        *,
        on_conflict: str | None = None,
    ) -> "FakeSupabaseClient":
        self._operation = "upsert"
        self._pending_rows = rows
        self._on_conflict = on_conflict
        return self

    def delete(self) -> "FakeSupabaseClient":
        self._operation = "delete"
        self._pending_source_file = None
        self.pending_keep_ids = None
        return self

    def eq(self, column: str, value: str) -> "FakeSupabaseClient":
        assert column == "source_file"
        self._pending_source_file = value
        return self

    @property
    def not_(self) -> _NotFilter:
        return _NotFilter(self)

    def execute(self) -> None:
        if self._operation == "upsert":
            copied_rows = [dict(row) for row in self._pending_rows]
            self.upsert_calls.append((self._table_name, copied_rows, self._on_conflict))
            for row in copied_rows:
                self.rows[row["id"]] = row
            return

        assert self._operation == "delete"
        source_file = self._pending_source_file
        keep_ids = self.pending_keep_ids
        assert source_file is not None
        self.delete_calls.append((source_file, keep_ids))
        self.rows = {
            row_id: row
            for row_id, row in self.rows.items()
            if row["source_file"] != source_file or (keep_ids is not None and row_id in keep_ids)
        }


@pytest.fixture
def fake_client() -> FakeSupabaseClient:
    return FakeSupabaseClient()


@pytest.fixture(autouse=True)
def reset_default_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings_module, "_default_embedder", RoSBERTaEmbedder(FakeModel()))


def read_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


def test_pdf_upload_is_indexed_under_its_own_file_name(fake_client: FakeSupabaseClient) -> None:
    file_bytes = (CORPUS_DIR / "contract_01.pdf").read_bytes()

    chunks_indexed = ingest_document(file_bytes, "contract_01.pdf", supabase_client=fake_client)

    assert chunks_indexed == len(fake_client.rows) > 0
    # The temporary file the loaders read must leave no trace: retrieval and
    # citations both key off source_file.
    assert {row["source_file"] for row in fake_client.rows.values()} == {"contract_01.pdf"}
    assert all(row["id"].startswith("contract_01.pdf::") for row in fake_client.rows.values())
    assert any(row["page_number"] is not None for row in fake_client.rows.values())


@pytest.mark.parametrize("filename", ["sample_contract.docx", "sample_contract.txt"])
def test_pageless_upload_is_indexed_without_page_numbers(
    fake_client: FakeSupabaseClient,
    filename: str,
) -> None:
    chunks_indexed = ingest_document(read_fixture(filename), filename, supabase_client=fake_client)

    assert chunks_indexed > 0
    assert all(row["page_number"] is None for row in fake_client.rows.values())
    assert pageless_warnings(filename) == [PAGELESS_WARNING.format(filename=filename)]


def test_pdf_upload_carries_no_pageless_warning() -> None:
    assert pageless_warnings("contract_01.pdf") == []


def test_unsupported_format_is_rejected_before_anything_is_written(
    fake_client: FakeSupabaseClient,
) -> None:
    with pytest.raises(UnsupportedFormatError) as error:
        ingest_document(b"anything", "contract.rtf", supabase_client=fake_client)

    assert error.value.extension == ".rtf"
    assert error.value.source_file == "contract.rtf"
    assert fake_client.upsert_calls == []


def test_unreadable_file_leaves_the_index_untouched(fake_client: FakeSupabaseClient) -> None:
    with pytest.raises(LoaderError):
        ingest_document(b"not a pdf at all", "broken.pdf", supabase_client=fake_client)

    assert fake_client.rows == {}


def test_empty_document_is_rejected(fake_client: FakeSupabaseClient) -> None:
    with pytest.raises(NoTextLayerError):
        ingest_document(b"   \n  ", "blank.txt", supabase_client=fake_client)

    assert fake_client.rows == {}


def test_unsupported_numbering_indexes_nothing(fake_client: FakeSupabaseClient) -> None:
    # No clause boundaries at all, so no strategy reaches its threshold.
    with pytest.raises(UnsupportedNumberingError):
        ingest_document(
            "одна строка без нумерации".encode(), "flat.txt", supabase_client=fake_client
        )

    assert fake_client.upsert_calls == []
    assert fake_client.rows == {}


def test_reupload_replaces_the_previous_version_without_duplicating_it(
    fake_client: FakeSupabaseClient,
) -> None:
    first = (
        "ДОГОВОР\n1.1 Первый пункт.\n1.2 Второй пункт.\n1.3 Третий пункт, который потом удалят.\n"
    ).encode()
    edited = "ДОГОВОР\n1.1 Первый пункт.\n1.2 Второй пункт изменён.\n1.4 Новый пункт.\n".encode()

    ingest_document(first, "contract.txt", supabase_client=fake_client)
    assert set(fake_client.rows) == {
        "contract.txt::preamble",
        "contract.txt::1.1",
        "contract.txt::1.2",
        "contract.txt::1.3",
    }

    chunks_indexed = ingest_document(edited, "contract.txt", supabase_client=fake_client)

    assert chunks_indexed == 4
    # 1.3 is gone from the file, so it must be gone from the index: a leftover
    # chunk is still retrievable and would be cited as if it were in force.
    assert set(fake_client.rows) == {
        "contract.txt::preamble",
        "contract.txt::1.1",
        "contract.txt::1.2",
        "contract.txt::1.4",
    }
    assert "изменён" in fake_client.rows["contract.txt::1.2"]["text"]


def test_stale_chunks_are_deleted_only_after_the_upsert(
    fake_client: FakeSupabaseClient,
) -> None:
    ingest_document(read_fixture("sample_contract.txt"), "s.txt", supabase_client=fake_client)

    source_file, keep_ids = fake_client.delete_calls[-1]
    assert source_file == "s.txt"
    assert keep_ids == set(fake_client.rows)
    assert len(fake_client.upsert_calls) == 1


def test_other_documents_survive_a_reupload(fake_client: FakeSupabaseClient) -> None:
    ingest_document(read_fixture("sample_contract.txt"), "first.txt", supabase_client=fake_client)
    untouched = set(fake_client.rows)

    ingest_document(
        read_fixture("sample_contract.docx"), "second.docx", supabase_client=fake_client
    )

    assert untouched <= set(fake_client.rows)


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("contract.pdf", "contract.pdf"),
        ("  contract.pdf  ", "contract.pdf"),
        ("/Users/me/Documents/contract.pdf", "contract.pdf"),
        ("C:\\Users\\me\\contract.pdf", "contract.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
    ],
)
def test_normalize_filename_keeps_only_the_basename(supplied: str, expected: str) -> None:
    assert normalize_filename(supplied) == expected


@pytest.mark.parametrize("supplied", ["", "   ", "/", ".", "..", "some/directory/"])
def test_normalize_filename_rejects_names_it_cannot_use(supplied: str) -> None:
    with pytest.raises(ValueError, match="no usable file name"):
        normalize_filename(supplied)
