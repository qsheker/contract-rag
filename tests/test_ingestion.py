from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

import contract_rag.embeddings.embeddings as embeddings_module
from contract_rag.chunker import UnsupportedNumberingError
from contract_rag.embeddings import (
    EMBEDDING_DIMENSION,
    FILTER_BUDGET_BYTES,
    MODEL_TOKEN_LIMIT,
    RoSBERTaEmbedder,
)
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


class FakeResponse:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class FakeSupabaseClient:
    """An in-memory contract_chunks table that records upserts and deletes."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[tuple[str, list[dict[str, Any]], str | None]] = []
        self.delete_calls: list[list[str]] = []
        self._table_name = ""
        self._operation = ""
        self._pending_rows: list[dict[str, Any]] = []
        self._on_conflict: str | None = None
        self._equals: dict[str, str] = {}
        self._in_values: list[str] | None = None

    def table(self, table_name: str) -> "FakeSupabaseClient":
        self._table_name = table_name
        return self

    def select(self, columns: str) -> "FakeSupabaseClient":
        self._operation = "select"
        self._equals = {}
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
        self._equals = {}
        self._in_values = None
        return self

    def eq(self, column: str, value: str) -> "FakeSupabaseClient":
        self._equals[column] = value
        return self

    def in_(self, column: str, values: list[str]) -> "FakeSupabaseClient":
        assert column == "id"
        self._in_values = list(values)
        return self

    def execute(self) -> FakeResponse | None:
        if self._operation == "select":
            return FakeResponse(
                [dict(row) for row in self.rows.values() if self._matches_equals(row)]
            )

        if self._operation == "upsert":
            copied_rows = [dict(row) for row in self._pending_rows]
            self.upsert_calls.append((self._table_name, copied_rows, self._on_conflict))
            for row in copied_rows:
                self.rows[row["id"]] = row
            return None

        assert self._operation == "delete"
        assert self._in_values is not None, "deletes must name the rows they remove"
        self.delete_calls.append(list(self._in_values))
        for row_id in self._in_values:
            self.rows.pop(row_id, None)
        return None

    def _matches_equals(self, row: dict[str, Any]) -> bool:
        return all(row.get(column) == value for column, value in self._equals.items())


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


def test_a_first_upload_issues_no_delete_at_all(fake_client: FakeSupabaseClient) -> None:
    ingest_document(read_fixture("sample_contract.txt"), "s.txt", supabase_client=fake_client)

    # Nothing is stale on a first upload, so no filter reaches the query string.
    assert fake_client.delete_calls == []
    assert len(fake_client.upsert_calls) == 1


def test_only_the_stale_ids_are_named_in_the_delete(fake_client: FakeSupabaseClient) -> None:
    first = "ДОГОВОР\n1.1 Первый.\n1.2 Второй.\n1.3 Третий.\n1.4 Четвёртый.\n".encode()
    edited = "ДОГОВОР\n1.1 Первый.\n1.2 Второй.\n1.3 Третий.\n".encode()

    ingest_document(first, "contract.txt", supabase_client=fake_client)
    ingest_document(edited, "contract.txt", supabase_client=fake_client)

    # Not "everything except the survivors": naming the survivors is what
    # overran the request line for a long non-ASCII file name.
    assert fake_client.delete_calls == [["contract.txt::1.4"]]


def test_a_long_cyrillic_file_name_stays_within_the_filter_budget(
    fake_client: FakeSupabaseClient,
) -> None:
    filename = "Трудовой_договор_2026_Алдияр_Java_Разработчик.txt"
    clauses = "".join(f"1.{index} Пункт номер {index}.\n" for index in range(1, 61))
    ingest_document(f"ДОГОВОР\n{clauses}".encode(), filename, supabase_client=fake_client)

    # Every clause disappears, so the delete has to name 61 long ids.
    ingest_document(
        "ДОГОВОР\n1.1 А.\n1.2 Б.\n1.3 В.\n".encode(), filename, supabase_client=fake_client
    )

    assert len(fake_client.delete_calls) > 1, "such a batch must be split, not sent whole"
    for batch in fake_client.delete_calls:
        encoded_length = sum(len(quote(f'"{value}"')) + 1 for value in batch)
        assert encoded_length <= FILTER_BUDGET_BYTES
    remaining = {row_id for row_id in fake_client.rows}
    assert remaining == {
        f"{filename}::preamble",
        f"{filename}::1.1",
        f"{filename}::1.2",
        f"{filename}::1.3",
    }


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
