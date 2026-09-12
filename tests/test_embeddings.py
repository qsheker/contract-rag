import logging
from pathlib import Path
from typing import Any

import pytest

import contract_rag.embeddings.embeddings as embeddings_module
from contract_rag.chunker import Chunk, UnsupportedNumberingError, chunk_by_clause
from contract_rag.embeddings import (
    CONTEXT_MAX_CHARS,
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_TOKEN_LIMIT,
    RoSBERTaEmbedder,
    create_supabase_client_from_env,
    embed_and_index,
)
from contract_rag.loader import load_pdf

CORPUS_DIR = Path(__file__).parents[1] / "corpus_raw" / "contracts"
MIGRATIONS_DIR = Path(__file__).parents[1] / "supabase" / "migrations"


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


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[tuple[str, list[dict[str, Any]], str | None]] = []
        self._table_name = ""
        self._pending_rows: list[dict[str, Any]] = []
        self._on_conflict: str | None = None

    def table(self, table_name: str) -> "FakeSupabaseClient":
        self._table_name = table_name
        return self

    def upsert(
        self,
        rows: list[dict[str, Any]],
        *,
        on_conflict: str | None = None,
    ) -> "FakeSupabaseClient":
        self._pending_rows = rows
        self._on_conflict = on_conflict
        return self

    def execute(self) -> None:
        copied_rows = [dict(row) for row in self._pending_rows]
        self.upsert_calls.append((self._table_name, copied_rows, self._on_conflict))
        for row in copied_rows:
            self.rows[row["id"]] = row


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def fake_client() -> FakeSupabaseClient:
    return FakeSupabaseClient()


@pytest.fixture(autouse=True)
def reset_default_embedder(monkeypatch: pytest.MonkeyPatch, fake_model: FakeModel) -> None:
    embedder = RoSBERTaEmbedder(fake_model)
    monkeypatch.setattr(embeddings_module, "_default_embedder", embedder)


def make_chunk(
    clause_id: str | None,
    *,
    text: str = "Clause text",
    page_number: int | None = 1,
    source_file: str = "contract.pdf",
) -> Chunk:
    return Chunk(
        text=text,
        clause_id=clause_id,
        page_number=page_number,
        source_file=source_file,
        detected_strategy="dotted_numbering",
    )


def test_embed_and_index_writes_one_row_per_unique_chunk(
    fake_client: FakeSupabaseClient,
) -> None:
    preamble = make_chunk(None, text="Preamble")
    clause = make_chunk("1.1")

    embed_and_index([preamble, clause, clause], fake_client)

    assert len(fake_client.rows) == 2
    assert set(fake_client.rows) == {
        "contract.pdf::preamble",
        "contract.pdf::1.1",
    }
    assert fake_client.upsert_calls[0][0] == "contract_chunks"
    assert fake_client.upsert_calls[0][2] == "id"


def test_the_encoded_text_carries_the_context_but_the_stored_text_does_not(
    fake_client: FakeSupabaseClient,
    fake_model: FakeModel,
) -> None:
    chunk = Chunk(
        text="3.1.1. пятидневная рабочая неделя;",
        clause_id="3.1.1",
        page_number=None,
        source_file="contract.docx",
        detected_strategy="dotted_numbering",
        context="3. РЕЖИМ РАБОЧЕГО ВРЕМЕНИ\n3.1. Работнику устанавливается рабочее время:",
    )

    embed_and_index([chunk], fake_client)

    encoded = fake_model.encoded_texts[0]
    assert encoded == f"{DOCUMENT_PREFIX}{chunk.context}\n{chunk.text}"
    # A citation quotes the contract, so the stored text stays verbatim.
    assert fake_client.rows["contract.docx::3.1.1"]["text"] == chunk.text


def test_a_long_chunk_is_encoded_without_its_context(
    fake_client: FakeSupabaseClient,
    fake_model: FakeModel,
) -> None:
    # A chunk this size carries its own framing. Prepending headings anyway
    # broadens it topically and pushes other documents' answers out of top-k.
    long_text = "5.5. " + "Условие договора. " * 40
    assert len(long_text) >= CONTEXT_MAX_CHARS
    chunk = Chunk(
        text=long_text,
        clause_id="5.5",
        page_number=None,
        source_file="contract.docx",
        detected_strategy="dotted_numbering",
        context="5. ОТВЕТСТВЕННОСТЬ СТОРОН",
    )

    embed_and_index([chunk], fake_client)

    assert fake_model.encoded_texts == [f"{DOCUMENT_PREFIX}{long_text}"]


def test_a_chunk_without_context_is_encoded_as_before(
    fake_client: FakeSupabaseClient,
    fake_model: FakeModel,
) -> None:
    chunk = make_chunk("1.1", text="1.1 Clause text")

    embed_and_index([chunk], fake_client)

    assert fake_model.encoded_texts == [f"{DOCUMENT_PREFIX}1.1 Clause text"]


def test_preamble_clause_id_is_native_none(fake_client: FakeSupabaseClient) -> None:
    embed_and_index([make_chunk(None, text="Preamble")], fake_client)

    assert fake_client.rows["contract.pdf::preamble"]["clause_id"] is None


def test_repeated_indexing_upserts_without_new_rows(fake_client: FakeSupabaseClient) -> None:
    chunks = [make_chunk(None, text="Preamble"), make_chunk("1.1")]

    embed_and_index(chunks, fake_client)
    first_ids = set(fake_client.rows)
    embed_and_index(chunks, fake_client)

    assert set(fake_client.rows) == first_ids
    assert len(fake_client.rows) == 2
    assert len(fake_client.upsert_calls) == 2


def test_embedder_adds_search_document_prefix(fake_model: FakeModel) -> None:
    embedder = RoSBERTaEmbedder(fake_model)

    embedder.embed_documents([make_chunk("1.1", text="Clean input")])

    assert fake_model.encoded_texts == [f"{DOCUMENT_PREFIX}Clean input"]


def test_long_input_warns_and_model_limit_is_enforced(
    caplog: pytest.LogCaptureFixture,
    fake_model: FakeModel,
) -> None:
    fake_model.max_seq_length = MODEL_TOKEN_LIMIT + 100
    embedder = RoSBERTaEmbedder(fake_model)
    long_text = "token " * MODEL_TOKEN_LIMIT

    with caplog.at_level(logging.WARNING, logger=embeddings_module.__name__):
        embedder.embed_documents(
            [make_chunk("4.2", text=long_text, source_file="long-contract.pdf")]
        )

    assert fake_model.max_seq_length == MODEL_TOKEN_LIMIT
    assert "source_file=long-contract.pdf" in caplog.text
    assert "clause_id=4.2" in caplog.text
    assert fake_model.encoded_texts == [f"{DOCUMENT_PREFIX}{long_text}"]


def test_metadata_and_embedding_dimension_are_preserved(
    fake_client: FakeSupabaseClient,
) -> None:
    chunk = make_chunk(
        "2.3",
        text="Payment terms",
        page_number=7,
        source_file="source.pdf",
    )

    embed_and_index([chunk], fake_client)

    row = fake_client.rows["source.pdf::2.3"]
    assert row["text"] == "Payment terms"
    assert len(row["embedding"]) == EMBEDDING_DIMENSION
    assert row["page_number"] == 7
    assert row["source_file"] == "source.pdf"
    assert row["detected_strategy"] == "dotted_numbering"


def test_colliding_clause_ids_keep_every_distinct_chunk(
    fake_client: FakeSupabaseClient,
) -> None:
    kazakh = make_chunk("1.1", text="Қазақша мәтін", page_number=1)
    russian = make_chunk("1.1", text="Русский текст", page_number=2)

    embed_and_index([kazakh, russian], fake_client)

    assert len(fake_client.rows) == 2
    assert "contract.pdf::1.1" in fake_client.rows
    assert "contract.pdf::1.1::page-2::occurrence-2" in fake_client.rows
    assert {row["text"] for row in fake_client.rows.values()} == {
        "Қазақша мәтін",
        "Русский текст",
    }


def test_empty_input_does_not_load_model_or_call_supabase(
    fake_client: FakeSupabaseClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        embeddings_module,
        "get_default_embedder",
        lambda: pytest.fail("model must not load for empty input"),
    )

    embed_and_index([], fake_client)

    assert fake_client.upsert_calls == []


def test_missing_supabase_environment_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embeddings_module, "load_dotenv", lambda: None)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SUPABASE_URL, SUPABASE_KEY"):
        create_supabase_client_from_env()


def test_migration_defines_required_pgvector_schema() -> None:
    (migration_path,) = MIGRATIONS_DIR.glob("*_create_contract_chunks.sql")
    migration = migration_path.read_text().lower()

    assert "create extension if not exists vector" in migration
    assert "id text primary key" in migration
    assert "embedding extensions.vector(1024) not null" in migration
    assert "clause_id text null" in migration
    assert "page_number integer not null" in migration
    assert "source_file text not null" in migration
    assert "detected_strategy text not null" in migration


def test_every_supported_corpus_chunk_can_be_indexed_without_data_loss(
    fake_client: FakeSupabaseClient,
) -> None:
    chunks: list[Chunk] = []
    unsupported_files: list[str] = []
    for pdf_path in sorted(CORPUS_DIR.glob("contract_*.pdf")):
        try:
            chunks.extend(chunk_by_clause(load_pdf(pdf_path)))
        except UnsupportedNumberingError:
            unsupported_files.append(pdf_path.name)

    embed_and_index(chunks, fake_client)

    assert len(chunks) == 211
    assert len(fake_client.rows) == len(set(chunks))
    assert unsupported_files == ["contract_04.pdf"]
    assert {row["source_file"] for row in fake_client.rows.values()} == {
        str(CORPUS_DIR / filename)
        for filename in (
            "contract_01.pdf",
            "contract_02.pdf",
            "contract_03.pdf",
            "contract_05.pdf",
        )
    }


def test_chunk_without_a_page_number_is_stored_as_null(fake_client: FakeSupabaseClient) -> None:
    chunk = make_chunk("1.1", page_number=None, source_file="agreement.docx")

    embed_and_index([chunk], fake_client)

    assert fake_client.rows["agreement.docx::1.1"]["page_number"] is None


def test_colliding_clauses_without_page_numbers_stay_distinct(
    fake_client: FakeSupabaseClient,
) -> None:
    first = make_chunk("1.1", text="Русский текст", page_number=None, source_file="a.docx")
    second = make_chunk("1.1", text="Қазақша мәтін", page_number=None, source_file="a.docx")

    embed_and_index([first, second], fake_client)

    assert set(fake_client.rows) == {
        "a.docx::1.1",
        "a.docx::1.1::page-None::occurrence-2",
    }


def test_migration_allows_chunks_without_a_page_number() -> None:
    (migration_path,) = MIGRATIONS_DIR.glob("*_allow_null_page_number.sql")
    migration = migration_path.read_text().lower()

    assert "alter table public.contract_chunks" in migration
    assert "alter column page_number drop not null" in migration

