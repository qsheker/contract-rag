from pathlib import Path
from typing import Any

import pytest

import contract_rag.retriever.retriever as retriever_module
from contract_rag.chunker import Chunk
from contract_rag.embeddings import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_TOKEN_LIMIT,
    QUERY_PREFIX,
    TABLE_NAME,
    RoSBERTaEmbedder,
)
from contract_rag.retriever import (
    RetrievedChunk,
    SupabaseRetriever,
    expand_with_related_clauses,
    retrieve,
)

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


class FakeResponse:
    def __init__(self, data: list[dict[str, Any]] | None) -> None:
        self.data = data


class FakeSupabaseClient:
    """Serves prepared rows and records how match_documents was called."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else []
        self.rpc_calls: list[tuple[str, dict[str, Any]]] = []
        self._pending_call: tuple[str, dict[str, Any]] | None = None

    def rpc(self, function_name: str, params: dict[str, Any]) -> "FakeSupabaseClient":
        self._pending_call = (function_name, dict(params))
        return self

    def execute(self) -> FakeResponse:
        assert self._pending_call is not None
        function_name, params = self._pending_call
        self.rpc_calls.append((function_name, params))
        match_count = params["match_count"]
        scope = params.get("source_files")
        # Filtered before the ranking, the way the SQL function does it.
        candidates = [
            row for row in self.rows if scope is None or row["source_file"] in scope
        ]
        ranked = sorted(candidates, key=lambda row: row["similarity"], reverse=True)
        return FakeResponse(ranked[:match_count])


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def fake_embedder(fake_model: FakeModel) -> RoSBERTaEmbedder:
    return RoSBERTaEmbedder(fake_model)


def make_row(
    clause_id: str | None,
    similarity: float,
    *,
    text: str = "Clause text",
    page_number: int = 1,
    source_file: str = "contract.pdf",
) -> dict[str, Any]:
    return {
        "id": f"{source_file}::{clause_id or 'preamble'}",
        "text": text,
        "clause_id": clause_id,
        "page_number": page_number,
        "source_file": source_file,
        "detected_strategy": "dotted_numbering",
        "similarity": similarity,
    }


def test_returns_at_most_k_hits_sorted_by_descending_similarity(
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    rows = [make_row(f"1.{index}", similarity) for index, similarity in enumerate([0.3, 0.9, 0.6])]
    rows.extend(make_row(f"2.{index}", 0.1) for index in range(5))
    client = FakeSupabaseClient(rows)

    hits = SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты", k=5)

    assert len(hits) == 5
    assert [hit.similarity for hit in hits] == sorted(
        (hit.similarity for hit in hits), reverse=True
    )
    assert hits[0].chunk.clause_id == "1.1"


def test_calls_match_documents_rpc_with_query_vector_and_k(
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    client = FakeSupabaseClient([make_row("1.1", 0.9)])

    SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты", k=3)

    function_name, params = client.rpc_calls[0]
    assert function_name == "match_documents"
    assert params["match_count"] == 3
    assert len(params["query_embedding"]) == EMBEDDING_DIMENSION


def test_query_is_encoded_with_search_query_prefix(
    fake_model: FakeModel,
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    client = FakeSupabaseClient([make_row("1.1", 0.9)])

    SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты")

    assert fake_model.encoded_texts == [f"{QUERY_PREFIX}срок оплаты"]
    assert not any(text.startswith(DOCUMENT_PREFIX) for text in fake_model.encoded_texts)


@pytest.mark.parametrize("invalid_k", [0, -1, -5])
def test_non_positive_k_is_rejected(
    invalid_k: int,
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    client = FakeSupabaseClient([make_row("1.1", 0.9)])

    with pytest.raises(ValueError, match="k must be"):
        SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты", k=invalid_k)

    assert client.rpc_calls == []


def test_empty_table_returns_empty_list(fake_embedder: RoSBERTaEmbedder) -> None:
    client = FakeSupabaseClient([])

    assert SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты") == []


def test_missing_response_payload_returns_empty_list(fake_embedder: RoSBERTaEmbedder) -> None:
    class NullPayloadClient(FakeSupabaseClient):
        def execute(self) -> FakeResponse:
            super().execute()
            return FakeResponse(None)

    client = NullPayloadClient([])

    assert SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты") == []


def test_hit_carries_the_whole_chunk_with_metadata(fake_embedder: RoSBERTaEmbedder) -> None:
    client = FakeSupabaseClient(
        [
            make_row(
                "2.3",
                0.87,
                text="Оплата производится в течение 10 дней.",
                page_number=7,
                source_file="corpus_raw/contracts/contract_01.pdf",
            )
        ]
    )

    hit = SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты", k=1)[0]

    assert isinstance(hit, RetrievedChunk)
    assert hit.chunk_id == "corpus_raw/contracts/contract_01.pdf::2.3"
    assert hit.chunk.text == "Оплата производится в течение 10 дней."
    assert hit.chunk.clause_id == "2.3"
    assert hit.chunk.page_number == 7
    assert hit.chunk.source_file == "corpus_raw/contracts/contract_01.pdf"
    assert hit.chunk.detected_strategy == "dotted_numbering"
    assert hit.similarity == pytest.approx(0.87)


def test_module_level_retrieve_accepts_an_injected_client(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    monkeypatch.setattr(retriever_module, "get_default_embedder", lambda: fake_embedder)
    client = FakeSupabaseClient([make_row("1.1", 0.9)])

    hits = retrieve("срок оплаты", k=1, supabase_client=client)

    assert [hit.chunk.clause_id for hit in hits] == ["1.1"]


def test_module_level_retrieve_does_not_touch_supabase_for_invalid_k(
    monkeypatch: pytest.MonkeyPatch,
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    monkeypatch.setattr(retriever_module, "get_default_embedder", lambda: fake_embedder)
    client = FakeSupabaseClient([make_row("1.1", 0.9)])

    with pytest.raises(ValueError, match="k must be"):
        retrieve("срок оплаты", k=0, supabase_client=client)

    assert client.rpc_calls == []


def test_migration_defines_cosine_match_documents_without_index() -> None:
    (migration_path,) = MIGRATIONS_DIR.glob("*_create_match_documents.sql")
    migration = migration_path.read_text().lower()

    assert "function public.match_documents" in migration
    assert "query_embedding extensions.vector(1024)" in migration
    assert "match_count int default 5" in migration
    assert "1 - (contract_chunks.embedding <=> query_embedding) as similarity" in migration
    assert "order by contract_chunks.embedding <=> query_embedding" in migration
    for column in ("clause_id", "page_number", "source_file", "detected_strategy"):
        assert column in migration
    assert "create index" not in migration


class FakeTableClient:
    """An in-memory contract_chunks table that answers filtered range reads."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.selects: list[str] = []
        self._equals: dict[str, Any] = {}
        self._range: tuple[int, int] | None = None

    def table(self, table_name: str) -> "FakeTableClient":
        assert table_name == TABLE_NAME
        return self

    def select(self, columns: str) -> "FakeTableClient":
        self.selects.append(columns)
        self._equals = {}
        self._range = None
        return self

    def eq(self, column: str, value: Any) -> "FakeTableClient":
        self._equals[column] = value
        return self

    def range(self, start: int, end: int) -> "FakeTableClient":
        self._range = (start, end)
        return self

    def execute(self) -> FakeResponse:
        matching = [
            row
            for row in self.rows
            if all(row.get(column) == value for column, value in self._equals.items())
        ]
        if self._range is not None:
            start, end = self._range
            matching = matching[start : end + 1]
        return FakeResponse(matching)


def make_hit(clause_id: str, similarity: float, *, text: str = "Clause text") -> RetrievedChunk:
    row = make_row(clause_id, similarity, text=text)
    return RetrievedChunk(
        chunk=Chunk(
            text=row["text"],
            clause_id=row["clause_id"],
            page_number=row["page_number"],
            source_file=row["source_file"],
            detected_strategy=row["detected_strategy"],
        ),
        chunk_id=row["id"],
        similarity=similarity,
    )


def test_a_lead_in_clause_pulls_in_the_sub_items_that_answer_it() -> None:
    # The failure this exists for: "5.2. Работник обязан:" is the best match for
    # "что мне придётся делать на работе", and on its own it answers nothing.
    client = FakeTableClient(
        [
            make_row("5.2", 0.0, text="5.2. Работник обязан:"),
            make_row("5.2.1", 0.0, text="5.2.1. оказать работы согласно трудовым функциям;"),
            make_row("5.2.2", 0.0, text="5.2.2. оформлять документы по результатам работ;"),
            make_row("6.1", 0.0, text="6.1. Работодатель имеет право:"),
        ]
    )

    context = expand_with_related_clauses(
        [make_hit("5.2", 0.63, text="5.2. Работник обязан:")],
        supabase_client=client,
    )

    assert [entry.chunk.clause_id for entry in context] == ["5.2", "5.2.1", "5.2.2"]
    # A sub-item keeps its own identifier, so the citation names 5.2.1 rather
    # than the heading it hangs under.
    assert [entry.was_retrieved for entry in context] == [True, False, False]
    assert context[1].similarity is None


def test_a_sub_item_pulls_in_the_sentence_it_continues() -> None:
    client = FakeTableClient(
        [
            make_row("3", 0.0, text="3. РЕЖИМ РАБОЧЕГО ВРЕМЕНИ"),
            make_row("3.1", 0.0, text="3.1. Работнику устанавливается рабочее время:"),
            make_row("3.1.1", 0.0, text="3.1.1. пятидневная рабочая неделя;"),
        ]
    )

    context = expand_with_related_clauses(
        [make_hit("3.1.1", 0.7, text="3.1.1. пятидневная рабочая неделя;")],
        supabase_client=client,
    )

    # Ancestors after the hit, nearest first: read alone, a sub-item is a
    # fragment of its parent's sentence.
    assert [entry.chunk.clause_id for entry in context] == ["3.1.1", "3.1", "3"]


def test_a_relative_that_does_not_fit_is_skipped_but_a_later_one_still_fits() -> None:
    client = FakeTableClient(
        [
            make_row("1.1", 0.0, text="1.1. lead-in:"),
            make_row("1.1.1", 0.0, text="x" * 400),
            make_row("1.1.2", 0.0, text="short tail"),
        ]
    )

    context = expand_with_related_clauses(
        [make_hit("1.1", 0.9, text="1.1. lead-in:")],
        supabase_client=client,
        budget_chars=100,
    )

    assert [entry.chunk.clause_id for entry in context] == ["1.1", "1.1.2"]


def test_a_retrieved_chunk_is_never_added_twice() -> None:
    client = FakeTableClient(
        [
            make_row("2.1", 0.0, text="2.1. lead-in:"),
            make_row("2.1.1", 0.0, text="2.1.1. first duty;"),
        ]
    )

    context = expand_with_related_clauses(
        [
            make_hit("2.1", 0.8, text="2.1. lead-in:"),
            make_hit("2.1.1", 0.7, text="2.1.1. first duty;"),
        ],
        supabase_client=client,
    )

    assert [entry.chunk_id for entry in context] == [
        "contract.pdf::2.1",
        "contract.pdf::2.1.1",
    ]
    assert all(entry.was_retrieved for entry in context)


def test_sub_items_are_ordered_by_number_not_by_string() -> None:
    client = FakeTableClient(
        [make_row("4.9", 0.0, text="4.9. lead-in:")]
        + [make_row(f"4.9.{index}", 0.0, text=f"item {index}") for index in (10, 2, 1)]
    )

    context = expand_with_related_clauses(
        [make_hit("4.9", 0.8, text="4.9. lead-in:")],
        supabase_client=client,
    )

    assert [entry.chunk.clause_id for entry in context] == ["4.9", "4.9.1", "4.9.2", "4.9.10"]


def test_expansion_without_hits_asks_supabase_nothing() -> None:
    client = FakeTableClient([make_row("1.1", 0.0)])

    assert expand_with_related_clauses([], supabase_client=client) == []
    assert client.selects == []


def test_a_scoped_search_only_returns_the_named_documents(
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    """A chat is about the document uploaded into it.

    Unscoped, a question about a rented flat could come back citing an
    employment contract - a formally valid citation pointing at a document the
    reader never opened here.
    """

    client = FakeSupabaseClient(
        [
            make_row("1.1", 0.9, source_file="employment.docx"),
            make_row("2.2", 0.8, source_file="rent_contract_ru.pdf"),
        ]
    )

    hits = SupabaseRetriever(client, fake_embedder).retrieve(
        "что мне нужно знать про аренду",
        k=5,
        source_files=["rent_contract_ru.pdf"],
    )

    assert [hit.chunk.source_file for hit in hits] == ["rent_contract_ru.pdf"]
    assert client.rpc_calls[0][1]["source_files"] == ["rent_contract_ru.pdf"]


def test_an_unscoped_search_still_covers_every_document(
    fake_embedder: RoSBERTaEmbedder,
) -> None:
    client = FakeSupabaseClient(
        [
            make_row("1.1", 0.9, source_file="employment.docx"),
            make_row("2.2", 0.8, source_file="rent_contract_ru.pdf"),
        ]
    )

    hits = SupabaseRetriever(client, fake_embedder).retrieve("срок оплаты", k=5)

    assert len(hits) == 2
    # None rather than an empty list: the SQL reads it as "no filter at all".
    assert client.rpc_calls[0][1]["source_files"] is None


def test_migration_scopes_match_documents_by_source_file() -> None:
    (migration_path,) = MIGRATIONS_DIR.glob("*_match_documents_by_source.sql")
    migration = migration_path.read_text().lower()

    # Recreated, not overloaded: a defaulted parameter added to the existing
    # function would leave PostgREST two candidates for one named call.
    assert "drop function if exists public.match_documents" in migration
    assert "source_files text[] default null" in migration
    assert "where source_files is null" in migration
    assert "contract_chunks.source_file = any(source_files)" in migration
