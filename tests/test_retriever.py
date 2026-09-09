from pathlib import Path
from typing import Any

import pytest

import contract_rag.retriever.retriever as retriever_module
from contract_rag.embeddings import (
    DOCUMENT_PREFIX,
    EMBEDDING_DIMENSION,
    MODEL_TOKEN_LIMIT,
    QUERY_PREFIX,
    RoSBERTaEmbedder,
)
from contract_rag.retriever import RetrievedChunk, SupabaseRetriever, retrieve

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
        ranked = sorted(self.rows, key=lambda row: row["similarity"], reverse=True)
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
