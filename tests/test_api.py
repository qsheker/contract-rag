import logging
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.main import ALLOWED_ORIGINS, MAX_UPLOAD_BYTES, app
from api.reformulate import StandaloneQuery, reformulate_query
from contract_rag.chunker import Chunk, UnsupportedNumberingError
from contract_rag.generator import (
    Answer,
    Citation,
    GenerationError,
    UnsupportedCitationError,
)
from contract_rag.loader import (
    LoaderError,
    LoaderErrorCode,
    NoTextLayerError,
    UnsupportedFormatError,
)
from contract_rag.retriever import ContextChunk, RetrievedChunk

TEST_MODEL = "ollama/qwen2.5:7b"
ORIGIN = ALLOWED_ORIGINS[0]
CONDENSED_QUESTION = "За сколько дней нужно уведомить о досрочном расторжении договора?"
HISTORY = [
    {"role": "user", "content": "Можно ли расторгнуть договор досрочно?"},
    {"role": "assistant", "content": "Да, при письменном уведомлении другой стороны."},
]


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


# Filled by the retrieval fake so a test can assert what scope reached search.
SEARCH_SCOPES: list[list[str] | None] = []

SUB_ITEM_CHUNK = Chunk(
    text="2.2.1. в безналичном порядке на счёт, указанный Исполнителем;",
    clause_id="2.2.1",
    page_number=4,
    source_file="contract_01.pdf",
    detected_strategy="dotted_numbering",
)


def make_chunk(clause_id: str | None = "2.2", page_number: int | None = 4) -> Chunk:
    return Chunk(
        text="Оплата производится в течение 10 банковских дней.",
        clause_id=clause_id,
        page_number=page_number,
        source_file="contract_01.pdf",
        detected_strategy="dotted_numbering",
    )


@pytest.fixture
# TestClient is used without its context manager on purpose: entering it would
# run the lifespan hook and load the real embedding model.
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def condensation_through_a_fake_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the real condensation step against a provider that always agrees.

    Patching the step out entirely would leave the endpoint's own handling of an
    empty history untested, which is the whole of AC-1.
    """

    payload = StandaloneQuery(query=CONDENSED_QUESTION).model_dump_json()

    def fake_completion(**kwargs: Any) -> FakeResponse:
        return FakeResponse(payload)

    def condense(history: Sequence[dict[str, str]], new_message: str) -> str:
        return reformulate_query(
            history,
            new_message,
            model=TEST_MODEL,
            completion_fn=fake_completion,
        )

    monkeypatch.setattr(main_module, "reformulate_query", condense)


@pytest.fixture(autouse=True)
def retrieval_and_generation(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve one chunk and an answer citing it, recording the query used."""

    queries: list[str] = []
    chunk = make_chunk()
    scopes = SEARCH_SCOPES
    scopes.clear()

    def fake_retrieve(query: str, k: int = 5, **kwargs: Any) -> list[RetrievedChunk]:
        queries.append(query)
        scopes.append(kwargs.get("source_files"))
        return [RetrievedChunk(chunk=chunk, chunk_id="contract_01.pdf::2.2", similarity=0.81)]

    def fake_expand(
        hits: Sequence[RetrievedChunk], **kwargs: Any
    ) -> list[ContextChunk]:
        """Stand in for the real expansion: the hit plus one sub-item of it.

        Faked rather than patched out, because the endpoint has to be seen
        handing generation more than search returned - that is the whole point
        of the step, and the excerpt list is how the UI learns about it.
        """

        context = [
            ContextChunk(chunk=hit.chunk, chunk_id=hit.chunk_id, similarity=hit.similarity)
            for hit in hits
        ]
        context.append(
            ContextChunk(chunk=SUB_ITEM_CHUNK, chunk_id="contract_01.pdf::2.2.1", similarity=None)
        )
        return context

    def fake_generate_answer(query: str, chunks: Sequence[Chunk], **kwargs: Any) -> Answer:
        return Answer(
            text="Оплата производится в течение 10 банковских дней.",
            citations=[
                Citation(
                    clause_id=chunks[0].clause_id,
                    page_number=chunks[0].page_number,
                    source_file=chunks[0].source_file,
                )
            ],
        )

    monkeypatch.setattr(main_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(main_module, "expand_with_related_clauses", fake_expand)
    monkeypatch.setattr(main_module, "generate_answer", fake_generate_answer)
    return queries


def patch_ingestion(
    monkeypatch: pytest.MonkeyPatch,
    ingest: Callable[..., int],
) -> None:
    monkeypatch.setattr(main_module, "ingest_document", ingest)


def test_an_empty_history_leaves_the_question_untouched(
    client: TestClient,
    retrieval_and_generation: list[str],
) -> None:
    response = client.post("/chat", json={"message": "Какой срок оплаты?", "history": []})

    assert response.status_code == 200
    assert response.json()["standalone_query"] == "Какой срок оплаты?"
    assert retrieval_and_generation == ["Какой срок оплаты?"]


def test_a_follow_up_is_retrieved_as_a_standalone_question(
    client: TestClient,
    retrieval_and_generation: list[str],
) -> None:
    response = client.post("/chat", json={"message": "а за сколько дней?", "history": HISTORY})

    assert response.status_code == 200
    assert response.json()["standalone_query"] == CONDENSED_QUESTION
    # The condensed form, not the literal message, is what reaches the index.
    assert retrieval_and_generation == [CONDENSED_QUESTION]


def test_the_response_carries_the_answer_its_citations_and_the_query(
    client: TestClient,
) -> None:
    response = client.post("/chat", json={"message": "Какой срок оплаты?"})

    payload = response.json()
    assert set(payload) == {"answer", "citations", "standalone_query", "excerpts"}
    assert payload["answer"] == "Оплата производится в течение 10 банковских дней."
    assert payload["citations"] == [
        {"clause_id": "2.2", "page_number": 4, "source_file": "contract_01.pdf"}
    ]


def test_every_citation_arrives_with_the_text_it_points_at(client: TestClient) -> None:
    # A citation the reader cannot open is only half a citation: the badge names
    # a clause, and the text is what lets them check the answer against it.
    payload = client.post("/chat", json={"message": "Какой срок оплаты?"}).json()

    cited = [excerpt for excerpt in payload["excerpts"] if excerpt["cited"]]
    assert [excerpt["clause_id"] for excerpt in cited] == ["2.2"]
    assert cited[0]["text"] == "Оплата производится в течение 10 банковских дней."
    assert {citation["clause_id"] for citation in payload["citations"]} == {
        excerpt["clause_id"] for excerpt in cited
    }


def test_the_excerpts_show_what_was_supplied_but_left_uncited(client: TestClient) -> None:
    """A weak answer is explained by what was in front of the model.

    The sub-item pulled in beside the hit is exactly that: it went to the model,
    the answer did not use it, and without this the reader could not tell the
    difference between "the contract does not say" and "search brought nothing
    that says it".
    """

    payload = client.post("/chat", json={"message": "Какой срок оплаты?"}).json()

    uncited = [excerpt for excerpt in payload["excerpts"] if not excerpt["cited"]]
    assert [excerpt["clause_id"] for excerpt in uncited] == ["2.2.1"]
    # Pulled in as a neighbour rather than found by search, and saying so is how
    # the UI can separate the two.
    assert uncited[0]["similarity"] is None
    assert payload["excerpts"][0]["similarity"] == 0.81


def test_a_pageless_citation_keeps_a_null_page_number(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module,
        "retrieve",
        lambda query, k=5, **kwargs: [
            RetrievedChunk(
                chunk=make_chunk(clause_id="1.1", page_number=None),
                chunk_id="contract.docx::1.1",
                similarity=0.7,
            )
        ],
    )

    response = client.post("/chat", json={"message": "Какой срок оплаты?"})

    assert response.json()["citations"][0]["page_number"] is None


def test_the_browser_origin_is_allowed(client: TestClient) -> None:
    response = client.post(
        "/chat",
        json={"message": "Какой срок оплаты?"},
        headers={"Origin": ORIGIN},
    )

    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_the_preflight_request_is_answered(client: TestClient) -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN


@pytest.mark.parametrize("message", ["", "   "])
def test_an_empty_message_is_rejected(client: TestClient, message: str) -> None:
    assert client.post("/chat", json={"message": message}).status_code == 422


def test_a_generation_failure_is_reported_as_a_provider_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(query: str, chunks: Sequence[Chunk], **kwargs: Any) -> Answer:
        raise GenerationError("Generation failed for model ollama/qwen2.5:7b: connection refused")

    monkeypatch.setattr(main_module, "generate_answer", fail)

    response = client.post("/chat", json={"message": "Какой срок оплаты?"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Модель генерации" in detail
    assert "connection refused" not in detail


def test_a_rejected_citation_explains_itself_without_leaking_row_ids(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invented = [Citation(clause_id="5.2.1", page_number=None, source_file="договор.docx")]

    def reject(query: str, chunks: Sequence[Chunk], **kwargs: Any) -> Answer:
        raise UnsupportedCitationError(invented, "договор.docx::5.2.1::page-None")

    monkeypatch.setattr(main_module, "generate_answer", reject)

    with caplog.at_level(logging.WARNING):
        response = client.post("/chat", json={"message": "Что я должен делать на работе?"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "нельзя проверить" in detail
    # The row ids are diagnostics for the developer, not text for the reader.
    assert "page-None" not in detail
    assert "5.2.1" not in detail
    assert "договор.docx::5.2.1::page-None" in caplog.text


def test_a_pdf_upload_reports_its_chunk_count_without_warnings(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ingestion(monkeypatch, lambda file_bytes, filename, **kwargs: 12)

    response = client.post(
        "/documents",
        files={"file": ("contract_01.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "contract_01.pdf",
        "chunks_indexed": 12,
        "warnings": [],
    }


@pytest.mark.parametrize("filename", ["contract.docx", "contract.txt"])
def test_a_pageless_upload_warns_that_citations_carry_no_page(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    patch_ingestion(monkeypatch, lambda file_bytes, name, **kwargs: 5)

    response = client.post("/documents", files={"file": (filename, b"fake bytes")})

    payload = response.json()
    assert payload["chunks_indexed"] == 5
    assert len(payload["warnings"]) == 1
    assert "no page number" in payload["warnings"][0]


def test_the_stored_file_name_drops_any_directory_the_browser_sent(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    def record(file_bytes: bytes, filename: str, **kwargs: Any) -> int:
        received.append(filename)
        return 1

    patch_ingestion(monkeypatch, record)

    response = client.post("/documents", files={"file": ("../../contract.txt", b"1.1 Clause.")})

    assert received == ["contract.txt"]
    assert response.json()["filename"] == "contract.txt"


def test_an_unsupported_format_is_a_client_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(file_bytes: bytes, filename: str, **kwargs: Any) -> int:
        raise UnsupportedFormatError(filename, ".rtf")

    patch_ingestion(monkeypatch, reject)

    response = client.post("/documents", files={"file": ("contract.rtf", b"{\\rtf1}")})

    assert response.status_code == 415
    assert "PDF, DOCX and TXT" in response.json()["detail"]


def test_a_scanned_pdf_says_so_instead_of_failing_opaquely(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(file_bytes: bytes, filename: str, **kwargs: Any) -> int:
        raise NoTextLayerError(filename)

    patch_ingestion(monkeypatch, reject)

    response = client.post("/documents", files={"file": ("scan.pdf", b"%PDF-1.4 fake")})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "scan.pdf" in detail
    assert "OCR" in detail


def test_an_unreadable_file_never_leaks_the_temporary_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(file_bytes: bytes, filename: str, **kwargs: Any) -> int:
        raise LoaderError(
            LoaderErrorCode.INVALID_PDF,
            "/var/folders/xyz/contract-rag-upload-abc/broken.pdf",
            "File is not a readable PDF: /var/folders/xyz/contract-rag-upload-abc/broken.pdf",
        )

    patch_ingestion(monkeypatch, reject)

    response = client.post("/documents", files={"file": ("broken.pdf", b"not a pdf")})

    assert response.status_code == 422
    assert response.json()["detail"] == "broken.pdf is not a readable PDF."


def test_numbering_the_chunker_cannot_split_is_a_client_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(file_bytes: bytes, filename: str, **kwargs: Any) -> int:
        raise UnsupportedNumberingError(filename)

    patch_ingestion(monkeypatch, reject)

    response = client.post("/documents", files={"file": ("flat.pdf", b"%PDF-1.4 fake")})

    assert response.status_code == 422
    assert "clause numbering" in response.json()["detail"]


def test_an_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("contract.pdf", b"")})

    assert response.status_code == 400


def test_an_oversized_upload_is_rejected(client: TestClient) -> None:
    oversized = b"x" * (MAX_UPLOAD_BYTES + 1)

    response = client.post("/documents", files={"file": ("contract.pdf", oversized)})

    assert response.status_code == 413


def test_an_unexpected_crash_answers_with_a_readable_cors_enabled_500(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crash(file_bytes: bytes, filename: str, **kwargs: Any) -> int:
        raise RuntimeError('null value in column "page_number" violates not-null constraint')

    patch_ingestion(monkeypatch, crash)

    response = client.post(
        "/documents",
        files={"file": ("contract.txt", b"1.1 Clause.")},
        headers={"Origin": ORIGIN},
    )

    assert response.status_code == 500
    assert "not-null constraint" in response.json()["detail"]
    # Without the header the browser discards the body and the UI can only say
    # "Failed to fetch", which is what sent us looking in the server log.
    assert response.headers["access-control-allow-origin"] == ORIGIN

