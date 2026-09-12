from typing import Any

import pytest

import contract_rag.generator.generator as generator_module
from contract_rag.chunker import Chunk
from contract_rag.generator import (
    MODEL_ENV_VAR,
    OLLAMA_CONTEXT_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    Answer,
    AnswerGenerator,
    Citation,
    GenerationError,
    detect_question_language,
    generate_answer,
    get_generation_model_from_env,
)

TEST_MODEL = "ollama/qwen2.5:7b"


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletion:
    """Records the call and replays a prepared payload or provider failure."""

    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeResponse(self.content)


def make_chunk(
    clause_id: str | None,
    *,
    text: str = "Оплата производится в течение 10 банковских дней.",
    page_number: int | None = 4,
    source_file: str = "contract.pdf",
) -> Chunk:
    return Chunk(
        text=text,
        clause_id=clause_id,
        page_number=page_number,
        source_file=source_file,
        detected_strategy="dotted_numbering",
    )


def answer_payload(text: str, citations: list[dict[str, Any]]) -> str:
    return Answer(
        text=text,
        citations=[Citation(**citation) for citation in citations],
    ).model_dump_json()


def test_answer_carries_text_and_citations_for_a_supported_question() -> None:
    chunk = make_chunk("2.2")
    completion = FakeCompletion(
        answer_payload(
            "Оплата производится в течение 10 банковских дней.",
            [{"clause_id": "2.2", "page_number": 4, "source_file": "contract.pdf"}],
        )
    )

    answer = generate_answer(
        "Какой срок оплаты?",
        [chunk],
        model=TEST_MODEL,
        completion_fn=completion,
    )

    assert answer.text == "Оплата производится в течение 10 банковских дней."
    assert [citation.clause_id for citation in answer.citations] == ["2.2"]
    assert answer.citations[0].page_number == 4
    assert answer.citations[0].source_file == "contract.pdf"


def test_missing_answer_yields_no_citations() -> None:
    completion = FakeCompletion(
        answer_payload("В предоставленных пунктах ответа нет.", [])
    )

    answer = generate_answer(
        "Какой размер уставного капитала?",
        [make_chunk("2.2")],
        model=TEST_MODEL,
        completion_fn=completion,
    )

    assert answer.citations == []
    assert answer.text


def test_request_pins_schema_temperature_and_model() -> None:
    completion = FakeCompletion(answer_payload("Ответ.", []))

    generate_answer("Вопрос?", [make_chunk("1.1")], model=TEST_MODEL, completion_fn=completion)

    request = completion.calls[0]
    assert request["model"] == TEST_MODEL
    assert request["temperature"] == TEMPERATURE == 0
    assert request["response_format"] is Answer
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_prompt_carries_every_chunk_with_its_citation_metadata() -> None:
    completion = FakeCompletion(answer_payload("Ответ.", []))
    chunks = [
        make_chunk("1.1", text="Первый пункт.", page_number=1),
        make_chunk(None, text="Преамбула.", page_number=2),
        make_chunk("3.1", text="Из DOCX.", page_number=None, source_file="agreement.docx"),
    ]

    generate_answer("Вопрос?", chunks, model=TEST_MODEL, completion_fn=completion)

    user_message = completion.calls[0]["messages"][1]["content"]
    assert "Первый пункт." in user_message
    assert "Преамбула." in user_message
    assert "Из DOCX." in user_message
    assert "clause_id: 1.1" in user_message
    assert "page_number: 1" in user_message
    # A preamble has no clause of its own and a DOCX has no pages; both must reach
    # the model as null so it copies null back instead of inventing a reference.
    assert "clause_id: null" in user_message
    assert "page_number: null" in user_message
    assert "Question: Вопрос?" in user_message


def test_provider_failure_becomes_generation_error() -> None:
    completion = FakeCompletion(error=RuntimeError("connection refused"))

    with pytest.raises(GenerationError, match="connection refused"):
        generate_answer("Вопрос?", [make_chunk("1.1")], model=TEST_MODEL, completion_fn=completion)


@pytest.mark.parametrize(
    "content",
    ["not json at all", '{"text": "Ответ."}', '{"citations": []}', ""],
)
def test_unusable_payload_becomes_generation_error(content: str) -> None:
    completion = FakeCompletion(content)

    with pytest.raises(GenerationError):
        generate_answer("Вопрос?", [make_chunk("1.1")], model=TEST_MODEL, completion_fn=completion)


def test_response_without_choices_becomes_generation_error() -> None:
    def completion_fn(**kwargs: Any) -> object:
        return object()

    with pytest.raises(GenerationError, match="no message content"):
        generate_answer(
            "Вопрос?",
            [make_chunk("1.1")],
            model=TEST_MODEL,
            completion_fn=completion_fn,
        )


def test_citation_to_a_clause_that_was_never_supplied_is_rejected() -> None:
    completion = FakeCompletion(
        answer_payload(
            "Договор расторгается в одностороннем порядке.",
            [{"clause_id": "9.9", "page_number": 12, "source_file": "contract.pdf"}],
        )
    )

    with pytest.raises(GenerationError, match="not supplied"):
        generate_answer("Вопрос?", [make_chunk("2.2")], model=TEST_MODEL, completion_fn=completion)


def test_citation_with_the_wrong_page_is_rejected() -> None:
    completion = FakeCompletion(
        answer_payload(
            "Оплата в течение 10 дней.",
            [{"clause_id": "2.2", "page_number": 3, "source_file": "contract.pdf"}],
        )
    )

    with pytest.raises(GenerationError, match="not supplied"):
        generate_answer("Вопрос?", [make_chunk("2.2")], model=TEST_MODEL, completion_fn=completion)


def test_citation_to_a_clauseless_preamble_is_accepted() -> None:
    chunk = make_chunk(None, text="Преамбула договора.", page_number=1)
    completion = FakeCompletion(
        answer_payload(
            "Стороны договора названы в преамбуле.",
            [{"clause_id": None, "page_number": 1, "source_file": "contract.pdf"}],
        )
    )

    answer = generate_answer("Кто стороны?", [chunk], model=TEST_MODEL, completion_fn=completion)

    assert answer.citations[0].clause_id is None
    assert answer.citations[0].page_number == 1


def test_empty_chunk_list_still_asks_the_model_without_excerpts() -> None:
    completion = FakeCompletion(answer_payload("Информация не найдена.", []))

    answer = generate_answer("Вопрос?", [], model=TEST_MODEL, completion_fn=completion)

    assert answer.citations == []
    assert "(no excerpts were retrieved)" in completion.calls[0]["messages"][1]["content"]


def test_missing_model_configuration_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_module, "load_dotenv", lambda: None)
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=MODEL_ENV_VAR):
        get_generation_model_from_env()

    with pytest.raises(RuntimeError, match=MODEL_ENV_VAR):
        generate_answer("Вопрос?", [make_chunk("1.1")], completion_fn=FakeCompletion("{}"))


def test_model_is_read_from_the_environment_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(generator_module, "load_dotenv", lambda: None)
    monkeypatch.setenv(MODEL_ENV_VAR, "anthropic/claude-sonnet-4-6")
    completion = FakeCompletion(answer_payload("Ответ.", []))

    AnswerGenerator(completion_fn=completion).generate("Вопрос?", [make_chunk("1.1")])

    assert completion.calls[0]["model"] == "anthropic/claude-sonnet-4-6"


def test_system_prompt_forbids_outside_knowledge_and_requires_citations() -> None:
    lowered = SYSTEM_PROMPT.lower()

    assert "only" in lowered
    assert "empty `citations`" in SYSTEM_PROMPT
    assert "clause_id" in SYSTEM_PROMPT
    assert "page_number" in SYSTEM_PROMPT


def test_citation_to_a_pageless_chunk_is_accepted() -> None:
    chunk = make_chunk("2.2", page_number=None, source_file="agreement.docx")
    completion = FakeCompletion(
        answer_payload(
            "Оплата в течение 10 дней.",
            [{"clause_id": "2.2", "page_number": None, "source_file": "agreement.docx"}],
        )
    )

    answer = generate_answer("Вопрос?", [chunk], model=TEST_MODEL, completion_fn=completion)

    assert answer.citations[0].page_number is None
    assert answer.citations[0].clause_id == "2.2"


def test_page_invented_for_a_pageless_chunk_is_rejected() -> None:
    chunk = make_chunk("2.2", page_number=None, source_file="agreement.docx")
    completion = FakeCompletion(
        answer_payload(
            "Оплата в течение 10 дней.",
            [{"clause_id": "2.2", "page_number": 1, "source_file": "agreement.docx"}],
        )
    )

    with pytest.raises(GenerationError, match="not supplied"):
        generate_answer("Вопрос?", [chunk], model=TEST_MODEL, completion_fn=completion)


def test_ollama_is_asked_for_a_window_large_enough_for_the_excerpts() -> None:
    """Ollama's default 4096-token window is smaller than what is now sent.

    A lead-in clause reaches generation together with its sub-items, and an
    over-long prompt is truncated where it hurts most: the rules at the top go
    first, so the model would stop being told to cite at all.
    """

    completion = FakeCompletion(answer_payload("Ответ.", []))

    generate_answer("Вопрос?", [make_chunk("2.2")], model=TEST_MODEL, completion_fn=completion)

    assert completion.calls[0]["num_ctx"] == OLLAMA_CONTEXT_TOKENS


def test_a_provider_that_is_not_ollama_is_not_sent_an_ollama_option() -> None:
    completion = FakeCompletion(answer_payload("Ответ.", []))

    generate_answer(
        "Вопрос?",
        [make_chunk("2.2")],
        model="anthropic/claude-opus-5",
        completion_fn=completion,
    )

    assert "num_ctx" not in completion.calls[0]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("А кто я", "Russian"),
        ("Какой срок оплаты по Article 5?", "Russian"),
        ("Төлем мерзімі қандай?", "Kazakh"),
        ("What is the payment term?", "English"),
        ("支付期限是多少？", None),
    ],
)
def test_the_question_language_is_read_from_its_alphabet(
    query: str, expected: str | None
) -> None:
    # Kazakh shares the Russian alphabet apart from a handful of letters, and a
    # Russian question quoting an English clause name is still Russian.
    assert detect_question_language(query) == expected


def test_the_answer_language_is_named_next_to_the_question() -> None:
    """Rule 5 alone did not hold: a Russian question came back in Chinese.

    The model read the language off the excerpts - half the corpus is English -
    rather than off the question. Naming it beside the question makes it a
    concrete instruction instead of a general one.
    """

    completion = FakeCompletion(answer_payload("Ответ.", []))

    generate_answer("А кто я", [make_chunk("2.2")], model=TEST_MODEL, completion_fn=completion)

    user_message = completion.calls[0]["messages"][1]["content"]
    assert user_message.endswith("Write `text` in Russian.")


def test_an_unrecognised_alphabet_falls_back_to_the_question_itself() -> None:
    completion = FakeCompletion(answer_payload("答案。", []))

    generate_answer(
        "支付期限是多少？",
        [make_chunk("2.2")],
        model=TEST_MODEL,
        completion_fn=completion,
    )

    user_message = completion.calls[0]["messages"][1]["content"]
    assert user_message.endswith("Write `text` in the language of the question above.")
