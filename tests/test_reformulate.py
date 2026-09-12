import logging
from typing import Any

import pytest

from api.reformulate import (
    HISTORY_TURN_LIMIT,
    SYSTEM_PROMPT,
    TEMPERATURE,
    QueryReformulator,
    StandaloneQuery,
    reformulate_query,
)

TEST_MODEL = "ollama/qwen2.5:7b"
NOTICE_PERIOD_HISTORY = [
    {"role": "user", "content": "Можно ли расторгнуть договор досрочно?"},
    {"role": "assistant", "content": "Да, при письменном уведомлении другой стороны."},
]


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


def standalone_payload(query: str) -> str:
    return StandaloneQuery(query=query).model_dump_json()


def test_empty_history_skips_the_model_entirely() -> None:
    completion = FakeCompletion(standalone_payload("never used"))

    standalone_query = reformulate_query(
        [],
        "Какой срок оплаты?",
        model=TEST_MODEL,
        completion_fn=completion,
    )

    assert standalone_query == "Какой срок оплаты?"
    assert completion.calls == []


def test_history_of_blank_turns_counts_as_no_history() -> None:
    completion = FakeCompletion(standalone_payload("never used"))

    standalone_query = reformulate_query(
        [{"role": "user", "content": "   "}],
        "Какой срок оплаты?",
        model=TEST_MODEL,
        completion_fn=completion,
    )

    assert standalone_query == "Какой срок оплаты?"
    assert completion.calls == []


def test_follow_up_is_condensed_into_a_standalone_question() -> None:
    condensed = "За сколько дней нужно уведомить о досрочном расторжении договора?"
    completion = FakeCompletion(standalone_payload(condensed))

    standalone_query = reformulate_query(
        NOTICE_PERIOD_HISTORY,
        "а за сколько дней?",
        model=TEST_MODEL,
        completion_fn=completion,
    )

    assert standalone_query == condensed


def test_the_model_receives_the_history_and_the_newest_message() -> None:
    completion = FakeCompletion(standalone_payload("Standalone question?"))

    reformulate_query(
        NOTICE_PERIOD_HISTORY,
        "а за сколько дней?",
        model=TEST_MODEL,
        completion_fn=completion,
    )

    call = completion.calls[0]
    assert call["model"] == TEST_MODEL
    assert call["temperature"] == TEMPERATURE
    assert call["response_format"] is StandaloneQuery
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    user_message = call["messages"][1]["content"]
    assert "Можно ли расторгнуть договор досрочно?" in user_message
    assert "а за сколько дней?" in user_message


def test_only_the_most_recent_turns_are_sent() -> None:
    history = [
        {"role": "user", "content": f"вопрос {index}"} for index in range(HISTORY_TURN_LIMIT + 4)
    ]
    completion = FakeCompletion(standalone_payload("Standalone question?"))

    reformulate_query(history, "а дальше?", model=TEST_MODEL, completion_fn=completion)

    user_message = completion.calls[0]["messages"][1]["content"]
    assert "вопрос 0" not in user_message
    assert f"вопрос {HISTORY_TURN_LIMIT + 3}" in user_message


def test_a_provider_failure_falls_back_to_the_literal_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    completion = FakeCompletion(error=RuntimeError("connection refused"))

    with caplog.at_level(logging.WARNING):
        standalone_query = reformulate_query(
            NOTICE_PERIOD_HISTORY,
            "а за сколько дней?",
            model=TEST_MODEL,
            completion_fn=completion,
        )

    # The chat must survive a condensation failure: an unrewritten follow-up is
    # a worse query, not a broken one.
    assert standalone_query == "а за сколько дней?"
    assert "connection refused" in caplog.text


@pytest.mark.parametrize(
    "content",
    [None, "", "Sure, here is the standalone question!", '{"query": ""}', '{"query": "   "}'],
)
def test_an_unusable_payload_falls_back_to_the_literal_message(
    content: str | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    completion = FakeCompletion(content)

    with caplog.at_level(logging.WARNING):
        standalone_query = reformulate_query(
            NOTICE_PERIOD_HISTORY,
            "а за сколько дней?",
            model=TEST_MODEL,
            completion_fn=completion,
        )

    assert standalone_query == "а за сколько дней?"
    assert "falling back" in caplog.text


def test_the_condensed_question_is_stripped() -> None:
    completion = FakeCompletion(standalone_payload("  Какой срок оплаты?  "))

    standalone_query = QueryReformulator(
        model=TEST_MODEL,
        completion_fn=completion,
    ).reformulate(NOTICE_PERIOD_HISTORY, "а срок?")

    assert standalone_query == "Какой срок оплаты?"


def test_a_missing_model_variable_falls_back_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Reformulation is optional by design, so even a configuration mistake must
    # degrade to the literal message rather than fail the request.
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.setattr("api.reformulate.get_generation_model_from_env", _raise_missing_model)
    completion = FakeCompletion(standalone_payload("never used"))

    with caplog.at_level(logging.WARNING):
        standalone_query = reformulate_query(
            NOTICE_PERIOD_HISTORY,
            "а за сколько дней?",
            completion_fn=completion,
        )

    assert standalone_query == "а за сколько дней?"
    assert completion.calls == []


def _raise_missing_model() -> str:
    raise RuntimeError("Missing required environment variables: GENERATION_MODEL")
