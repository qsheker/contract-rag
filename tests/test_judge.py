from typing import Any

import pytest

import eval.judge as judge_module
from eval.judge import (
    GENERATION_MODEL_ENV_VAR,
    JUDGE_MODEL_ENV_VAR,
    SYSTEM_PROMPT,
    TEMPERATURE,
    FaithfulnessJudge,
    FaithfulnessVerdict,
    JudgeError,
    get_judge_model_from_env,
    judge_faithfulness,
)

TEST_JUDGE_MODEL = "ollama/llama3.1:8b"
TEST_GENERATION_MODEL = "ollama/qwen2.5:7b"


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
    """Records the call and replays a prepared verdict or provider failure."""

    def __init__(self, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeResponse(self.content)


def verdict_payload(faithful: bool, explanation: str = "Because the excerpt says so.") -> str:
    return FaithfulnessVerdict(faithful=faithful, explanation=explanation).model_dump_json()


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # load_dotenv would otherwise pull the developer's real .env into the test.
    monkeypatch.setattr(judge_module, "load_dotenv", lambda: None)
    monkeypatch.delenv(JUDGE_MODEL_ENV_VAR, raising=False)
    monkeypatch.delenv(GENERATION_MODEL_ENV_VAR, raising=False)


def test_verdict_carries_the_flag_and_the_explanation() -> None:
    completion = FakeCompletion(verdict_payload(True, "Every claim appears in the excerpt."))

    verdict = judge_faithfulness(
        "Какой срок оплаты?",
        "2.2 Оплата в течение 10 дней.",
        "Оплата в течение 10 дней.",
        model=TEST_JUDGE_MODEL,
        completion_fn=completion,
    )

    assert verdict.faithful is True
    assert verdict.explanation == "Every claim appears in the excerpt."


def test_unfaithful_answer_is_reported_as_such() -> None:
    completion = FakeCompletion(verdict_payload(False, "The excerpt never mentions a penalty."))

    verdict = judge_faithfulness(
        "Какой штраф?",
        "2.2 Оплата в течение 10 дней.",
        "Штраф составляет 5% в день.",
        model=TEST_JUDGE_MODEL,
        completion_fn=completion,
    )

    assert verdict.faithful is False
    assert "penalty" in verdict.explanation


def test_request_pins_schema_temperature_and_model() -> None:
    completion = FakeCompletion(verdict_payload(True))

    judge_faithfulness(
        "Вопрос?",
        "Текст пункта.",
        "Ответ.",
        model=TEST_JUDGE_MODEL,
        completion_fn=completion,
    )

    request = completion.calls[0]
    assert request["model"] == TEST_JUDGE_MODEL
    assert request["temperature"] == TEMPERATURE == 0
    assert request["response_format"] is FaithfulnessVerdict
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_prompt_carries_question_excerpt_and_answer() -> None:
    completion = FakeCompletion(verdict_payload(True))

    judge_faithfulness(
        "Какой срок оплаты?",
        "2.2 Оплата в течение 10 дней.",
        "Оплата в течение 10 дней.",
        model=TEST_JUDGE_MODEL,
        completion_fn=completion,
    )

    user_message = completion.calls[0]["messages"][1]["content"]
    assert "Question: Какой срок оплаты?" in user_message
    assert "2.2 Оплата в течение 10 дней." in user_message
    assert "Answer to check:" in user_message


def test_provider_failure_becomes_judge_error() -> None:
    completion = FakeCompletion(error=RuntimeError("connection refused"))

    with pytest.raises(JudgeError, match="connection refused"):
        judge_faithfulness("Вопрос?", "Пункт.", "Ответ.", model=TEST_JUDGE_MODEL,
                           completion_fn=completion)


@pytest.mark.parametrize(
    "content",
    ["not json at all", '{"faithful": true}', '{"explanation": "x"}', ""],
)
def test_unusable_payload_becomes_judge_error(content: str) -> None:
    completion = FakeCompletion(content)

    with pytest.raises(JudgeError):
        judge_faithfulness("Вопрос?", "Пункт.", "Ответ.", model=TEST_JUDGE_MODEL,
                           completion_fn=completion)


def test_response_without_choices_becomes_judge_error() -> None:
    def completion_fn(**kwargs: Any) -> object:
        return object()

    with pytest.raises(JudgeError, match="no message content"):
        judge_faithfulness("Вопрос?", "Пункт.", "Ответ.", model=TEST_JUDGE_MODEL,
                           completion_fn=completion_fn)


def test_identical_judge_and_generation_models_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, TEST_GENERATION_MODEL)
    monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, TEST_GENERATION_MODEL)

    with pytest.raises(RuntimeError, match="must differ from"):
        get_judge_model_from_env()


def test_distinct_models_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, TEST_JUDGE_MODEL)
    monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, TEST_GENERATION_MODEL)

    assert get_judge_model_from_env() == TEST_JUDGE_MODEL


@pytest.mark.parametrize(
    ("judge_model", "generation_model", "expected"),
    [
        (None, TEST_GENERATION_MODEL, "JUDGE_MODEL"),
        (TEST_JUDGE_MODEL, None, "GENERATION_MODEL"),
        (None, None, "JUDGE_MODEL, GENERATION_MODEL"),
    ],
)
def test_missing_configuration_names_the_missing_variables(
    monkeypatch: pytest.MonkeyPatch,
    judge_model: str | None,
    generation_model: str | None,
    expected: str,
) -> None:
    if judge_model:
        monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, judge_model)
    if generation_model:
        monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, generation_model)

    with pytest.raises(RuntimeError, match=expected):
        get_judge_model_from_env()


def test_judge_resolves_the_model_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JUDGE_MODEL_ENV_VAR, TEST_JUDGE_MODEL)
    monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, TEST_GENERATION_MODEL)
    completion = FakeCompletion(verdict_payload(True))

    FaithfulnessJudge(completion_fn=completion).judge("Вопрос?", "Пункт.", "Ответ.")

    assert completion.calls[0]["model"] == TEST_JUDGE_MODEL
