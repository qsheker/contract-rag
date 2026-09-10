"""Judge whether a generated answer is supported by the excerpt it cites.

The judge deliberately runs on a different model family than generation: a model
grading its own output rates it as faithful more often than an independent one
does, so reusing GENERATION_MODEL here would quietly inflate the score.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

JUDGE_MODEL_ENV_VAR = "JUDGE_MODEL"
GENERATION_MODEL_ENV_VAR = "GENERATION_MODEL"
# Deterministic verdicts: the same pair must not score differently between runs.
TEMPERATURE = 0

SYSTEM_PROMPT = """You check whether an answer is supported by a contract excerpt.

You are given a question, one contract excerpt, and an answer that claims to rest
on that excerpt.

Apply this test. Split the answer into individual claims. Check each claim
against the excerpt, one at a time. If even ONE claim is not stated in the
excerpt, `faithful` is false - no matter how plausible, minor, or generally true
that claim is. Extra correct-sounding detail is the most common failure and must
be marked false.

Examples:

Excerpt: "2.2 Payment is made within 10 banking days of signing the act."
Answer: "Payment is made within 10 banking days of signing the act."
-> faithful: true. Every claim appears in the excerpt.

Excerpt: "2.2 Payment is made within 10 banking days of signing the act."
Answer: "Payment is made within 10 banking days; late payment incurs a 0.5%
daily penalty."
-> faithful: false. The excerpt says nothing about a penalty, so that claim is
unsupported even though the first claim is correct.

Excerpt: "2.2 Payment is made within 10 banking days of signing the act."
Answer: "The excerpt does not state the amount of the charter capital."
-> faithful: true. Refusing to answer adds no unsupported claim.

Judge support only - not style, completeness or helpfulness. A short answer that
adds nothing is faithful; a longer, more helpful one that adds anything is not.

Write `explanation` in English, in one or two sentences, and name the
unsupported claim when there is one."""


class FaithfulnessVerdict(BaseModel):
    """Whether one answer stands on its excerpt, and why the judge thinks so."""

    faithful: bool
    explanation: str


class JudgeError(Exception):
    """Raised when the judge provider fails or returns something unusable."""


def get_judge_model_from_env() -> str:
    """Return the judge model id, refusing to grade with the generating model.

    As with GENERATION_MODEL there is no default: a silent fallback would turn a
    configuration mistake into a timeout, or worse, into a self-graded score.
    """

    load_dotenv()
    judge_model = os.environ.get(JUDGE_MODEL_ENV_VAR)
    generation_model = os.environ.get(GENERATION_MODEL_ENV_VAR)

    missing = [
        name
        for name, value in (
            (JUDGE_MODEL_ENV_VAR, judge_model),
            (GENERATION_MODEL_ENV_VAR, generation_model),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    if judge_model == generation_model:
        raise RuntimeError(
            f"{JUDGE_MODEL_ENV_VAR} must differ from {GENERATION_MODEL_ENV_VAR}; both are "
            f"{judge_model!r}. A model grading its own answers scores them as faithful more "
            f"often than an independent judge, so the resulting rate would be meaningless."
        )

    assert judge_model is not None  # narrowed by the missing-variable check above
    return judge_model


class FaithfulnessJudge:
    """Grades answer/excerpt pairs through the configured judge provider.

    ``completion_fn`` defaults to ``litellm.completion``; injecting it keeps
    tests off the network, exactly as in the generator.
    """

    def __init__(
        self,
        model: str | None = None,
        completion_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._model = model
        self._completion_fn = completion_fn

    def judge(self, question: str, chunk_text: str, answer_text: str) -> FaithfulnessVerdict:
        """Return whether ``answer_text`` follows from ``chunk_text`` alone."""

        model = self._model or get_judge_model_from_env()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Contract excerpt:\n{chunk_text}\n\n"
                    f"Answer to check:\n{answer_text}"
                ),
            },
        ]

        try:
            response = self._get_completion_fn()(
                model=model,
                messages=messages,
                response_format=FaithfulnessVerdict,
                temperature=TEMPERATURE,
            )
        except Exception as error:  # every provider raises its own exception type
            raise JudgeError(f"Judging failed for model {model}: {error}") from error

        return _parse_verdict(response)

    def _get_completion_fn(self) -> Callable[..., Any]:
        if self._completion_fn is None:
            # Imported lazily: litellm pulls in a large dependency tree.
            from litellm import completion

            self._completion_fn = completion
        return self._completion_fn


def judge_faithfulness(
    question: str,
    chunk_text: str,
    answer_text: str,
    *,
    model: str | None = None,
    completion_fn: Callable[..., Any] | None = None,
) -> FaithfulnessVerdict:
    """Check one answer against the excerpt it was built from."""

    return FaithfulnessJudge(model=model, completion_fn=completion_fn).judge(
        question, chunk_text, answer_text
    )


def _parse_verdict(response: Any) -> FaithfulnessVerdict:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise JudgeError(f"Judge response carried no message content: {response!r}") from error
    if not content:
        raise JudgeError("Judge returned an empty message")

    try:
        # Structured output only, for the same reason as in generation: parsing
        # free text would let an unusable verdict pass as a real one.
        return FaithfulnessVerdict.model_validate_json(content)
    except ValidationError as error:
        raise JudgeError(f"Judge returned an unusable payload: {content!r}") from error
