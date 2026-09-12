"""Generate cited answers from retrieved chunks through LiteLLM."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from contract_rag.chunker import Chunk

MODEL_ENV_VAR = "GENERATION_MODEL"
# Deterministic output: the same question over the same chunks must not drift.
TEMPERATURE = 0
# Ollama runs a model with a 4096-token window by default however large a window
# the model itself supports, and silently truncates anything longer - taking the
# rules at the top of the prompt rather than the excerpts at the bottom. Asked
# for explicitly because a lead-in clause now arrives with its sub-items, so the
# excerpts alone can approach that ceiling. Sent only to Ollama: it is an
# option of that server, and other providers reject what they do not know.
OLLAMA_MODEL_PREFIX = "ollama/"
OLLAMA_CONTEXT_TOKENS = 8192

SYSTEM_PROMPT = """You answer questions about legal contracts.

Rules:
1. Use ONLY the numbered contract excerpts in the user message. Never rely on
   outside knowledge, and never infer beyond what an excerpt states.
2. If the excerpts do not contain the answer, say so plainly in `text` and
   return an empty `citations` list. Do not guess.
3. Every factual statement in `text` must be supported by an excerpt you cite.
4. Copy `clause_id`, `page_number` and `source_file` into each citation exactly
   as the excerpt shows them. Use null where the excerpt shows null - some
   formats have no page numbers.
5. Answer in the same language as the question, whatever language the excerpts
   are in.
6. Write `text` as prose a person reads: never repeat `clause_id`,
   `source_file` or `page_number` inside it. They are returned in `citations`
   and shown beside the answer, so spelling them out again only clutters it."""

# Rule 5 is not enough on its own: a 7B model reads the language off the
# excerpts instead of the question, and a Russian question over a mixed corpus
# has come back in English and in Chinese. The language is worked out here and
# named next to the question, where the instruction is hardest to overlook.
LANGUAGE_INSTRUCTION = "Write `text` in {language}."
FALLBACK_LANGUAGE_INSTRUCTION = "Write `text` in the language of the question above."
# Kazakh and Russian share the alphabet, so only these letters tell them apart.
KAZAKH_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_LATIN = re.compile(r"[A-Za-z]")


class Citation(BaseModel):
    """One contract location backing a statement in the answer."""

    clause_id: str | None
    page_number: int | None
    source_file: str


class Answer(BaseModel):
    """A generated answer together with the excerpts it stands on."""

    text: str
    citations: list[Citation]


class GenerationError(Exception):
    """Raised when the provider fails or returns something unusable."""


class UnsupportedCitationError(GenerationError):
    """Raised when the answer cites something that was never in the context.

    A distinct type rather than a message to grep: a caller showing this to a
    person needs to say "the answer was withheld because its references could
    not be verified", which is a different sentence from "the provider broke".
    """

    def __init__(self, citations: Sequence[Citation], described: str) -> None:
        super().__init__(f"Answer cited excerpts that were not supplied: {described}")
        self.citations = list(citations)


def get_generation_model_from_env() -> str:
    """Return the configured model id, or fail loudly at start-up.

    There is deliberately no default: silently falling back to a local model
    that may not be running turns a configuration mistake into a timeout.
    """

    load_dotenv()
    model = os.environ.get(MODEL_ENV_VAR)
    if not model:
        raise RuntimeError(f"Missing required environment variables: {MODEL_ENV_VAR}")
    return model


class AnswerGenerator:
    """Turns a question plus its retrieved chunks into a cited answer.

    ``completion_fn`` defaults to ``litellm.completion``; injecting it keeps
    tests off the network and leaves room for retries or caching later.
    """

    def __init__(
        self,
        model: str | None = None,
        completion_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._model = model
        self._completion_fn = completion_fn

    def generate(self, query: str, chunks: Sequence[Chunk]) -> Answer:
        """Answer ``query`` using only ``chunks``, rejecting invented citations."""

        model = self._model or get_generation_model_from_env()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, chunks)},
        ]

        try:
            response = self._get_completion_fn()(
                model=model,
                messages=messages,
                response_format=Answer,
                temperature=TEMPERATURE,
                **_provider_options(model),
            )
        except Exception as error:  # every provider raises its own exception type
            raise GenerationError(f"Generation failed for model {model}: {error}") from error

        answer = _parse_answer(response)
        _reject_unsupported_citations(answer, chunks)
        return answer

    def _get_completion_fn(self) -> Callable[..., Any]:
        if self._completion_fn is None:
            # Imported lazily: litellm pulls in a large dependency tree.
            from litellm import completion

            self._completion_fn = completion
        return self._completion_fn


def generate_answer(
    query: str,
    chunks: Sequence[Chunk],
    *,
    model: str | None = None,
    completion_fn: Callable[..., Any] | None = None,
) -> Answer:
    """Answer ``query`` from ``chunks`` using the configured provider."""

    return AnswerGenerator(model=model, completion_fn=completion_fn).generate(query, chunks)


def _provider_options(model: str) -> dict[str, Any]:
    """Return the provider-specific options this model needs, if any."""

    if model.startswith(OLLAMA_MODEL_PREFIX):
        return {"num_ctx": OLLAMA_CONTEXT_TOKENS}
    return {}


def _build_user_message(query: str, chunks: Sequence[Chunk]) -> str:
    if not chunks:
        excerpts = "(no excerpts were retrieved)"
    else:
        excerpts = "\n\n".join(
            _format_excerpt(position, chunk) for position, chunk in enumerate(chunks, start=1)
        )
    return (
        f"Contract excerpts:\n\n{excerpts}\n\n"
        f"Question: {query}\n\n{_language_instruction(query)}"
    )


def detect_question_language(query: str) -> str | None:
    """Name the language of ``query``, or None when the script does not say.

    Deliberately a script check rather than a library or a model call: the
    three languages this corpus is asked in are separable by alphabet, and the
    answer has to be the same on every run for the same question.
    """

    if KAZAKH_LETTERS & set(query):
        return "Kazakh"
    cyrillic = len(_CYRILLIC.findall(query))
    latin = len(_LATIN.findall(query))
    if cyrillic and cyrillic >= latin:
        return "Russian"
    if latin:
        return "English"
    return None


def _language_instruction(query: str) -> str:
    language = detect_question_language(query)
    if language is None:
        return FALLBACK_LANGUAGE_INSTRUCTION
    return LANGUAGE_INSTRUCTION.format(language=language)


def _format_excerpt(position: int, chunk: Chunk) -> str:
    return (
        f"[{position}]\n"
        f"source_file: {chunk.source_file}\n"
        f"clause_id: {_as_literal(chunk.clause_id)}\n"
        f"page_number: {_as_literal(chunk.page_number)}\n"
        f"text: {chunk.text}"
    )


def _as_literal(value: object) -> str:
    return "null" if value is None else str(value)


def _parse_answer(response: Any) -> Answer:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise GenerationError(f"Provider response carried no message content: {response!r}") from (
            error
        )
    if not content:
        raise GenerationError("Provider returned an empty message")

    try:
        # Structured output only: parsing free text would defeat the schema.
        return Answer.model_validate_json(content)
    except ValidationError as error:
        raise GenerationError(f"Provider returned an unusable payload: {content!r}") from error


def _reject_unsupported_citations(answer: Answer, chunks: Sequence[Chunk]) -> None:
    """Fail when a citation points at something that was never in the context.

    A fabricated reference to a contract clause is worse than no answer, so an
    unmatched citation is an error rather than something to quietly drop.
    """

    supplied = {(chunk.source_file, chunk.clause_id, chunk.page_number) for chunk in chunks}
    unsupported = [
        citation
        for citation in answer.citations
        if (citation.source_file, citation.clause_id, citation.page_number) not in supplied
    ]
    if unsupported:
        described = ", ".join(
            f"{citation.source_file}::{citation.clause_id}::page-{citation.page_number}"
            for citation in unsupported
        )
        raise UnsupportedCitationError(unsupported, described)
