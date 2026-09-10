"""Rewrite a follow-up message into a question that stands on its own.

Retrieval sees one string and no conversation, so "а на сколько дней?" embeds
into nothing useful. Condensing the question against the history before the
search is what makes a second turn work at all.

This runs on GENERATION_MODEL rather than a separate model: rewriting a question
is mechanical, and the self-preference bias that forces eval/judge.py onto a
different model has nothing to grade here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from contract_rag.generator import get_generation_model_from_env

# Deterministic rewriting: the same follow-up must condense the same way twice.
TEMPERATURE = 0
# Only the tail of the conversation is sent. A follow-up refers to what was just
# said, and an unbounded history would grow the prompt without improving it.
HISTORY_TURN_LIMIT = 6

SYSTEM_PROMPT = """You rewrite a follow-up message into a self-contained question.

You are given the recent turns of a conversation about a legal contract and the
user's newest message. Return that newest message as a question that can be
understood on its own, with every pronoun and ellipsis resolved from the history.

Rules:
1. Resolve references. "а на сколько дней?" after a question about the notice
   period becomes "На сколько дней нужно уведомить о расторжении договора?".
2. Change nothing else. Keep the user's language, wording and level of detail;
   add no facts, no clause numbers and no assumptions the history does not state.
3. If the newest message already stands on its own, return it unchanged.
4. Never answer the question. Return only the rewritten question."""

logger = logging.getLogger(__name__)


class StandaloneQuery(BaseModel):
    """The newest message rewritten so retrieval can be run on it alone."""

    query: str


class QueryReformulator:
    """Condenses a follow-up against its history through the configured provider.

    ``completion_fn`` defaults to ``litellm.completion``; injecting it keeps
    tests off the network, exactly as in the generator and the judge.
    """

    def __init__(
        self,
        model: str | None = None,
        completion_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._model = model
        self._completion_fn = completion_fn

    def reformulate(self, history: Sequence[dict[str, str]], new_message: str) -> str:
        """Return a standalone form of ``new_message``, or ``new_message`` itself.

        A failure here must never take the chat down with it: the unrewritten
        message is a worse search query, not a broken one, so every error ends in
        a logged warning and the original text.
        """

        turns = _recent_turns(history)
        if not turns:
            # Nothing to resolve against, so nothing to gain from a model call.
            return new_message

        try:
            model = self._model or get_generation_model_from_env()
            response = self._get_completion_fn()(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(turns, new_message)},
                ],
                response_format=StandaloneQuery,
                temperature=TEMPERATURE,
            )
            standalone_query = _parse_standalone_query(response)
        except Exception as error:  # every provider raises its own exception type
            logger.warning(
                "Query condensation failed, falling back to the literal message: %s",
                error,
            )
            return new_message

        return standalone_query

    def _get_completion_fn(self) -> Callable[..., Any]:
        if self._completion_fn is None:
            # Imported lazily: litellm pulls in a large dependency tree.
            from litellm import completion

            self._completion_fn = completion
        return self._completion_fn


def reformulate_query(
    history: Sequence[dict[str, str]],
    new_message: str,
    *,
    model: str | None = None,
    completion_fn: Callable[..., Any] | None = None,
) -> str:
    """Condense ``new_message`` against ``history`` into a standalone question."""

    return QueryReformulator(model=model, completion_fn=completion_fn).reformulate(
        history, new_message
    )


def _recent_turns(history: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    turns = [turn for turn in history if turn.get("content", "").strip()]
    return turns[-HISTORY_TURN_LIMIT:]


def _build_user_message(turns: Sequence[dict[str, str]], new_message: str) -> str:
    conversation = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    return f"Conversation so far:\n{conversation}\n\nNewest message: {new_message}"


def _parse_standalone_query(response: Any) -> str:
    """Extract the rewritten question, refusing anything the schema cannot hold.

    Structured output rather than raw text on purpose: a 7B model asked for a
    bare question routinely prefixes "Sure, here is the standalone question:",
    and that preamble would go straight into the query embedding with nothing to
    tell it apart from the question itself.
    """

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Provider returned an empty message")

    try:
        standalone_query = StandaloneQuery.model_validate_json(content).query.strip()
    except ValidationError as error:
        raise ValueError(f"Provider returned an unusable payload: {content!r}") from error

    if not standalone_query:
        raise ValueError("Provider returned an empty standalone query")
    return standalone_query
