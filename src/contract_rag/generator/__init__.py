"""Public API for cited answer generation."""

from contract_rag.generator.generator import (
    MODEL_ENV_VAR,
    SYSTEM_PROMPT,
    TEMPERATURE,
    Answer,
    AnswerGenerator,
    Citation,
    GenerationError,
    UnsupportedCitationError,
    generate_answer,
    get_generation_model_from_env,
)

__all__ = [
    "MODEL_ENV_VAR",
    "SYSTEM_PROMPT",
    "TEMPERATURE",
    "Answer",
    "AnswerGenerator",
    "Citation",
    "GenerationError",
    "UnsupportedCitationError",
    "generate_answer",
    "get_generation_model_from_env",
]
