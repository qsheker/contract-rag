"""Run the mini eval set end to end and write eval/results.csv.

Talks to Supabase, the generation model and the judge model, so it is a manual
script rather than a pytest case.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

# Running this file directly puts eval/ on sys.path instead of the repo root, so
# `import eval.judge` would fail. Both `python eval/run_eval.py` and
# `python -m eval.run_eval` need to work, hence the explicit bootstrap.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_rag.embeddings import create_supabase_client_from_env
from contract_rag.generator import (
    GenerationError,
    generate_answer,
    get_generation_model_from_env,
)
from contract_rag.retriever import retrieve
from eval.judge import JudgeError, get_judge_model_from_env, judge_faithfulness

DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.csv")
RETRIEVAL_K = 5
RESULT_COLUMNS = (
    "id",
    "question",
    "expected_source_file",
    "indexed",
    "retrieval_hit",
    "faithful",
    "judge_explanation",
    "latency_ms",
    "error",
)


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    language: str
    question: str
    expected_clause_id: str
    expected_source_file: str


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: EvalCase
    indexed: bool
    retrieval_hit: bool
    faithful: bool | None
    judge_explanation: str
    latency_ms: int
    error: str


def load_cases(path: Path = DATASET_PATH) -> list[EvalCase]:
    records = json.loads(path.read_text())
    return [
        EvalCase(
            case_id=record["id"],
            language=record["language"],
            question=record["question"],
            expected_clause_id=record["expected_clause_id"],
            expected_source_file=record["expected_source_file"],
        )
        for record in records
    ]


def load_indexed_source_files(supabase_client: object) -> set[str]:
    """Return the documents that actually reached the index.

    Read from the table rather than hard-coded: a document the chunker cannot
    split yet is absent here, and the moment it can be split this stops
    excluding it without anyone editing the script.
    """

    rows = supabase_client.table("contract_chunks").select("source_file").execute().data or []
    return {row["source_file"] for row in rows}


def evaluate_case(case: EvalCase, indexed_source_files: set[str]) -> CaseResult:
    indexed = case.expected_source_file in indexed_source_files

    started = perf_counter()
    hits = retrieve(case.question, k=RETRIEVAL_K)
    try:
        answer = generate_answer(case.question, [hit.chunk for hit in hits])
    except GenerationError as error:
        return CaseResult(
            case=case,
            indexed=indexed,
            retrieval_hit=_is_retrieval_hit(case, hits),
            faithful=None,
            judge_explanation="",
            latency_ms=_elapsed_ms(started),
            error=str(error),
        )
    latency_ms = _elapsed_ms(started)

    try:
        verdict = judge_faithfulness(
            case.question,
            _text_the_answer_stands_on(answer, hits),
            answer.text,
        )
    except JudgeError as error:
        return CaseResult(
            case=case,
            indexed=indexed,
            retrieval_hit=_is_retrieval_hit(case, hits),
            faithful=None,
            judge_explanation="",
            latency_ms=latency_ms,
            error=str(error),
        )

    return CaseResult(
        case=case,
        indexed=indexed,
        retrieval_hit=_is_retrieval_hit(case, hits),
        faithful=verdict.faithful,
        judge_explanation=verdict.explanation,
        latency_ms=latency_ms,
        error="",
    )


def write_results(results: list[CaseResult], path: Path = RESULTS_PATH) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_COLUMNS)
        for result in results:
            writer.writerow(
                [
                    result.case.case_id,
                    result.case.question,
                    result.case.expected_source_file,
                    result.indexed,
                    result.retrieval_hit,
                    "" if result.faithful is None else result.faithful,
                    result.judge_explanation,
                    result.latency_ms,
                    result.error,
                ]
            )


def main() -> None:
    # Both models are resolved before any work starts: a judge that matches the
    # generator must fail here, not after minutes of generation (AC-4).
    generation_model = get_generation_model_from_env()
    judge_model = get_judge_model_from_env()
    print(f"generation: {generation_model}")
    print(f"judge:      {judge_model}\n")

    cases = load_cases()
    indexed_source_files = load_indexed_source_files(create_supabase_client_from_env())

    # Printed per case: local models take minutes over the set, and a silent run
    # is indistinguishable from a hung one.
    print(f"{'case':<5} {'hit':<5} {'faithful':<9} {'ms':<7} note", flush=True)
    results: list[CaseResult] = []
    for case in cases:
        result = evaluate_case(case, indexed_source_files)
        results.append(result)
        note = result.error or ("" if result.indexed else "source not indexed")
        faithful = "-" if result.faithful is None else str(result.faithful).lower()
        print(
            f"{result.case.case_id:<5} {str(result.retrieval_hit).lower():<5} "
            f"{faithful:<9} {result.latency_ms:<7} {note[:60]}",
            flush=True,
        )

    write_results(results)
    _print_summary(results)
    print(f"\nwrote {RESULTS_PATH}")


def _print_summary(results: list[CaseResult]) -> None:
    indexed = [result for result in results if result.indexed]
    judged = [result for result in results if result.faithful is not None]
    rejected = [result for result in results if result.error]

    print()
    if indexed:
        hits = sum(1 for result in indexed if result.retrieval_hit)
        print(f"retrieval accuracy: {hits}/{len(indexed)} = {hits / len(indexed):.2f}")
    if judged:
        faithful = sum(1 for result in judged if result.faithful)
        print(f"faithfulness rate:  {faithful}/{len(judged)} = {faithful / len(judged):.2f}")
    print(f"rejected or unjudged: {len(rejected)}/{len(results)}")

    excluded = [result for result in results if not result.indexed]
    if excluded:
        # Kept out of retrieval accuracy on purpose: a document the chunker
        # cannot split is a corpus gap, not a ranking failure, and averaging the
        # two together would hide both.
        names = sorted({result.case.expected_source_file for result in excluded})
        print(f"excluded, source not indexed: {len(excluded)} case(s) from {', '.join(names)}")

    latencies = sorted(result.latency_ms for result in results)
    if latencies:
        median = latencies[len(latencies) // 2]
        print(f"median latency: {median} ms (retrieval + generation)")


def _is_retrieval_hit(case: EvalCase, hits: list) -> bool:
    return any(
        hit.chunk.source_file == case.expected_source_file
        and hit.chunk.clause_id == case.expected_clause_id
        for hit in hits
    )


def _text_the_answer_stands_on(answer: object, hits: list) -> str:
    """Return the excerpts the judge should hold the answer against.

    Normally that is what the answer cited. A declined answer cites nothing, so
    the judge sees everything retrieved and can tell a correct refusal from a
    lazy one.
    """

    cited = {(citation.source_file, citation.clause_id) for citation in answer.citations}
    relevant = [hit for hit in hits if (hit.chunk.source_file, hit.chunk.clause_id) in cited]
    chosen = relevant or hits
    return "\n\n".join(hit.chunk.text for hit in chosen) or "(no excerpts were retrieved)"


def _elapsed_ms(started: float) -> int:
    return int((perf_counter() - started) * 1000)


if __name__ == "__main__":
    main()
