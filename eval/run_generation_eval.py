"""Measure mechanical answer quality for the mini eval set against live services.

Run manually: this needs Supabase and a running generation provider, so it is
deliberately not a pytest case. Faithfulness judged by a second model is a
separate concern and belongs to the evaluation ticket; what is counted here can
be counted without a judge.
"""

from __future__ import annotations

from dataclasses import dataclass

from run_retrieval_eval import REPORTED_K, EvalCase, load_cases

from contract_rag.generator import GenerationError, generate_answer, get_generation_model_from_env
from contract_rag.retriever import retrieve


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case_id: str
    cited_expected_clause: bool
    citation_count: int
    declined: bool
    error: str | None


def evaluate_case(case: EvalCase) -> CaseOutcome:
    hits = retrieve(case.query, k=REPORTED_K)
    try:
        answer = generate_answer(case.query, [hit.chunk for hit in hits])
    except GenerationError as error:
        return CaseOutcome(
            case_id=case.case_id,
            cited_expected_clause=False,
            citation_count=0,
            declined=False,
            error=str(error),
        )

    cited = {(citation.source_file, citation.clause_id) for citation in answer.citations}
    return CaseOutcome(
        case_id=case.case_id,
        cited_expected_clause=bool(cited & set(case.expected)),
        citation_count=len(answer.citations),
        declined=not answer.citations,
        error=None,
    )


def main() -> None:
    # Fail at start-up rather than after the first slow retrieval round trip.
    model = get_generation_model_from_env()
    cases = load_cases()
    print(f"model: {model}\n")

    # Printed as each case lands: a local 7B model takes minutes over the set,
    # and a silent run is indistinguishable from a hung one.
    print(f"{'case':<5} {'cites':<6} {'expected':<9} status", flush=True)
    outcomes: list[CaseOutcome] = []
    for case in cases:
        outcome = evaluate_case(case)
        outcomes.append(outcome)
        status = outcome.error or ("declined" if outcome.declined else "answered")
        expected = "yes" if outcome.cited_expected_clause else "no"
        print(
            f"{outcome.case_id:<5} {outcome.citation_count:<6} {expected:<9} {status}",
            flush=True,
        )

    total = len(outcomes)
    answered = sum(1 for outcome in outcomes if not outcome.declined and outcome.error is None)
    correct = sum(1 for outcome in outcomes if outcome.cited_expected_clause)
    declined = sum(1 for outcome in outcomes if outcome.declined)
    failed = sum(1 for outcome in outcomes if outcome.error is not None)

    print()
    print(f"answered with a citation: {answered}/{total}")
    print(f"cited the expected clause: {correct}/{total} = {correct / total:.2f}")
    print(f"declined (no answer in context): {declined}/{total}")
    print(f"rejected (invented citation or unusable payload): {failed}/{total}")


if __name__ == "__main__":
    main()
