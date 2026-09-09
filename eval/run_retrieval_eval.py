"""Measure retrieval recall@k for the mini eval set against the live index.

Run manually: this talks to Supabase, so it is deliberately not a pytest case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contract_rag.retriever import RetrievedChunk, retrieve

EVAL_PATH = Path(__file__).with_name("retrieval_mini.jsonl")
REPORTED_K = 5
# Recall at a smaller k shows how much of the win depends on the tail of top-5.
INSPECTED_CUTOFFS = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    language: str
    query: str
    expected: tuple[tuple[str, str], ...]


def load_cases(path: Path = EVAL_PATH) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        cases.append(
            EvalCase(
                case_id=record["id"],
                language=record["language"],
                query=record["query"],
                expected=tuple(
                    (item["source_file"], item["clause_id"]) for item in record["expected"]
                ),
            )
        )
    return cases


def first_hit_rank(case: EvalCase, hits: list[RetrievedChunk]) -> int | None:
    """1-based rank of the first expected chunk, or None when it never appears."""

    for rank, hit in enumerate(hits, start=1):
        if (hit.chunk.source_file, hit.chunk.clause_id) in case.expected:
            return rank
    return None


def main() -> None:
    cases = load_cases()
    ranks: dict[str, int | None] = {}

    print(f"{'case':<5} {'lang':<5} {'rank':<5} query")
    for case in cases:
        hits = retrieve(case.query, k=REPORTED_K)
        rank = first_hit_rank(case, hits)
        ranks[case.case_id] = rank
        shown_rank = str(rank) if rank else "miss"
        print(f"{case.case_id:<5} {case.language:<5} {shown_rank:<5} {case.query}")

    print()
    for cutoff in INSPECTED_CUTOFFS:
        found = sum(1 for rank in ranks.values() if rank is not None and rank <= cutoff)
        print(f"recall@{cutoff}: {found}/{len(cases)} = {found / len(cases):.2f}")


if __name__ == "__main__":
    main()
