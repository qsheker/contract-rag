"use client";

import { describeCitation } from "@/components/citation-badges";
import type { Excerpt } from "@/lib/api";

// Why an answer is thin is a fact about retrieval, and until now it lived only
// in the server log: a question can be answered badly because the contract is
// silent, or because search handed the model five headings and no substance.
// These are different problems and they look identical from the answer alone.
export function RetrievalDetails({ excerpts }: { excerpts: Excerpt[] }) {
  if (excerpts.length === 0) {
    return null;
  }

  const found = excerpts.filter((excerpt) => excerpt.similarity !== null).length;
  const related = excerpts.length - found;

  return (
    <details className="text-xs text-muted-foreground">
      <summary className="cursor-pointer">
        Что нашёл поиск: {describeCount(found, related)}
      </summary>
      <ol className="mt-2 space-y-2">
        {excerpts.map((excerpt, index) => (
          <li key={`${excerpt.source_file}-${excerpt.clause_id}-${index}`}>
            <p className="font-mono text-[11px]">
              {describeCitation(excerpt)}
              {excerpt.similarity === null
                ? " · рядом с найденным"
                : ` · ${excerpt.similarity.toFixed(3)}`}
              {excerpt.cited ? " · процитирован" : ""}
            </p>
            <p className="mt-0.5 whitespace-pre-wrap border-l-2 pl-3">
              {excerpt.text}
            </p>
          </li>
        ))}
      </ol>
    </details>
  );
}

function describeCount(found: number, related: number): string {
  const foundLabel = `${found} ${pluralize(found, "фрагмент", "фрагмента", "фрагментов")}`;
  if (related === 0) {
    return foundLabel;
  }
  // Named separately because they got there differently: these were pulled in
  // by clause number because a hit is unreadable without them.
  return `${foundLabel} + ${related} рядом`;
}

function pluralize(
  count: number,
  one: string,
  few: string,
  many: string,
): string {
  const lastTwo = count % 100;
  const last = count % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return many;
  if (last === 1) return one;
  if (last >= 2 && last <= 4) return few;
  return many;
}
