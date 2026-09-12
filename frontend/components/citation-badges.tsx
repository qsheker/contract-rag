"use client";

import { Badge } from "@/components/ui/badge";
import type { Citation, Excerpt } from "@/lib/api";

// A citation the user cannot check is worthless, so the badge always names the
// document and the clause - and opens into the clause's own words, which is
// what checking an answer actually requires. The page is shown when the source
// format has one: DOCX and TXT store no page boundaries, and saying so beats
// leaving a gap the reader has to interpret.
export function CitationBadges({
  citations,
  excerpts,
}: {
  citations: Citation[];
  excerpts: Excerpt[];
}) {
  if (citations.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Ответ не опирается ни на один пункт договора.
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      {citations.map((citation, index) => (
        <CitationBadge
          key={`${citation.source_file}-${citation.clause_id}-${citation.page_number}-${index}`}
          citation={citation}
          text={findExcerptText(citation, excerpts)}
        />
      ))}
    </div>
  );
}

function CitationBadge({
  citation,
  text,
}: {
  citation: Citation;
  text: string | null;
}) {
  const label = describeCitation(citation);

  // Without the text there is nothing to open, which happens only for an answer
  // that predates this field - never for one the backend just produced.
  if (text === null) {
    return (
      <Badge variant="outline" className="h-auto py-1 font-mono text-[11px]">
        {label}
      </Badge>
    );
  }

  return (
    <details className="group">
      <summary className="cursor-pointer list-none">
        <Badge
          variant="outline"
          className="h-auto py-1 font-mono text-[11px] group-hover:bg-accent"
        >
          {label}
          <span className="ml-1 text-muted-foreground group-open:hidden">▸</span>
          <span className="ml-1 hidden text-muted-foreground group-open:inline">
            ▾
          </span>
        </Badge>
      </summary>
      <blockquote className="mt-1.5 whitespace-pre-wrap border-l-2 pl-3 text-xs text-muted-foreground">
        {text}
      </blockquote>
    </details>
  );
}

// The same triple the backend verifies citations against, so a citation that
// passed that check always finds its text here.
function findExcerptText(citation: Citation, excerpts: Excerpt[]): string | null {
  const match = excerpts.find(
    (excerpt) =>
      excerpt.source_file === citation.source_file &&
      excerpt.clause_id === citation.clause_id &&
      excerpt.page_number === citation.page_number,
  );
  return match?.text ?? null;
}

export function describeCitation(citation: Citation): string {
  const parts = [citation.source_file];
  parts.push(citation.clause_id ? `п. ${citation.clause_id}` : "преамбула");
  parts.push(
    citation.page_number === null
      ? "без страницы"
      : `стр. ${citation.page_number}`,
  );
  return parts.join(" · ");
}
