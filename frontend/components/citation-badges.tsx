import { Badge } from "@/components/ui/badge";
import type { Citation } from "@/lib/api";

// A citation the user cannot check is worthless, so the badge always names the
// document and the clause. The page is shown when the source format has one -
// DOCX and TXT store no page boundaries, and saying so beats leaving a gap the
// reader has to interpret.
export function CitationBadges({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Ответ не опирается ни на один пункт договора.
      </p>
    );
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {citations.map((citation, index) => (
        <Badge
          key={`${citation.source_file}-${citation.clause_id}-${citation.page_number}-${index}`}
          variant="outline"
          className="h-auto py-1 font-mono text-[11px]"
        >
          {describeCitation(citation)}
        </Badge>
      ))}
    </div>
  );
}

function describeCitation(citation: Citation): string {
  const parts = [citation.source_file];
  parts.push(citation.clause_id ? `п. ${citation.clause_id}` : "преамбула");
  parts.push(
    citation.page_number === null
      ? "без страницы"
      : `стр. ${citation.page_number}`,
  );
  return parts.join(" · ");
}
