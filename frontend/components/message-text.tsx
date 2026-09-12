import { Fragment, type ReactNode } from "react";

// The model writes lists and **bold** because that is how an answer of eight
// duties reads best. Rendered as one pre-wrapped blob it arrived as a wall of
// text with literal asterisks in it. This covers what the answers actually
// contain - paragraphs, bulleted and numbered items, bold runs - rather than
// pulling in a full Markdown parser for three constructs.

const BULLET = /^\s*[-*•]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;

type Block =
  | { kind: "paragraph"; lines: string[] }
  | { kind: "list"; ordered: boolean; items: string[] };

export function MessageText({ text }: { text: string }) {
  return (
    <div className="space-y-3">
      {toBlocks(text).map((block, index) =>
        block.kind === "list" ? (
          <List key={index} block={block} />
        ) : (
          <p key={index} className="whitespace-pre-wrap">
            {block.lines.map((line, lineIndex) => (
              <Fragment key={lineIndex}>
                {lineIndex > 0 && <br />}
                {withBold(line)}
              </Fragment>
            ))}
          </p>
        ),
      )}
    </div>
  );
}

function List({ block }: { block: Extract<Block, { kind: "list" }> }) {
  const className = "ml-5 space-y-1.5 " + (block.ordered ? "list-decimal" : "list-disc");
  const items = block.items.map((item, index) => (
    <li key={index} className="pl-1">
      {withBold(item)}
    </li>
  ));
  return block.ordered ? <ol className={className}>{items}</ol> : <ul className={className}>{items}</ul>;
}

function toBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) {
      // A blank line ends whatever was open, which is what separates the
      // model's paragraphs from each other.
      blocks.push({ kind: "paragraph", lines: [] });
      continue;
    }

    const bullet = BULLET.exec(trimmed);
    const numbered = bullet ? null : NUMBERED.exec(trimmed);
    if (bullet || numbered) {
      const ordered = numbered !== null;
      const item = trimmed.slice((bullet ?? numbered)![0].length);
      const open = blocks.at(-1);
      if (open?.kind === "list" && open.ordered === ordered) {
        open.items.push(item);
      } else {
        blocks.push({ kind: "list", ordered, items: [item] });
      }
      continue;
    }

    const open = blocks.at(-1);
    if (open?.kind === "paragraph" && open.lines.length > 0) {
      open.lines.push(trimmed);
    } else {
      blocks.push({ kind: "paragraph", lines: [trimmed] });
    }
  }

  return blocks.filter(
    (block) => block.kind === "list" || block.lines.length > 0,
  );
}

function withBold(text: string): ReactNode {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={index} className="font-semibold">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <Fragment key={index}>{part}</Fragment>
    ),
  );
}
