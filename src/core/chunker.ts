import { fromMarkdown } from "mdast-util-from-markdown";
import type { Heading, PhrasingContent } from "mdast";

export interface HeadingPart { text: string; level: number }
export interface Chunk {
  id: string;
  path: string;
  title: string;
  headings: HeadingPart[];
  text: string;
  line: number;
  endLine: number;
}

function plain(nodes: PhrasingContent[]): string {
  return nodes.map(node => "value" in node ? node.value : "children" in node ? plain(node.children) : "").join("");
}

/** Preserve positions while keeping YAML keys and delimiters out of the index. */
function withoutFrontmatter(markdown: string): string {
  return markdown.replace(/^(?:\uFEFF)?---[^\S\r\n]*\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)[^\S\r\n]*(?:\r?\n|$)/,
    match => match.replace(/[^\r\n]/g, " "));
}

/** Root-level CommonMark headings define sections; fences/comments are parsed, not guessed. */
export function chunkMarkdown(path: string, markdown: string, maxChars = 1200): Chunk[] {
  if (!Number.isInteger(maxChars) || maxChars < 128) throw new Error("Chunk size must be at least 128 characters.");
  const source = withoutFrontmatter(markdown.replace(/\r\n?/g, "\n"));
  const lines = source.split("\n");
  const title = path.replace(/\.md$/i, "").split("/").pop() ?? path;
  const headings = fromMarkdown(source).children.filter((node): node is Heading => node.type === "heading");
  const result: Chunk[] = [];
  const ancestry: HeadingPart[] = [];
  let sectionStart = 0;
  let sectionAnchor = 0;

  const emit = (end: number) => {
    let text = "";
    let start = sectionStart;
    let last = sectionStart;
    const flush = () => {
      if (text.trim()) result.push({
        id: `${path}:${start}:${result.length}`, path, title,
        headings: ancestry.map(part => ({ ...part })),
        text: text.trim(), line: text.trim() ? start : sectionAnchor, endLine: last,
      });
      text = "";
    };
    for (let line = sectionStart; line < end; line++) {
      let remaining = lines[line];
      if (text && text.length + remaining.length + 1 > maxChars) flush();
      while (remaining.length > maxChars) {
        // Prefer whitespace without losing the rest of a very long paragraph.
        let cut = remaining.lastIndexOf(" ", maxChars);
        if (cut < maxChars / 2) cut = maxChars;
        // Do not split a UTF-16 surrogate pair.
        if (/^[\uDC00-\uDFFF]$/.test(remaining[cut] ?? "")) cut--;
        start = line; last = line; text = remaining.slice(0, cut); flush();
        remaining = remaining.slice(cut).trimStart();
      }
      if (!text) start = line;
      text += (text ? "\n" : "") + remaining;
      last = line;
    }
    flush();
  };

  for (const heading of headings) {
    const start = heading.position!.start.line - 1;
    emit(start);
    while (ancestry.length && ancestry[ancestry.length - 1].level >= heading.depth) ancestry.pop();
    ancestry.push({ text: plain(heading.children).trim(), level: heading.depth });
    sectionAnchor = start;
    sectionStart = heading.position!.end.line;
    const next = headings[headings.indexOf(heading) + 1];
    const end = next ? next.position!.start.line - 1 : lines.length;
    if (!lines.slice(sectionStart, end).join("\n").trim()) {
      result.push({ id: `${path}:${start}:${result.length}`, path, title,
        headings: ancestry.map(part => ({ ...part })), text: "", line: start, endLine: start });
    }
  }
  emit(lines.length);
  return result;
}

export function embeddingText(chunk: Chunk): string {
  const context = [chunk.title, ...chunk.headings.map(h => h.text)].filter(Boolean).join(" > ");
  // Bound context so a deeply nested heading cannot crowd out the paragraph.
  return `${context.slice(0, 240)}\n${chunk.text}`.trim();
}
