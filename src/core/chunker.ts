import { fromMarkdown } from "mdast-util-from-markdown";
import type { Heading, PhrasingContent, Nodes } from "mdast";

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
  return nodes.map(node => node.type === "html" ? "" : "value" in node ? node.value : "children" in node ? plain(node.children) : "alt" in node ? node.alt ?? "" : "").join("");
}

/** Preserve positions while keeping YAML keys and delimiters out of the index. */
function withoutFrontmatter(markdown: string): string {
  return markdown.replace(/^(?:\uFEFF)?---[^\S\r\n]*\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)[^\S\r\n]*(?:\r?\n|$)/,
    match => match.replace(/[^\r\n]/g, " "));
}

function withoutObsidianComments(source:string):string {
  if(!source.includes("%%"))return source;
  const codeRanges:[number,number][]=[];
  const collect=(node:Nodes):void=>{
    if(node.type==="code" || node.type==="inlineCode" || (node.type==="html" && node.value.startsWith("<!--")))codeRanges.push([node.position!.start.offset!,node.position!.end.offset!]);
    else if("children" in node)for(const child of node.children)collect(child);
  };
  collect(fromMarkdown(source));
  const comments:[number,number][]=[];
  let start=-1;
  for(const marker of source.matchAll(/%%/g)) {
    const offset=marker.index!;
    if(start>=0) {comments.push([start,offset+2]);start=-1;continue;}
    if(codeRanges.some(([a,b])=>offset>=a && offset<b))continue;
    let escapes=0;
    for(let i=offset-1;i>=0 && source[i]==="\\";i--)escapes++;
    if(escapes%2===0)start=offset;
  }
  if(start>=0)comments.push([start,source.length]);
  let result="",cursor=0;
  for(const [a,b] of comments) {result+=source.slice(cursor,a)+source.slice(a,b).replace(/[^\n]/g," ");cursor=b;}
  return result+source.slice(cursor);
}

/** Root-level CommonMark headings define sections; fences/comments are parsed, not guessed. */
export function chunkMarkdown(path: string, markdown: string, maxChars = 1200): Chunk[] {
  if (!Number.isInteger(maxChars) || maxChars < 128) throw new Error("Chunk size must be at least 128 characters.");
  const source = withoutObsidianComments(withoutFrontmatter(markdown.replace(/\r\n?/g, "\n")));
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
  // Retain both the note title and nearest heading when the ancestry is long.
  const bounded=context.length<=240 ? context : `${context.slice(0,80)} … ${context.slice(-157)}`;
  return `${bounded}\n${chunk.text}`.trim();
}
