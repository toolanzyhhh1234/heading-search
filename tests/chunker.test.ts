import { test } from "node:test";
import assert from "node:assert/strict";
import { chunkMarkdown, embeddingText } from "../src/core/chunker";

test("nested and skipped heading levels keep their ancestors without mixing sections",()=>{
  const chunks=chunkMarkdown("Notes/Guide.md","Preamble.\n# Parent\nParent body\n### Child\nChild body\n## Sibling\nSibling body\n# Other\nOther body");
  assert.deepEqual(chunks.map(c=>c.headings.map(h=>[h.level,h.text])),[[],[[1,"Parent"]],[[1,"Parent"],[3,"Child"]],[[1,"Parent"],[2,"Sibling"]],[[1,"Other"]]]);
  assert.equal(chunks[2].line,4);assert.equal(chunks[3].text,"Sibling body");
  assert.equal(embeddingText(chunks[2]),"Guide > Parent > Child\nChild body");
});
test("CommonMark ignores headings inside fenced code, comments and quotes",()=>{
  const md="# Real\n~~~md\n# code\n~~~\n<!--\n# comment\n-->\n> # quote\n\n## Next\nText";
  const chunks=chunkMarkdown("Note.md",md);
  assert.deepEqual(chunks.map(c=>c.headings.at(-1)?.text),["Real","Next"]);
  assert.match(chunks[0].text,/# code/);
});
test("setext headings, formatted text, duplicate headings, CRLF and frontmatter retain source lines",()=>{
  const md="---\r\nsecret: hidden\r\n---\r\n**Overview**\r\n====\r\nFirst\r\n\r\nDetails\r\n---\r\nSecond\r\n## Details\r\nThird";
  const chunks=chunkMarkdown("Note.md",md);
  assert.deepEqual(chunks.map(c=>c.headings.at(-1)?.text),["Overview","Details","Details"]);
  assert.equal(chunks[0].line,5);assert.equal(chunks[1].line,9);assert.equal(chunks[2].line,11);
  assert.ok(chunks.every(c=>!c.text.includes("secret")));assert.equal(new Set(chunks.map(c=>c.id)).size,chunks.length);
});
test("long lines are split without dropping words or crossing a heading",()=>{
  const body=Array.from({length:500},(_,i)=>`word${i}`).join(" ");
  const chunks=chunkMarkdown("N.md",`# One\n${body}\n# Two\nTail`,256);
  const first=chunks.filter(c=>c.headings[0].text==="One");
  assert.equal(first.map(c=>c.text).join(" "),body);assert.ok(chunks.every(c=>c.text.length<=256));
  assert.equal(chunks.at(-1)?.text,"Tail");
});
test("empty headings remain searchable, empty notes produce no chunk",()=>{
  assert.equal(chunkMarkdown("N.md","").length,0);
  assert.deepEqual(chunkMarkdown("N.md","# Empty\n## Also empty").map(c=>[c.text,c.line]),[["",0],["",1]]);
});
test("heading labels retain image alt text and discard HTML tags",()=>{
  const [chunk]=chunkMarkdown("Guide.md","# <span>Install</span> ![steps](image.png)\nInstructions");
  assert.equal(chunk.headings[0].text,"Install steps");
});
test("long embedding context keeps the nearest heading and note title",()=>{
  const chunks=chunkMarkdown("My note.md",`# ${"Parent ".repeat(80)}\n## Specific leaf\nUseful paragraph`);
  const text=embeddingText(chunks.at(-1)!);
  assert.ok(text.startsWith("My note >"));assert.match(text,/Specific leaf\nUseful paragraph$/);
  assert.ok(text.split("\n")[0].length<=240);
});
test("Obsidian comments hide headings and text, while code literals remain intact",()=>{
  const md="# Visible\nFirst\n%%\n# Hidden\nSecret words\n%%\n```md\n%% literal %%\n# Code\n```\n## Next\nLast";
  const chunks=chunkMarkdown("N.md",md);
  assert.deepEqual(chunks.map(c=>c.headings.at(-1)?.text),["Visible","Next"]);
  assert.ok(chunks.every(c=>!c.text.includes("Secret words")));assert.match(chunks[0].text,/%% literal %%/);
  assert.equal(chunks[1].line,11);
  assert.equal(chunkMarkdown("N.md","# One\n<!-- %% literal -->\n## Two\nBody").at(-1)?.headings.at(-1)?.text,"Two");
});
