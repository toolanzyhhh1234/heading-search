import { stemmer } from "porter-stemmer";
import type { Chunk } from "./chunker";

// Fixed English analyzer, shared by the benchmark and the shipped plugin.
const STOP = new Set(("a an and are as at be been being but by can did do does for from had has have he her hers him his how i if in into is it its itself me more most my no nor not of on once only or other our ours out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who why will with you your yours").split(" "));
export function tokenize(text: string): string[] {
  const words = text.toLowerCase().normalize("NFKC").match(/[\p{L}\p{N}]+/gu) ?? [];
  return words.filter(word => !STOP.has(word)).map(word => /^[a-z]+$/.test(word) ? stemmer(word) : word);
}

export interface LexicalDocument { id: string; text: string; title?: string; headings?: {text: string; level: number}[] }
export interface Ranked { id: string; score: number }

export class BM25 {
  private postings = new Map<string, Map<string, number>>();
  private lengths = new Map<string, number>();
  private termsByDocument = new Map<string, string[]>();
  private totalLength = 0;

  constructor(documents: LexicalDocument[] = [], private headingAware = false, private k1 = 1.2, private b = 0.75) {
    for (const document of documents) this.add(document);
  }

  add(document: LexicalDocument): void {
    this.remove(document.id);
    const frequencies = new Map<string, number>();
    const addField = (text: string, weight: number) => {
      const terms = tokenize(text);
      for (const term of terms) frequencies.set(term, (frequencies.get(term) ?? 0) + weight);
      return terms.length;
    };
    let length = addField(document.text, 1);
    length += addField(document.title ?? "", this.headingAware ? 2 : 1);
    for (const heading of document.headings ?? []) {
      // H1 has more influence than H6; the nearest heading still contributes.
      length += addField(heading.text, this.headingAware ? 1 + 2 / heading.level : 1);
    }
    this.lengths.set(document.id, length);
    this.termsByDocument.set(document.id, [...frequencies.keys()]);
    this.totalLength += length;
    for (const [term, frequency] of frequencies) {
      let posting = this.postings.get(term);
      if (!posting) this.postings.set(term, posting = new Map());
      posting.set(document.id, frequency);
    }
  }

  remove(id: string): void {
    const length = this.lengths.get(id);
    if (length === undefined) return;
    this.totalLength -= length;
    this.lengths.delete(id);
    for (const term of this.termsByDocument.get(id) ?? []) {
      const posting = this.postings.get(term)!;
      posting.delete(id);
      if (!posting.size) this.postings.delete(term);
    }
    this.termsByDocument.delete(id);
  }

  search(query: string, limit = 20): Ranked[] {
    if (!this.lengths.size) return [];
    const scores = new Map<string, number>();
    const avg = this.totalLength / this.lengths.size || 1;
    for (const term of new Set(tokenize(query))) {
      const posting = this.postings.get(term);
      if (!posting) continue;
      const idf = Math.log(1 + (this.lengths.size - posting.size + 0.5) / (posting.size + 0.5));
      for (const [id, frequency] of posting) {
        const normalization = this.k1 * (1 - this.b + this.b * this.lengths.get(id)! / avg);
        scores.set(id, (scores.get(id) ?? 0) + idf * frequency * (this.k1 + 1) / (frequency + normalization));
      }
    }
    return [...scores].map(([id, score]) => ({ id, score }))
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, limit);
  }

  get size(): number { return this.lengths.size; }
}

export function chunkDocument(chunk: Chunk): LexicalDocument {
  return { id: chunk.id, text: chunk.text, title: chunk.title, headings: chunk.headings };
}

export function reciprocalRankFusion(lists: Ranked[][], limit = 20, k = 60): Ranked[] {
  const scores = new Map<string, number>();
  for (const list of lists) list.forEach((item, rank) => scores.set(item.id, (scores.get(item.id) ?? 0) + 1 / (k + rank + 1)));
  return [...scores].map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, limit);
}
