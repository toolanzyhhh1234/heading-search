import { test } from "node:test";
import assert from "node:assert/strict";
import { BM25, tokenize, reciprocalRankFusion } from "../src/core/bm25";
import { evaluate, pairedBootstrap } from "../src/core/metrics";

test("BM25 agrees with its hand-calculated score",()=>{
  const index=new BM25([{id:"1",text:"alpha alpha beta"},{id:"2",text:"beta gamma"}]);
  const expected=Math.log(1+(2-1+0.5)/(1+0.5))*2*2.2/(2+1.2*(0.25+0.75*3/2.5));
  assert.ok(Math.abs(index.search("alpha")[0].score-expected)<1e-12);
});
test("heading level changes ranking and body text remains searchable",()=>{
  const docs=[{id:"body",text:"storage design"},{id:"h6",text:"design",headings:[{level:6,text:"storage"}]},{id:"h1",text:"design",headings:[{level:1,text:"storage"}]}];
  const index=new BM25(docs,true);
  assert.deepEqual(index.search("storage").map(r=>r.id),["h1","h6","body"]);
  assert.equal(index.search("design").length,3);
});
test("incremental replacement and deletion remove old terms and document statistics",()=>{
  const index=new BM25([{id:"x",text:"old topic"},{id:"y",text:"remaining"}]);
  index.add({id:"x",text:"new subject"});assert.equal(index.search("old").length,0);
  index.remove("x");assert.equal(index.search("new").length,0);assert.equal(index.size,1);
  index.remove("missing");assert.deepEqual(index.search("remaining").map(r=>r.id),["y"]);
});
test("English stems, Unicode, IDs and stopwords use a stable analyzer",()=>{
  assert.deepEqual(tokenize("The vectors are running 東京 123"),["vector","run","東京","123"]);
});
test("RRF rewards agreement without comparing incompatible score scales",()=>{
  const result=reciprocalRankFusion([[{id:"a",score:9999},{id:"b",score:10}],[{id:"b",score:0.9}]]);
  assert.equal(result[0].id,"b");
});
test("metrics account for graded judgments and missing results",()=>{
  const exact=evaluate(["a","b"],{a:2,b:1});assert.equal(exact.ndcg10,1);assert.equal(exact.recall10,1);
  assert.equal(evaluate(["wrong","b"],{a:2,b:1}).mrr10,0.5);
  assert.equal(evaluate([],{a:1}).ndcg10,0);
  const expected=(1+2/Math.log2(3))/(2+1/Math.log2(3));
  assert.ok(Math.abs(evaluate(["b","a"],{a:2,b:1}).ndcg10-expected)<1e-12);
  assert.deepEqual(pairedBootstrap([0.1,0.1,0.1]),[0.10000000000000002,0.10000000000000002]);
});
