import { test } from "node:test";
import assert from "node:assert/strict";
import { EmbeddingEngine, RerankEngine, applyHead, parseSafetensors, scoringHead, type ModelConfig, type OrtModule, type OrtSession, type OrtTensor } from "../src/embedding/engine";
import type { Tokenizer } from "@huggingface/tokenizers";

class Tensor implements OrtTensor {
  constructor(readonly type:string,readonly data:BigInt64Array|Float32Array,readonly dims:number[]) {}
}
const ort={Tensor,InferenceSession:{create:async()=>{throw new Error("unused");}}} as unknown as OrtModule;
/** Hidden state of token t in a row: [id, position, 1]. Padding (mask 0) gets garbage, which pooling must ignore. */
function session(calls:number[][]=[]):OrtSession {
  return {inputNames:["input_ids","attention_mask"],outputNames:["last_hidden_state"],release:async()=>{},
    run:async feeds=>{
      const ids=feeds.input_ids.data as BigInt64Array,mask=feeds.attention_mask.data as BigInt64Array,[rows,width]=feeds.input_ids.dims;
      calls.push([rows,width]);
      const hidden=new Float32Array(rows*width*3);
      for(let i=0;i<rows*width;i++)hidden.set(mask[i] ? [Number(ids[i]),i%width,1] : [999,999,999],i*3);
      return {last_hidden_state:new Tensor("float32",hidden,[rows,width,3])};
    }};
}
/** Each character becomes one token, wrapped in [CLS]=1 … [SEP]=2. */
const tokenizer={encode:(text:string)=>({ids:[1,...[...text].map(c=>c.charCodeAt(0)),2]})} as unknown as Tokenizer;
const config=(pooling:"cls"|"mean",maxTokens=512):ModelConfig=>({dimensions:3,maxTokens,pooling,queryPrefix:"q:",documentPrefix:""});
const unit=(v:number[])=>{const n=Math.hypot(...v);return v.map(x=>x/n);};
const close=(actual:Float32Array,expected:number[])=>assert.ok(actual.every((x,i)=>Math.abs(x-expected[i])<1e-6),`${[...actual]} != ${expected}`);

test("CLS pooling takes the first token, independent of batch padding",async()=>{
  const engine=new EmbeddingEngine(ort,session(),tokenizer,{tokens:4096,rows:8},config("cls"));
  const [short,long]=await engine.embed(["a","much longer text"]);
  close(short,unit([1,0,1]));close(long,unit([1,0,1]));
});

test("mean pooling averages only the row's real tokens, so batching does not change vectors",async()=>{
  const engine=new EmbeddingEngine(ort,session(),tokenizer,{tokens:4096,rows:8,fixedShapes:true},config("mean"));
  const batched=await engine.embed(["ab","a much longer text"]);
  const single=await engine.embed(["ab"]);
  // [CLS] a b [SEP] → ids 1,97,98,2 at positions 0..3
  close(batched[0],unit([(1+97+98+2)/4,(0+1+2+3)/4,1]));
  close(single[0],[...batched[0]]);
});

test("queries get the model's prefix and truncation keeps the closing token",async()=>{
  const calls:number[][]=[];
  const engine=new EmbeddingEngine(ort,session(calls),tokenizer,{tokens:512,rows:1},config("mean",4));
  assert.deepEqual(engine.encode("xyz",true),[1,"q".charCodeAt(0),":".charCodeAt(0),2]);
  assert.deepEqual(engine.encode("xyz",false),[1,"x".charCodeAt(0),"y".charCodeAt(0),2]);
  await engine.embed(["one","two","three"]);
  assert.deepEqual(calls,[[1,4],[1,4],[1,4]]);
});

test("fixed shapes round batches to bucketed widths and power-of-two rows",async()=>{
  const calls:number[][]=[];
  const engine=new EmbeddingEngine(ort,session(calls),tokenizer,{tokens:2048,rows:8,fixedShapes:true},config("cls"));
  await engine.embed(["a","bb","ccc"]);
  assert.deepEqual(calls,[[4,32]]);
});

function safetensors(tensors:Record<string,{shape:number[];data:number[]}>):Uint8Array {
  let offset=0;const header:Record<string,unknown>={},parts:Buffer[]=[];
  for(const [name,t] of Object.entries(tensors)) {
    const bytes=Buffer.from(new Float32Array(t.data).buffer);
    header[name]={dtype:"F32",shape:t.shape,data_offsets:[offset,offset+bytes.length]};offset+=bytes.length;parts.push(bytes);
  }
  const json=Buffer.from(JSON.stringify(header)),length=Buffer.alloc(8);length.writeBigUInt64LE(BigInt(json.length));
  // Return a view inside a larger pool, as Node's readFile can, to catch reads of the wrong backing buffer.
  const pool=Buffer.concat([Buffer.alloc(13,7),length,json,...parts]);
  return pool.subarray(13);
}

test("safetensors parsing copies each tensor out of a pooled Buffer",()=>{
  const parsed=parseSafetensors(safetensors({a:{shape:[2],data:[1.5,-2]},b:{shape:[1],data:[3]}}));
  assert.deepEqual([...parsed.a.data],[1.5,-2]);assert.deepEqual([...parsed.b.data],[3]);assert.deepEqual(parsed.a.shape,[2]);
});

test("the cross-encoder head applies Dense+GELU, LayerNorm, and the score layer",()=>{
  const head=scoringHead(
    safetensors({"linear.weight":{shape:[2,2],data:[1,0,0,1]}}),
    safetensors({"norm.weight":{shape:[2],data:[1,1]},"norm.bias":{shape:[2],data:[0,0]}}),
    safetensors({"linear.weight":{shape:[1,2],data:[2,-1]},"linear.bias":{shape:[1],data:[0.5]}}));
  // GELU(3)≈2.99595, GELU(-1)≈-0.15866; LayerNorm of two values maps them to ±1 (minus the epsilon effect).
  const score=applyHead(head,Float32Array.from([3,-1]));
  assert.ok(Math.abs(score-(0.5+2*1-1*-1))<1e-3,String(score));
});

test("pair encoding shortens only the passage and keeps the whole query",()=>{
  const pairTokenizer={encode:(text:string,options?:{text_pair?:string})=>{
    const ids=[1,...[...text].map(c=>c.charCodeAt(0)),2,...[...(options?.text_pair??"")].map(c=>c.charCodeAt(0)),2];
    return {ids,token_type_ids:ids.map((_,i)=>i<=text.length+1 ? 0 : 1)};
  }} as unknown as Tokenizer;
  const reranker=new RerankEngine(ort,session(),pairTokenizer,{tokens:512,rows:1},16);
  const {ids,types}=reranker.encodePair("query","a long passage that will not fit");
  assert.ok(ids.length<=16);
  assert.deepEqual(ids.slice(0,7),[1,...[..."query"].map(c=>c.charCodeAt(0)),2]);
  assert.equal(ids.at(-1),2);assert.equal(types[0],0);assert.equal(types.at(-1),1);
});
