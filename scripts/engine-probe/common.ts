import { EmbeddingEngine, createTokenizer, type BatchLimits, type ModelConfig, type OrtModule, type OrtSession } from "../../src/embedding/engine";

export interface Job {tokenizer:object;config:object;model:ModelConfig;limits:BatchLimits;chunk:number;docs:string[];queries:string[]}
export interface BenchResult {firstPassMsPerDoc:number;msPerDoc:number;queryMs:number;vectors:Float32Array[]}

/** Embeds the documents in the plugin's indexing chunks (first pass includes GPU shader compilation), then single queries. */
export async function bench(ort:OrtModule,session:OrtSession,job:Job):Promise<BenchResult> {
  const tokenizer=createTokenizer(job.tokenizer,job.config);
  const engine=new EmbeddingEngine(ort,session,tokenizer,job.limits,job.model);
  const queries=new EmbeddingEngine(ort,session,tokenizer,{tokens:512,rows:1},job.model);
  const embedAll=async()=>{const out:Float32Array[]=[];for(let i=0;i<job.docs.length;i+=job.chunk)out.push(...await engine.embed(job.docs.slice(i,i+job.chunk)));return out;};
  let start=performance.now();
  let vectors=await embedAll();
  const firstMs=performance.now()-start;
  let best=Infinity;
  for(let attempt=0;attempt<2;attempt++) {start=performance.now();vectors=await embedAll();best=Math.min(best,performance.now()-start);}
  await queries.embed(job.queries.slice(0,1),true);
  start=performance.now();
  const queryVectors:Float32Array[]=[];
  for(const text of job.queries) queryVectors.push((await queries.embed([text],true))[0]);
  const queryMs=(performance.now()-start)/job.queries.length;
  return {firstPassMsPerDoc:firstMs/job.docs.length,msPerDoc:best/job.docs.length,queryMs,vectors:[...vectors,...queryVectors]};
}

export const describeError=(error:unknown)=>error instanceof Error ? `${error.message}`.split("\n").slice(0,3).join(" | ") : String(error);
