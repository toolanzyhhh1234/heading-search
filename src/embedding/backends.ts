import { createTokenizer, EmbeddingEngine, RerankEngine, scoringHead, type BatchLimits, type Embedder, type ModelConfig, type OrtModule } from "./engine";

export type NativeDevice="directml"|"native-cpu";
export type BackendKind=NativeDevice;
export interface BackendInfo {kind:BackendKind;label:string;threads:number;batchTexts:number;
  /** Set when the chosen device could not start and a compatible slower one is running instead. */
  fallback?:string}
/** How many texts DirectML embeds per call; chosen per computer at setup (see calibrateGpuBatch). */
export type GpuBatch="standard"|"large";
export interface NativeRequest {runtimeDirectory:string;modelPath:string;tokenizer:object;config:object;threads:number;device:NativeDevice;model:ModelConfig;
  /** The graph DirectML runs, when it differs from the CPU's (see scripts/patch-gpu-graph.py). Same weights, same vectors. */
  gpuModelPath?:string;gpuBatch?:GpuBatch}
export interface CalibrationRequest {runtimeDirectory:string;cpuModelPath:string;gpuModelPath:string;threads:number;tokenizer:object;config:object;model:ModelConfig}
export interface Calibration {cpuMs:number;gpuMs:number|null;gpuError?:string;
  /** DirectML on the batch-size probe with each batch tier, when the GPU was chosen. largeMs is null if that tier failed. */
  batch?:{standardMs:number;largeMs:number|null}}

// Measured on real section-length text: CPU is fastest one text at a time (batch padding is wasted work and four
// threads are already busy), while DirectML needs fixed-shape batches to amortize dispatch.
const CPU_LIMITS={tokens:512,rows:1};
/**
 * DirectML is bound by per-call overhead. Small batches suit integrated graphics (tuned on Iris Xe); a discrete GPU
 * indexes about twice as fast with large ones (RTX 3070: 14 against 27 ms per document). Padding is cheap by comparison:
 * closing batches early to save padding was slower at every threshold tried.
 */
export const GPU_BATCHES:Record<GpuBatch,{limits:BatchLimits;texts:number}>={
  standard:{limits:{tokens:2048,rows:8,fixedShapes:true},texts:16},
  large:{limits:{tokens:16384,rows:64,fixedShapes:true},texts:64},
};

/** Node's require: Obsidian enables Node integration in workers; the benchmark harness provides it explicitly. */
function loadRuntime(directory:string):OrtModule {
  const require=(globalThis as {require?:unknown}).require;
  if(typeof require!=="function")throw new Error("This Obsidian build does not allow native modules in workers.");
  return (require as (id:string)=>unknown)(`${directory}/node_modules/onnxruntime-node`) as OrtModule;
}
const cpuSession=(ort:OrtModule,model:string,threads:number)=>
  ort.InferenceSession.create(model,{executionProviders:["cpu"],intraOpNumThreads:threads,interOpNumThreads:1,graphOptimizationLevel:"all"});
// DirectML requires sequential execution without memory patterns.
const gpuSession=(ort:OrtModule,model:string)=>
  ort.InferenceSession.create(model,{executionProviders:["dml"],enableMemPattern:false,executionMode:"sequential",graphOptimizationLevel:"all"});
const cpuLabel=(threads:number)=>`Native CPU (${threads} threads)`;
const firstLine=(error:unknown)=>error instanceof Error ? error.message.split("\n")[0] : String(error);

/**
 * GPU for indexing, CPU for queries. DirectML is slow for a single row longer than 16 tokens (about 90 ms on Iris Xe,
 * against about 12 ms on the CPU), and both devices produce the same vectors from the same fp32 weights.
 */
class SplitEmbedder implements Embedder {
  constructor(private documents:EmbeddingEngine,private queries:EmbeddingEngine) {}
  embed(texts:string[],query=false):Promise<Float32Array[]> {return (query ? this.queries : this.documents).embed(texts,query);}
  async dispose():Promise<void> {await Promise.all([this.documents.dispose(),this.queries.dispose()]);}
}

/** Opens the device chosen at setup. If DirectML fails, the CPU runs the same weights, so stored vectors stay valid. */
export async function openNative(request:NativeRequest):Promise<{engine:Embedder;info:BackendInfo}> {
  const ort=loadRuntime(request.runtimeDirectory);
  const tokenizer=createTokenizer(request.tokenizer,request.config);
  const cpu=new EmbeddingEngine(ort,await cpuSession(ort,request.modelPath,request.threads),tokenizer,CPU_LIMITS,request.model);
  let fallback:string|undefined;
  if(request.device==="directml") {
    try {
      const batch=GPU_BATCHES[request.gpuBatch ?? "standard"];
      const gpu=new EmbeddingEngine(ort,await gpuSession(ort,request.gpuModelPath ?? request.modelPath),tokenizer,batch.limits,request.model);
      return {engine:new SplitEmbedder(gpu,cpu),info:{kind:"directml",label:"DirectML GPU (queries on the CPU)",threads:request.threads,batchTexts:batch.texts}};
    } catch(error) {fallback=`The GPU could not start (${firstLine(error)}).`;}
  }
  return {engine:cpu,info:{kind:"native-cpu",label:cpuLabel(request.threads),threads:request.threads,batchTexts:16,fallback}};
}

/** Section lengths in tokens: mostly short notes, some long ones, like a typical vault. */
const PROBE_LENGTHS=[24,48,64,96,128,160,192,256,320,384,448,510];
/**
 * For the batch tier: 64 sections skewed towards short ones, as vaults are. The 12 evenly spread probe lengths above
 * favour small batches (each length occurs once, so one large batch pads them all to 510 tokens), unlike real text.
 */
const BATCH_PROBE_LENGTHS=Array.from({length:64},(_,i)=>Math.round(24+486*(i/63)**2));
/** "note" is a single WordPiece token, so each text has exactly the intended length with [CLS] and [SEP]. */
const probeTexts=(lengths:number[])=>lengths.map(length=>Array(length-2).fill("note").join(" "));
/** Best of a few runs after a warm-up: embed through the real engine, so batching, shape buckets, and padding all count. */
async function timeEngine(engine:EmbeddingEngine,texts:string[],chunk=texts.length,attempts=3):Promise<number> {
  const run=async()=>{for(let i=0;i<texts.length;i+=chunk)await engine.embed(texts.slice(i,i+chunk));};
  await run();
  let best=Infinity;
  for(let attempt=0;attempt<attempts;attempt++) {const start=performance.now();await run();best=Math.min(best,performance.now()-start);}
  return best;
}

/** Compares int8 weights on the CPU with fp32 weights on DirectML, each in its fastest mode. */
export async function calibrate(request:CalibrationRequest):Promise<Calibration> {
  const ort=loadRuntime(request.runtimeDirectory),tokenizer=createTokenizer(request.tokenizer,request.config);
  const texts=probeTexts(PROBE_LENGTHS);
  const cpu=new EmbeddingEngine(ort,await cpuSession(ort,request.cpuModelPath,request.threads),tokenizer,CPU_LIMITS,request.model);
  let cpuMs:number;
  try {cpuMs=await timeEngine(cpu,texts);} finally {await cpu.dispose();}
  let gpu:EmbeddingEngine|undefined;
  try {
    gpu=new EmbeddingEngine(ort,await gpuSession(ort,request.gpuModelPath),tokenizer,GPU_BATCHES.standard.limits,request.model);
    return {cpuMs,gpuMs:await timeEngine(gpu,texts)};
  } catch(error) {return {cpuMs,gpuMs:null,gpuError:firstLine(error)};}
  finally {await gpu?.dispose().catch(()=>{});}
}

/**
 * Times DirectML with each batch tier, sending texts in the tier's own chunks as indexing does. Run only once the GPU
 * is chosen, since integrated graphics can take seconds per pass. A tier that fails (for example, out of graphics
 * memory) reports null and is never chosen.
 */
export async function calibrateGpuBatch(request:CalibrationRequest):Promise<NonNullable<Calibration["batch"]>> {
  const ort=loadRuntime(request.runtimeDirectory),tokenizer=createTokenizer(request.tokenizer,request.config);
  const texts=probeTexts(BATCH_PROBE_LENGTHS),session=await gpuSession(ort,request.gpuModelPath);
  try {
    const time=(tier:GpuBatch)=>timeEngine(new EmbeddingEngine(ort,session,tokenizer,GPU_BATCHES[tier].limits,request.model),texts,GPU_BATCHES[tier].texts,2);
    const standardMs=await time("standard");
    let largeMs:number|null=null;
    try {largeMs=await time("large");} catch {}
    return {standardMs,largeMs};
  } finally {await session.release().catch(()=>{});}
}

export interface RerankRequest {runtimeDirectory:string;modelPath:string;tokenizer:object;config:object;threads:number;device:NativeDevice;maxTokens:number;
  /** For encoder-only exports: the sentence-transformers head files (2_Dense, 3_LayerNorm, 4_Dense safetensors). */
  head?:{dense:Uint8Array;norm:Uint8Array;score:Uint8Array}}
/** Opens a cross-encoder reranker on the chosen device, falling back to the CPU like the embedding engine. */
export async function openReranker(request:RerankRequest):Promise<{reranker:RerankEngine;info:BackendInfo}> {
  const ort=loadRuntime(request.runtimeDirectory),tokenizer=createTokenizer(request.tokenizer,request.config);
  const head=request.head ? scoringHead(request.head.dense,request.head.norm,request.head.score) : undefined;
  let fallback:string|undefined;
  if(request.device==="directml") {
    try {
      return {reranker:new RerankEngine(ort,await gpuSession(ort,request.modelPath),tokenizer,GPU_BATCHES.standard.limits,request.maxTokens,head),
        info:{kind:"directml",label:"DirectML GPU",threads:1,batchTexts:32}};
    } catch(error) {fallback=`The GPU could not start (${firstLine(error)}).`;}
  }
  return {reranker:new RerankEngine(ort,await cpuSession(ort,request.modelPath,request.threads),tokenizer,CPU_LIMITS,request.maxTokens,head),
    info:{kind:"native-cpu",label:cpuLabel(request.threads),threads:request.threads,batchTexts:32,fallback}};
}
