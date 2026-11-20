// The built-in engine: onnxruntime-web's JSPI build with its native WebGPU execution provider, bundled into main.js
// (scripts/build.mjs). Nothing is downloaded or installed for it; the model weights come from the shared model cache.
import * as ort from "onnxruntime-web/jspi";
import { createTokenizer, EmbeddingEngine, type Embedder, type ModelConfig, type OrtModule, type OrtSession } from "./engine";
import { GPU_BATCHES, PROBE_LENGTHS, probeTexts, timeEngine, type BackendInfo } from "./backends";

export interface WebGpuRequest {
  /** The WebAssembly binary, decompressed by the window (see web-runtime.ts). */
  wasm:ArrayBuffer;
  /** The ONNX graph and its external-data files, by the name the graph refers to them with. */
  graph:ArrayBuffer;externalData:{path:string;data:ArrayBuffer}[];
  tokenizer:object;config:object;model:ModelConfig;
}

/**
 * Small fixed-shape batches, as for DirectML. On WebGPU they beat large ones on both computers measured
 * (RTX 3070: 9.5 against 12.1 ms per document; Iris Xe: 290 against 450 with fp32 weights).
 */
const BATCH=GPU_BATCHES.standard;
export const WEBGPU_INFO:BackendInfo={kind:"webgpu",label:"Built-in GPU engine (WebGPU)",threads:1,batchTexts:BATCH.texts};

let configured=false;
function configure(wasm:ArrayBuffer):void {
  if(configured) return;
  ort.env.wasm.wasmBinary=wasm;
  // Obsidian is not cross-origin isolated, and threaded WebAssembly hangs there instead of falling back.
  ort.env.wasm.numThreads=1;
  ort.env.logLevel="error";
  try {ort.env.webgpu.powerPreference="high-performance";} catch {}
  configured=true;
}

async function open(request:WebGpuRequest):Promise<EmbeddingEngine> {
  configure(request.wasm);
  const session=await ort.InferenceSession.create(new Uint8Array(request.graph),{executionProviders:["webgpu"],graphOptimizationLevel:"all",
    externalData:request.externalData.map(file=>({path:file.path,data:new Uint8Array(file.data)}))});
  return new EmbeddingEngine(ort as unknown as OrtModule,session as unknown as OrtSession,createTokenizer(request.tokenizer,request.config),BATCH.limits,request.model);
}

/**
 * Setup's test: the vectors must be finite (half-precision arithmetic can overflow), then the probe sections are timed
 * the way indexing sends them.
 */
async function calibrate(request:WebGpuRequest):Promise<{ms:number}> {
  const engine=await open(request);
  try {
    const texts=probeTexts(PROBE_LENGTHS);
    const vectors=[...await engine.embed(texts),...await engine.embed(["a short test query"],true)];
    if(vectors.some(vector=>vector.length!==request.model.dimensions || vector.some(value=>!Number.isFinite(value))))
      throw new Error("the GPU returned invalid vectors");
    return {ms:await timeEngine(engine,texts)};
  } finally {await engine.dispose().catch(()=>{});}
}

let engine:Embedder|undefined;
// Serialize initialization and inference, including requests from fast repeated searches.
let queue=Promise.resolve();
self.onmessage=(event:MessageEvent)=>{
  const {id,type,...payload}=event.data;
  queue=queue.then(async()=>{
    try {
      if(type==="initialize-webgpu") {
        if(engine) throw new Error("The embedding engine is already initialized.");
        engine=await open(payload as WebGpuRequest);
        self.postMessage({id,info:WEBGPU_INFO});
      } else if(type==="calibrate-webgpu") {
        self.postMessage({id,calibration:await calibrate(payload as WebGpuRequest)});
      } else if(type==="dispose") {
        await engine?.dispose();
        engine=undefined;
        self.postMessage({id});
      } else if(type==="embed") {
        if(!engine) throw new Error("The embedding model is not ready.");
        const vectors=(await engine.embed(payload.texts,payload.query)).map(vector=>vector.buffer as ArrayBuffer);
        self.postMessage({id,vectors},{transfer:vectors});
      } else throw new Error("Unknown embedding request.");
    } catch(error) {self.postMessage({id,error:error instanceof Error ? error.message : String(error)});}
  });
};
