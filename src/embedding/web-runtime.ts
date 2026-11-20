import { brotliDecompressSync } from "node:zlib";

/**
 * onnxruntime-web's JSPI WebAssembly binary (17.9 MB), Brotli-compressed and base64-encoded into main.js at build time
 * (3.8 MB). Decompressed for each engine start and handed to the worker, which cannot fetch files from the plugin.
 */
export function webRuntimeWasm():ArrayBuffer {
  const bytes=brotliDecompressSync(Buffer.from(WEB_WASM_BROTLI,"base64"));
  return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength) as ArrayBuffer;
}

type GpuAdapter={info?:{vendor?:string;architecture?:string;description?:string;isFallbackAdapter?:boolean};isFallbackAdapter?:boolean};
type Gpu={requestAdapter(options?:{powerPreference?:string}):Promise<GpuAdapter|null>};

/**
 * A cheap check before setup downloads the built-in engine's weights: the WebAssembly feature it needs (JSPI) and a
 * hardware graphics adapter. The full test runs the model (EmbeddingClient.calibrateWebGpu). Errors are sentences.
 */
export async function checkWebGpu():Promise<{ok:true;adapter:string}|{ok:false;error:string}> {
  const electron=(process.versions as Record<string,string|undefined>).electron;
  if(typeof (WebAssembly as {Suspending?:unknown}).Suspending!=="function")
    return {ok:false,error:`This Obsidian installer${electron ? ` (Electron ${electron})` : ""} is too old for it. Installing the latest version from obsidian.md enables it.`};
  const gpu=(navigator as {gpu?:Gpu}).gpu;
  if(!gpu) return {ok:false,error:"WebGPU is not available in this Obsidian window."};
  const adapter=await gpu.requestAdapter({powerPreference:"high-performance"}).catch(()=>null);
  if(!adapter) return {ok:false,error:"WebGPU found no graphics adapter on this computer."};
  if(adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter) return {ok:false,error:"WebGPU offers only a software adapter on this computer."};
  const info=adapter.info ?? {};
  return {ok:true,adapter:[info.vendor,info.architecture,info.description].filter(Boolean).join(" ") || "an unnamed adapter"};
}
