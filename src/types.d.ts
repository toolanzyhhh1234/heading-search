declare module "porter-stemmer" {
  export function stemmer(word: string): string;
}
declare const EMBEDDING_WORKER_SOURCE: string;
/** The built-in WebGPU engine's module worker (src/embedding/web-worker.ts), built by scripts/build.mjs. */
declare const WEB_WORKER_SOURCE: string;
/** onnxruntime-web's JSPI WebAssembly binary, Brotli-compressed and base64-encoded (see web-runtime.ts). */
declare const WEB_WASM_BROTLI: string;
/** Bundled model assets (esbuild's binary loader), such as the DirectML graph from scripts/patch-gpu-graph.py. */
declare module "*.onnx" {
  const data: Uint8Array;
  export default data;
}
