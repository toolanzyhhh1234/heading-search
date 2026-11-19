declare module "porter-stemmer" {
  export function stemmer(word: string): string;
}
declare const EMBEDDING_WORKER_SOURCE: string;
/** Bundled model assets (esbuild's binary loader), such as the DirectML graph from scripts/patch-gpu-graph.py. */
declare module "*.onnx" {
  const data: Uint8Array;
  export default data;
}
