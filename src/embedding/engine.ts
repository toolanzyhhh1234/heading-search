import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/wasm";
import wasmBase64 from "onnxruntime-web/dist/ort-wasm-simd.wasm?base64";
import spec from "../model-spec.json";

/** The WASM runtime is shipped inside main.js; inference never fetches code or models. */
export class EmbeddingEngine {
  private session?: ort.InferenceSession;
  private tokenizer?: Tokenizer;

  async initialize(model: Uint8Array, tokenizer: object, config: object): Promise<void> {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = { "ort-wasm-simd.wasm": `data:application/wasm;base64,${wasmBase64}` };
    this.tokenizer = new Tokenizer(tokenizer, config);
    this.session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });
  }

  async embed(text: string, query = false): Promise<Float32Array> {
    if (!this.session || !this.tokenizer) throw new Error("The embedding model is not ready.");
    const tokens = this.tokenizer.encode((query ? spec.queryPrefix : "") + text).ids;
    // BERT truncation retains CLS and the closing SEP token.
    const ids = tokens.length <= spec.maxTokens ? tokens : [...tokens.slice(0, spec.maxTokens - 1), tokens[tokens.length - 1]];
    const dimensions = [1, ids.length];
    const feeds: Record<string, ort.Tensor> = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), dimensions),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(ids.length).fill(1n), dimensions),
    };
    if (this.session.inputNames.includes("token_type_ids"))
      feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(ids.length), dimensions);
    const outputs = await this.session.run(feeds);
    const tensor = outputs[this.session.outputNames[0]];
    const data = tensor.data as Float32Array;
    // BGE uses CLS pooling, not MiniLM's mean pooling.
    const vector = Float32Array.from(data.subarray(0, spec.dimensions));
    let norm = 0;
    for (const value of vector) norm += value * value;
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < vector.length; i++) vector[i] /= norm;
    for (const output of Object.values(outputs)) output.dispose();
    return vector;
  }

  async dispose(): Promise<void> { await this.session?.release(); this.session = undefined; }
}
