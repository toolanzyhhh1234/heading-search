import { EmbeddingEngine } from "./engine";
const engine = new EmbeddingEngine();
// Serialize initialization and inference, including requests from fast repeated searches.
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent) => {
  const { id, type, ...payload } = event.data;
  queue = queue.then(async () => {
    try {
      if (type === "initialize") {
        await engine.initialize(new Uint8Array(payload.model), payload.tokenizer, payload.config);
        self.postMessage({ id, ok: true });
      } else if (type === "embed") {
        const vectors: ArrayBuffer[] = [];
        for (const text of payload.texts) vectors.push((await engine.embed(text, payload.query)).buffer as ArrayBuffer);
        self.postMessage({ id, vectors }, { transfer: vectors });
      } else throw new Error("Unknown embedding request.");
    } catch (error) { self.postMessage({ id, error: error instanceof Error ? error.message : String(error) }); }
  });
};
