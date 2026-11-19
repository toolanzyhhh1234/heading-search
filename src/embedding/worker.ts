import { calibrate, openNative } from "./backends";
import type { Embedder } from "./engine";

let engine: Embedder | undefined;
// Serialize initialization and inference, including requests from fast repeated searches.
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent) => {
  const { id, type, ...payload } = event.data;
  queue = queue.then(async () => {
    try {
      if (type === "initialize-native") {
        if (engine) throw new Error("The embedding engine is already initialized.");
        const opened = await openNative(payload);
        engine = opened.engine;
        self.postMessage({ id, info: opened.info });
      } else if (type === "dispose") {
        await engine?.dispose();
        engine = undefined;
        self.postMessage({ id });
      } else if (type === "calibrate") {
        self.postMessage({ id, calibration: await calibrate(payload) });
      } else if (type === "embed") {
        if (!engine) throw new Error("The embedding model is not ready.");
        const vectors = (await engine.embed(payload.texts, payload.query)).map(vector => vector.buffer as ArrayBuffer);
        self.postMessage({ id, vectors }, { transfer: vectors });
      } else throw new Error("Unknown embedding request.");
    } catch (error) { self.postMessage({ id, error: error instanceof Error ? error.message : String(error) }); }
  });
};
