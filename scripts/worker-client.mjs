import { Worker } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

export class CPUWorker {
  constructor() {
    this.next = 0;
    this.pending = new Map();
    this.worker = new Worker(resolve('scripts/node-embedding-worker.cjs'), {workerData: {source: resolve('dist/embedding-worker.js')}});
    this.worker.on('message', message => {
      const promise = this.pending.get(message.id);
      if (!promise) return;
      this.pending.delete(message.id);
      if (message.error) promise.reject(new Error(message.error));
      else promise.resolve(message.vectors?.map(buffer => new Float32Array(buffer)));
    });
    this.worker.on('error', error => { for (const promise of this.pending.values()) promise.reject(error); this.pending.clear(); });
  }
  call(type, payload) {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.worker.postMessage({id, type, ...payload});
    });
  }
  async initialize() {
    const spec = JSON.parse(await readFile('src/model-spec.json', 'utf8'));
    const directory = resolve('.cache/models', spec.id, spec.revision);
    for(const file of spec.files) {
      const bytes=await readFile(resolve(directory,file.path));
      if(bytes.length!==file.bytes || createHash('sha256').update(bytes).digest('hex')!==file.sha256)
        throw new Error(`Benchmark model asset does not match its pinned hash: ${file.path}`);
    }
    const model = await readFile(resolve(directory, 'onnx/model_quantized.onnx'));
    const tokenizer = JSON.parse(await readFile(resolve(directory, 'tokenizer.json'), 'utf8'));
    const config = JSON.parse(await readFile(resolve(directory, 'tokenizer_config.json'), 'utf8'));
    await this.call('initialize', {model: Uint8Array.from(model).buffer, tokenizer, config});
  }
  embed(texts, query = false) { return this.call('embed', {texts, query}); }
  async dispose() { await this.worker.terminate(); }
}
