import { Worker } from 'node:worker_threads';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

/** The plugin's embedding worker, running the native engine from runtimeDirectory (default: .cache/runtime-dev). */
export class CPUWorker {
  constructor({runtimeDirectory = '.cache/runtime-dev', weights = 'int8', threads = 4, device = weights === 'fp32' ? 'directml' : 'native-cpu', spec = 'src/model-spec.json'} = {}) {
    this.options = {runtimeDirectory, weights, threads, device, spec};
    this.next = 0;
    this.pending = new Map();
    this.worker = new Worker(resolve('scripts/node-embedding-worker.cjs'), {workerData: {source: resolve('dist/embedding-worker.js')}});
    this.worker.on('message', message => {
      const promise = this.pending.get(message.id);
      if (!promise) return;
      this.pending.delete(message.id);
      if (message.error) promise.reject(new Error(message.error));
      else promise.resolve(message.info ?? message.calibration ?? message.vectors?.map(buffer => new Float32Array(buffer)));
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
    const spec = JSON.parse(await readFile(this.options.spec, 'utf8'));
    const directory = resolve('.cache/models', spec.id, spec.revision);
    const weights = spec.weights[this.options.weights];
    for(const file of [...spec.files, weights, ...(weights.extra ?? [])]) {
      const bytes=await readFile(resolve(directory,file.path));
      if(bytes.length!==file.bytes || createHash('sha256').update(bytes).digest('hex')!==file.sha256)
        throw new Error(`Benchmark model asset does not match its pinned hash: ${file.path}`);
    }
    const tokenizer = JSON.parse(await readFile(resolve(directory, 'tokenizer.json'), 'utf8'));
    const config = JSON.parse(await readFile(resolve(directory, 'tokenizer_config.json'), 'utf8'));
    const {runtimeDirectory, threads, device} = this.options;
    const model = {dimensions: spec.dimensions, maxTokens: spec.maxTokens, pooling: spec.pooling, queryPrefix: spec.queryPrefix, documentPrefix: spec.documentPrefix ?? ''};
    this.info = await this.call('initialize-native', {runtimeDirectory: resolve(runtimeDirectory).replaceAll('\\', '/'),
      modelPath: resolve(directory, weights.path), tokenizer, config, threads, device, model});
    return this.info;
  }
  embed(texts, query = false) { return this.call('embed', {texts, query}); }
  async dispose() { await this.worker.terminate(); }
}
