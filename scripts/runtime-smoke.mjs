import { buildWorker } from './build.mjs';
import { mkdir, writeFile } from 'node:fs/promises';
import { CPUWorker } from './worker-client.mjs';
await mkdir('dist', {recursive:true});
await writeFile('dist/embedding-worker.js', await buildWorker());
const worker = new CPUWorker();
try {
  console.time('model initialization'); await worker.initialize(); console.timeEnd('model initialization');
  const texts = ['A cat sleeps on the sofa.', 'A kitten rests on a couch.', 'The financial report describes quarterly revenue.'];
  console.time('three embeddings'); const vectors = await worker.embed(texts); console.timeEnd('three embeddings');
  const dot = (a,b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
  if (vectors.some(vector => vector.length !== 384 || Math.abs(dot(vector, vector) - 1) > 0.001)) throw new Error('Invalid normalized embedding.');
  if (dot(vectors[0],vectors[1]) <= dot(vectors[0], vectors[2])) throw new Error('Semantic sanity check failed.');
  console.log('CPU WASM smoke test passed.', {similar: dot(vectors[0],vectors[1]), unrelated: dot(vectors[0],vectors[2])});
} finally { await worker.dispose(); }
