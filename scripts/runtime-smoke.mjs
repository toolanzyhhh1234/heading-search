// Smoke-tests the bundled worker on the native engine: the CPU, plus DirectML on Windows when --weights fp32.
// Usage: node scripts/runtime-smoke.mjs [--runtime <dir containing node_modules/onnxruntime-node>] [--weights int8|fp32] [--model <spec.json>]
import { buildWorker } from './build.mjs';
import { mkdir, writeFile } from 'node:fs/promises';
import { CPUWorker } from './worker-client.mjs';
const option=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const runtimeDirectory=option('runtime')??'.cache/runtime-dev',weights=option('weights')??'int8',spec=option('model')??'src/model-spec.json';
await mkdir('dist', {recursive:true});
await writeFile('dist/embedding-worker.js', await buildWorker());
const texts = ['A cat sleeps on the sofa.', 'A kitten rests on a couch.', 'The financial report describes quarterly revenue.'];
const dot = (a,b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
// Setup pairs int8 with the CPU and fp32 with DirectML; fp32 is also what the CPU uses for queries next to DirectML.
const devices=['native-cpu',...(weights==='fp32' && process.platform==='win32' ? ['directml'] : [])];
const results={};
for(const device of devices) {
  const worker = new CPUWorker({runtimeDirectory,weights,spec,device});
  try {
    const started=performance.now();const info=await worker.initialize();const loadMs=performance.now()-started;
    // Batched and one-by-one embeddings must agree: padding is masked and must not change the pooled vector.
    const batched=await worker.embed(texts),single=[];
    for(const text of texts)single.push((await worker.embed([text]))[0]);
    if(batched.some(vector => Math.abs(dot(vector, vector) - 1) > 0.001)) throw new Error('Invalid normalized embedding.');
    const agreement=Math.min(...batched.map((vector,i)=>dot(vector,single[i])));
    if(agreement<0.9999)throw new Error(`Batched and single embeddings disagree (${agreement}).`);
    if(dot(batched[0],batched[1]) <= dot(batched[0], batched[2])) throw new Error('Semantic sanity check failed.');
    results[info.label]={loadMs:Math.round(loadMs),fallback:info.fallback,batchAgreement:agreement,similar:dot(batched[0],batched[1]),unrelated:dot(batched[0],batched[2]),vectors:batched};
  } finally { await worker.dispose(); }
}
const engines=Object.values(results);
if(engines.length===2) {
  // Documents embedded on the GPU and queries embedded on the CPU must share one vector space.
  const agreement=Math.min(...engines[0].vectors.map((vector,i)=>dot(vector,engines[1].vectors[i])));
  console.log(`CPU vs DirectML agreement with ${weights} weights: ${agreement.toFixed(6)}`);
  if(agreement<0.999)throw new Error('CPU and DirectML embeddings are not interchangeable.');
}
for(const result of engines)delete result.vectors;
console.log('Runtime smoke test passed.', results);
