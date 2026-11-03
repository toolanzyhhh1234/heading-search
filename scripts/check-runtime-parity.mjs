import { readFile,writeFile } from 'node:fs/promises';
import { CPUWorker } from './worker-client.mjs';
const corpus=(await readFile('benchmarks/data/scifact/corpus.jsonl','utf8')).trim().split('\n').map(JSON.parse);
const texts=Array.from({length:12},(_,i)=>{const d=corpus[Math.floor(i*corpus.length/12)];return `${d.title}\n${d.text}`.trim();});
const worker=new CPUWorker();
try {
  await worker.initialize();
  const vectors=await worker.embed(texts);
  await writeFile('benchmarks/cache/parity-input.json',JSON.stringify(texts));
  const array=new Float32Array(12*384);vectors.forEach((v,i)=>array.set(v,i*384));
  await writeFile('benchmarks/cache/parity-wasm.f32',new Uint8Array(array.buffer));
}finally{await worker.dispose();}
