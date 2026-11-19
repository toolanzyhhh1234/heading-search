// Indexing throughput and query latency for each engine, through the plugin's real worker bundle.
// Usage: node scripts/benchmark-engines.mjs --runtime <dir containing node_modules/onnxruntime-node> [--model <spec.json>] [--out engines.json]
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { buildWorker } from './build.mjs';
import { CPUWorker } from './worker-client.mjs';

const option=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const runtimeDirectory=option('runtime')??'.cache/runtime-dev',spec=option('model')??'src/model-spec.json',output=option('out')??'engines.json';
const runtime=JSON.parse(await readFile('src/runtime-spec.json','utf8'));
await mkdir('dist',{recursive:true});await mkdir('benchmarks/results',{recursive:true});
await writeFile('dist/embedding-worker.js',await buildWorker());

const percentile=(values,p)=>{const sorted=[...values].sort((a,b)=>a-b);return sorted[Math.min(sorted.length-1,Math.floor(sorted.length*p))];};
const documents=[],queries=[];
for(const dataset of ['scifact','nfcorpus']) {
  const corpus=(await readFile(`benchmarks/data/${dataset}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
  const all=(await readFile(`benchmarks/data/${dataset}/queries.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
  // Evenly spaced, so the sample follows each corpus's length distribution.
  for(let i=0;i<64;i++){const d=corpus[Math.floor(i*corpus.length/64)];documents.push(`${d.title}\n${d.text}`.trim());}
  for(let i=0;i<16;i++)queries.push(all[Math.floor(i*all.length/16)].text);
}
const engines=[
  {name:'native-cpu-int8',options:{runtimeDirectory,spec,weights:'int8',device:'native-cpu'}},
  {name:'native-cpu-fp32',options:{runtimeDirectory,spec,weights:'fp32',device:'native-cpu'}},
  ...(process.platform==='win32'?[{name:'directml-fp32',options:{runtimeDirectory,spec,weights:'fp32',device:'directml'}}]:[]),
];
const results={createdAt:new Date().toISOString(),model:JSON.parse(await readFile(spec,'utf8')).id,runtime:`onnxruntime-node ${runtime.version}`,
  hardware:{cpu:os.cpus()[0].model,logicalCPUs:os.cpus().length,platform:`${os.platform()}-${os.arch()}`},
  sample:`${documents.length} evenly spaced SciFact/NFCorpus documents (title + text), sent 16 at a time as the plugin indexes; ${queries.length} queries one at a time`,
  engines:{}};
for(const {name,options} of engines) {
  const worker=new CPUWorker(options);
  try {
    const start=performance.now();const info=await worker.initialize();const loadMs=performance.now()-start;
    await worker.embed(documents.slice(0,16));await worker.embed([queries[0]],true); // warm caches and GPU shapes
    const indexStart=performance.now();
    for(let i=0;i<documents.length;i+=16)await worker.embed(documents.slice(i,i+16));
    const msPerDocument=(performance.now()-indexStart)/documents.length;
    const latencies=[];
    for(const query of queries){const t=performance.now();await worker.embed([query],true);latencies.push(performance.now()-t);}
    results.engines[name]={label:info.label,fallback:info.fallback,weights:options.weights,loadMs,msPerDocument,queryP50Ms:percentile(latencies,.5),queryP95Ms:percentile(latencies,.95)};
    console.log(`${name.padEnd(16)} ${info.label.padEnd(36)} ${msPerDocument.toFixed(1).padStart(7)} ms/document  query p50 ${percentile(latencies,.5).toFixed(1)} ms`);
  } finally {await worker.dispose();}
}
await writeFile(`benchmarks/results/${output}`,JSON.stringify(results,null,2)+'\n');
