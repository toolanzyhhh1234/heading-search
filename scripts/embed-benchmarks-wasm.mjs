import { readFile,writeFile,open,stat,mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { CPUWorker } from './worker-client.mjs';
import { buildWorker } from './build.mjs';

const spec=JSON.parse(await readFile('src/model-spec.json','utf8'));
const concurrency=4;
await mkdir('dist',{recursive:true});await mkdir('benchmarks/cache',{recursive:true});
await writeFile('dist/embedding-worker.js',await buildWorker());
const workers=Array.from({length:concurrency},()=>new CPUWorker());
try {
  await Promise.all(workers.map(worker=>worker.initialize()));
  for(const dataset of ['scifact','nfcorpus']) {
    const base=`benchmarks/data/${dataset}`;
    const corpus=(await readFile(`${base}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
    const qrels=new Set((await readFile(`${base}/qrels/test.tsv`,'utf8')).trim().split('\n').slice(1).map(line=>line.split('\t')[0]));
    const queries=(await readFile(`${base}/queries.jsonl`,'utf8')).trim().split('\n').map(JSON.parse).filter(q=>qrels.has(q._id));
    for(const [kind,texts,query] of [['queries',queries.map(q=>q.text),true],['corpus',corpus.map(d=>`${d.title}\n${d.text}`.trim()),false]]) {
      const digest=createHash('sha256').update(spec.revision+JSON.stringify(texts)+String(query)).digest('hex').slice(0,16);
      const destination=`benchmarks/cache/${dataset}-${kind}-${digest}.wasm.f32`;
      try {if((await stat(destination)).size===texts.length*384*4){console.log(`${dataset}-${kind}: WASM cached`);continue;}}catch{}
      const partial=destination+'.partial',checkpoint=destination+'.progress.json';
      let done=new Set();
      try {done=new Set(JSON.parse(await readFile(checkpoint,'utf8')));}catch{}
      const output=await open(partial,done.size?'r+':'w+');
      let cursor=0,completed=done.size;
      const latencies=[],before=performance.now();
      const pool=workers.map(async worker=>{
        while(cursor<texts.length) {
          const i=cursor++;
          if(done.has(i))continue;
          const start=performance.now(),[vector]=await worker.embed([texts[i]],query);
          latencies.push(performance.now()-start);
          await output.write(new Uint8Array(vector.buffer),0,384*4,i*384*4);
          done.add(i);completed++;
          if(completed%250===0) {
            await writeFile(checkpoint,JSON.stringify([...done]));
            console.log(`${dataset}-${kind}: ${completed}/${texts.length} exact WASM embeddings, ${((performance.now()-before)/1000).toFixed(1)}s`);
          }
        }
      });
      try {await Promise.all(pool);await output.sync();}finally{await output.close();}
      // Copy to a final file only after every indexed row has been written.
      await writeFile(destination,await readFile(partial));
      const sorted=latencies.sort((a,b)=>a-b);
      const timing={totalMs:performance.now()-before,p50Ms:sorted[Math.floor(sorted.length*.5)],p95Ms:sorted[Math.floor(sorted.length*.95)],
        backend:`onnxruntime-web 1.17.3 SIMD WASM, one thread per request, ${concurrency} independent workers for benchmark preparation`,completedPreviously:texts.length-latencies.length};
      await writeFile(destination+'.json',JSON.stringify(timing));
      console.log(`${dataset}-${kind}: finished in ${(timing.totalMs/1000).toFixed(1)}s`);
    }
  }
}finally{await Promise.all(workers.map(worker=>worker.dispose()));}
