import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import os from 'node:os';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { buildWorker } from './build.mjs';
import { CPUWorker } from './worker-client.mjs';

await mkdir('dist', {recursive:true});
await mkdir('benchmarks/cache', {recursive:true});
await mkdir('benchmarks/results', {recursive:true});
await build({entryPoints:['src/core/bm25.ts','src/core/vectors.ts','src/core/metrics.ts','src/core/chunker.ts'],outdir:'dist/core',bundle:true,platform:'node',format:'esm'});
const { BM25, reciprocalRankFusion }=await import('../dist/core/bm25.js');
const { packVectors, PackedVectors }=await import('../dist/core/vectors.js');
const { evaluate, pairedBootstrap }=await import('../dist/core/metrics.js');
await writeFile('dist/embedding-worker.js', await buildWorker());
const spec=JSON.parse(await readFile('src/model-spec.json','utf8'));
const worker=new CPUWorker();
const started=performance.now();
await worker.initialize();
const modelLoadMs=performance.now()-started;

async function cachedEmbeddings(key,texts,query=false) {
  const digest=createHash('sha256').update(spec.revision+JSON.stringify(texts)+String(query)).digest('hex').slice(0,16);
  const file=`benchmarks/cache/${key}-${digest}.wasm.f32`;
  try {
    const bytes=await readFile(file);
    const timing=JSON.parse(await readFile(file+'.json','utf8'));
    if(bytes.length!==texts.length*spec.dimensions*4) throw new Error('Invalid cached length');
    const values=new Float32Array(Uint8Array.from(bytes).buffer);
    return {vectors:texts.map((_,i)=>values.slice(i*spec.dimensions,(i+1)*spec.dimensions)),...timing,cached:true};
  } catch {}
  if(!process.argv.includes('--wasm-encode-all'))throw new Error(`Missing exact WASM embedding cache for ${key}. Run node scripts/embed-benchmarks-wasm.mjs first, or explicitly use --wasm-encode-all for slow serial WASM encoding.`);
  const vectors=[],latencies=[];
  const before=performance.now();
  for(let i=0;i<texts.length;i++) {
    const start=performance.now();
    vectors.push((await worker.embed([texts[i]],query))[0]);
    latencies.push(performance.now()-start);
    if((i+1)%250===0) console.log(`${key}: ${i+1}/${texts.length}, ${(performance.now()-before).toFixed(0)} ms`);
  }
  const timing={totalMs:performance.now()-before,p50Ms:percentile(latencies,0.5),p95Ms:percentile(latencies,0.95)};
  const all=new Float32Array(vectors.length*spec.dimensions);
  vectors.forEach((v,i)=>all.set(v,i*spec.dimensions));
  await writeFile(file,new Uint8Array(all.buffer)); await writeFile(file+'.json',JSON.stringify(timing));
  return {vectors,...timing,cached:false};
}
function percentile(values,p) { const sorted=[...values].sort((a,b)=>a-b); return sorted[Math.min(sorted.length-1,Math.floor(sorted.length*p))]??0; }
const average=values=>values.reduce((a,b)=>a+b,0)/values.length;
let sourceCommit=null;
try{sourceCommit=execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim();}catch{}
const results={createdAt:new Date().toISOString(), model:spec, runtime:'Full quality and sampled latency: exact shipping onnxruntime-web 1.17.3 SIMD WASM engine, one CPU thread per inference, no GPU. Four independent workers accelerate corpus preparation only; latency samples use one worker.',
  sourceCommit,
  hardware:{cpu:os.cpus()[0].model,logicalCPUs:os.cpus().length,totalMemoryBytes:os.totalmem(),platform:os.platform(),node:process.version},
  modelLoadMs,datasets:{},policy:{minimumMeanNdcgGain:0.01,minimumPerDatasetGain:0,maxQuantizationNdcgLoss:0.005,maxQuantizationScanSlowdown:1.5},
  methodology:'Full BEIR test queries and full document corpora. Title + text. Canonical document retrieval (no synthetic headings). BM25 k1=1.2, b=0.75, fixed English stopwords and Porter stemming. BGE CLS pooling, L2 normalized, query instruction, 512 tokens. No tuning on test queries. Unjudged documents treated as non-relevant. Timings include worker messaging; model-load time separate.'};
try {
  for(const dataset of ['scifact','nfcorpus']) {
    const base=`benchmarks/data/${dataset}`;
    const corpus=(await readFile(`${base}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
    const allQueries=(await readFile(`${base}/queries.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
    const qrels={};
    for(const line of (await readFile(`${base}/qrels/test.tsv`,'utf8')).trim().split('\n').slice(1)) {
      const [query,id,grade]=line.split('\t');
      (qrels[query]??={})[id]=Number(grade);
    }
    const queries=allQueries.filter(q=>qrels[q._id]);
    const ids=corpus.map(d=>d._id), texts=corpus.map(d=>`${d.title}\n${d.text}`.trim());
    console.log(`${dataset}: ${corpus.length} corpus documents, ${queries.length} test queries`);
    const before=performance.now();
    const bm25=new BM25(corpus.map(d=>({id:d._id,text:d.text,title:d.title})));
    const bm25IndexMs=performance.now()-before;
    const titleBM25=new BM25(corpus.map(d=>({id:d._id,text:d.text,title:d.title})),true);
    const documentEmbeddings=await cachedEmbeddings(`${dataset}-corpus`,texts);
    const queryEmbeddings=await cachedEmbeddings(`${dataset}-queries`,queries.map(q=>q.text),true);
    // Measure one-worker latency separately from parallel benchmark preparation.
    const parity={maxAbsComponentError:0,minCosine:1,documents:12,queries:32,wasmDocumentMs:[],wasmQueryMs:[]};
    for(const [kind,count,input,reference,isQuery] of [
      ['documents',12,texts,documentEmbeddings.vectors,false],['queries',32,queries.map(q=>q.text),queryEmbeddings.vectors,true]]) {
      for(let j=0;j<count;j++) {
        const i=Math.floor(j*input.length/count),before=performance.now();
        const [vector]=await worker.embed([input[i]],isQuery);
        (isQuery ? parity.wasmQueryMs : parity.wasmDocumentMs).push(performance.now()-before);
        const other=reference[i];
        parity.minCosine=Math.min(parity.minCosine,vector.reduce((sum,x,k)=>sum+x*other[k],0));
        for(let k=0;k<vector.length;k++)parity.maxAbsComponentError=Math.max(parity.maxAbsComponentError,Math.abs(vector[k]-other[k]));
      }
    }
    const wasmRuntime={parity:{maxAbsComponentError:parity.maxAbsComponentError,minCosine:parity.minCosine,documents:12,queries:32},
      documentP50Ms:percentile(parity.wasmDocumentMs,0.5),documentP95Ms:percentile(parity.wasmDocumentMs,0.95),
      queryP50Ms:percentile(parity.wasmQueryMs,0.5),queryP95Ms:percentile(parity.wasmQueryMs,0.95)};
    if(parity.minCosine<0.999){await writeFile('benchmarks/results/runtime-parity-failure.json',JSON.stringify(wasmRuntime,null,2));throw new Error(`Native and WASM embeddings disagree (minimum cosine ${parity.minCosine}); do not use native quality results for this plugin.`);}
    const indexes={}; const storage={};
    for(const codec of ['f32','int8','sq4','rotated-sq4']) {
      const start=performance.now(); const packed=packVectors(documentEmbeddings.vectors,codec);
      indexes[codec]=new PackedVectors(packed,ids);
      storage[codec]={bytes:packed.length,bytesPerVector:(packed.length-16)/corpus.length,packMs:performance.now()-start};
    }
    const quality={}, timings={}, perQuery={}, overlap={};
    for(const mode of ['bm25','bm25-title','f32','int8','sq4','rotated-sq4','hybrid-int8']) { quality[mode]=[];timings[mode]=[];perQuery[mode]=[]; }
    for(const codec of Object.keys(indexes)) overlap[codec]=[];
    for(let i=0;i<queries.length;i++) {
      const query=queries[i], rankings={};
      let before=performance.now();rankings.bm25=bm25.search(query.text,101).filter(r=>r.id!==query._id).slice(0,100);timings.bm25.push(performance.now()-before);
      before=performance.now();rankings['bm25-title']=titleBM25.search(query.text,101).filter(r=>r.id!==query._id).slice(0,100);timings['bm25-title'].push(performance.now()-before);
      for(const [codec,index] of Object.entries(indexes)) {
        before=performance.now();rankings[codec]=index.search(queryEmbeddings.vectors[i],101).filter(r=>r.id!==query._id).slice(0,100);timings[codec].push(performance.now()-before);
      }
      before=performance.now();rankings['hybrid-int8']=reciprocalRankFusion([rankings['bm25-title'],rankings.int8],100);
      timings['hybrid-int8'].push(timings['bm25-title'].at(-1)+timings.int8.at(-1)+performance.now()-before);
      for(const [mode,ranking] of Object.entries(rankings)) {
        const metrics=evaluate(ranking.map(r=>r.id),qrels[query._id]);quality[mode].push(metrics);perQuery[mode].push({queryId:query._id,...metrics});
      }
      const floatTop=new Set(rankings.f32.slice(0,10).map(r=>r.id));
      for(const codec of Object.keys(indexes)) overlap[codec].push(rankings[codec].slice(0,10).filter(r=>floatTop.has(r.id)).length/10);
    }
    const hashes={};
    for(const name of ['corpus.jsonl','queries.jsonl','qrels/test.tsv'])hashes[name]=createHash('sha256').update(await readFile(`${base}/${name}`)).digest('hex');
    const summary={corpus:corpus.length,queries:queries.length,inputSha256:hashes,bm25IndexMs,wasmRuntime,documentEmbedding:{...documentEmbeddings,vectors:undefined},
      queryEmbedding:{...queryEmbeddings,vectors:undefined},storage,modes:{}};
    for(const mode of Object.keys(quality)) summary.modes[mode]={
      ...Object.fromEntries(['ndcg10','recall10','mrr10'].map(metric=>[metric,average(quality[mode].map(q=>q[metric]))])),
      searchP50Ms:percentile(timings[mode],0.5),searchP95Ms:percentile(timings[mode],0.95),
      top10OverlapWithFloat:overlap[mode]?average(overlap[mode]):undefined,
      ndcgGainOverBM25CI95:pairedBootstrap(quality[mode].map((q,i)=>q.ndcg10-quality.bm25[i].ndcg10)),
      ndcgGainOverTitleBM25CI95:pairedBootstrap(quality[mode].map((q,i)=>q.ndcg10-quality['bm25-title'][i].ndcg10)),
    };
    results.datasets[dataset]=summary;
    await writeFile(`benchmarks/results/${dataset}-per-query.json`,JSON.stringify(perQuery,null,2));
    await writeFile('benchmarks/results/summary.json',JSON.stringify(results,null,2)+'\n');
    console.log(JSON.stringify({dataset,modes:summary.modes},null,2));
  }
  const datasets=Object.values(results.datasets);
  const candidates=['int8','hybrid-int8'];
  const baseline=d=>Math.max(d.modes.bm25.ndcg10,d.modes['bm25-title'].ndcg10);
  const clears=mode=>datasets.every(d=>d.modes[mode].ndcg10>=baseline(d) && d.modes[mode].ndcgGainOverBM25CI95[0]>0 && d.modes[mode].ndcgGainOverTitleBM25CI95[0]>0) &&
    average(datasets.map(d=>d.modes[mode].ndcg10-baseline(d)))>=results.policy.minimumMeanNdcgGain;
  const denseWins=clears('int8');
  const acceptable=candidates.filter(mode=>denseWins && clears(mode));
  results.recommendation={defaultMode:acceptable.sort((a,b)=>average(datasets.map(d=>d.modes[b].ndcg10-d.modes[a].ndcg10)))[0]??'bm25',
    reason:'Dense vectors must first beat both plain and title-weighted BM25 with >= 0.01 absolute macro nDCG@10 gain over the stronger baseline, no dataset regression, and positive paired bootstrap lower bounds on each dataset. A hybrid default must independently clear the same gate. Otherwise default to heading-aware BM25. Quantization is evaluated separately from model quality.'};
  const eligibleCodecs=['int8','sq4','rotated-sq4'].filter(codec=>datasets.every(d=>d.modes.f32.ndcg10-d.modes[codec].ndcg10<=results.policy.maxQuantizationNdcgLoss && d.modes[codec].top10OverlapWithFloat>=0.95 && d.modes[codec].searchP95Ms<=results.policy.maxQuantizationScanSlowdown*d.modes.int8.searchP95Ms));
  results.recommendation.codec=eligibleCodecs.sort((a,b)=>datasets[0].storage[a].bytes-datasets[0].storage[b].bytes)[0]??'f32';
  await writeFile('benchmarks/results/summary.json',JSON.stringify(results,null,2)+'\n');
  console.log('Recommendation:',results.recommendation);
} finally { await worker.dispose(); }
