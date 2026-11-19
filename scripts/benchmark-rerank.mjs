// Reranker benchmark: rerank the top-K first-stage results for every official test query and measure nDCG@10 gains.
// Usage: node scripts/benchmark-rerank.mjs --reranker benchmarks/rerankers/<name>.json --runtime <dir> [--weights fp32|int8]
//   [--device directml|native-cpu] [--first benchmarks/models/granite-small-r2.json] [--embeddings granite-small-r2-fp32]
// The first stage is hybrid search (title-weighted BM25 + int8-stored dense vectors, RRF), computed exactly as in
// scripts/benchmark.mjs from cached embeddings. Scores are cached per (query, document), so reruns and other K reuse them.
import { build } from 'esbuild';
import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { cpus } from 'node:os';
import { performance } from 'node:perf_hooks';

const option=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const rerankerSpec=JSON.parse(await readFile(option('reranker'),'utf8'));
const firstSpec=JSON.parse(await readFile(option('first')??'benchmarks/models/granite-small-r2.json','utf8'));
const profile=option('embeddings')??`${firstSpec.key??'native'}-fp32`;
const weights=option('weights')??'fp32',device=option('device')??'directml',depth=Number(option('k')??50);
const runtime=resolve(option('runtime')??'.cache/runtime-dev').replaceAll('\\','/');

await mkdir('dist/bench',{recursive:true});await mkdir('benchmarks/results',{recursive:true});
for(const name of ['bm25','vectors','metrics'])
  await build({entryPoints:[`src/core/${name}.ts`],outfile:`dist/bench/${name}.mjs`,bundle:true,platform:'node',format:'esm',logLevel:'error'});
await build({entryPoints:['src/embedding/backends.ts'],outfile:'dist/bench/backends.cjs',bundle:true,platform:'node',format:'cjs',logLevel:'error'});
const load=name=>import(pathToFileURL(resolve(`dist/bench/${name}.mjs`)).href);
const { BM25, reciprocalRankFusion }=await load('bm25');
const { packVectors, PackedVectors }=await load('vectors');
const { evaluate, pairedBootstrap }=await load('metrics');
globalThis.require=createRequire(import.meta.url);
const { openReranker }=globalThis.require(resolve('dist/bench/backends.cjs'));

const directory=resolve('.cache/models',...rerankerSpec.id.split('/'),rerankerSpec.revision);
const variant=rerankerSpec.weights[weights];
for(const file of [...rerankerSpec.files,variant]) {
  const bytes=await readFile(join(directory,file.path));
  if(bytes.length!==file.bytes || createHash('sha256').update(bytes).digest('hex')!==file.sha256)throw new Error(`Reranker asset does not match its pinned hash: ${file.path}`);
}
const head=rerankerSpec.head ? {dense:await readFile(join(directory,'2_Dense/model.safetensors')),norm:await readFile(join(directory,'3_LayerNorm/model.safetensors')),
  score:await readFile(join(directory,'4_Dense/model.safetensors'))} : undefined;
const threads=Math.max(1,Math.min(4,Math.floor(cpus().length/2)));
const started=performance.now();
const {reranker,info}=await openReranker({runtimeDirectory:runtime,modelPath:join(directory,variant.path),
  tokenizer:JSON.parse(await readFile(join(directory,'tokenizer.json'),'utf8')),config:JSON.parse(await readFile(join(directory,'tokenizer_config.json'),'utf8')),
  threads,device,maxTokens:rerankerSpec.maxTokens,head});
const loadMs=performance.now()-started;
console.log(`${rerankerSpec.key} ${weights} on ${info.label}${info.fallback?` (${info.fallback})`:''}; first stage: ${firstSpec.key} hybrid, K=${depth}`);

async function cachedVectors(dataset,kind,texts,query) {
  const digest=createHash('sha256').update(firstSpec.revision+JSON.stringify(texts)+String(query)).digest('hex').slice(0,16);
  const bytes=await readFile(`benchmarks/cache/${dataset}-${kind}-${digest}.${profile}.f32`);
  if(bytes.length!==texts.length*firstSpec.dimensions*4)throw new Error(`Embedding cache for ${dataset}-${kind} has the wrong size`);
  const values=new Float32Array(Uint8Array.from(bytes).buffer);
  return texts.map((_,i)=>values.slice(i*firstSpec.dimensions,(i+1)*firstSpec.dimensions));
}
const scoreFile=`benchmarks/cache/rerank-${rerankerSpec.key}-${weights}.json`;
let scoreCache={};
try {scoreCache=JSON.parse(await readFile(scoreFile,'utf8'));} catch(error) {if(error.code!=='ENOENT')console.log(`Ignoring unreadable score cache: ${error.message}`);}
// Write-then-rename, so an interrupted run never leaves a truncated cache behind.
const saveScores=async()=>{await writeFile(`${scoreFile}.partial`,JSON.stringify(scoreCache));await rename(`${scoreFile}.partial`,scoreFile);};
const percentile=(values,p)=>{const s=[...values].sort((a,b)=>a-b);return s[Math.min(s.length-1,Math.floor(s.length*p))];};
const average=values=>values.reduce((a,b)=>a+b,0)/values.length;

const summary={createdAt:new Date().toISOString(),reranker:{key:rerankerSpec.key,id:rerankerSpec.id,revision:rerankerSpec.revision,parameters:rerankerSpec.parameters,weights},
  engine:info.label,firstStage:`${firstSpec.id} hybrid (title-weighted BM25 + int8 dense, RRF k=60)`,depth,loadMs,datasets:{}};
try {
  for(const dataset of ['scifact','nfcorpus']) {
    const base=`benchmarks/data/${dataset}`;
    const corpus=(await readFile(`${base}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
    const qrels={};
    for(const line of (await readFile(`${base}/qrels/test.tsv`,'utf8')).trim().split('\n').slice(1)) {const [q,id,grade]=line.split('\t');(qrels[q]??={})[id]=Number(grade);}
    const queries=(await readFile(`${base}/queries.jsonl`,'utf8')).trim().split('\n').map(JSON.parse).filter(q=>qrels[q._id]);
    const texts=corpus.map(d=>`${d.title}\n${d.text}`.trim()),textById=new Map(corpus.map((d,i)=>[d._id,texts[i]]));
    const titleBM25=new BM25(corpus.map(d=>({id:d._id,text:d.text,title:d.title})),true);
    const documents=await cachedVectors(dataset,'corpus',texts,false),queryVectors=await cachedVectors(dataset,'queries',queries.map(q=>q.text),true);
    const index=new PackedVectors(packVectors(documents,'int8'),corpus.map(d=>d._id));
    const cache=scoreCache[dataset]??={};
    // Depths to report; a run at K=20 reports only rerank@20.
    const depths=[...new Set([20,depth])].filter(k=>k<=depth);
    const runs={first:[],...Object.fromEntries(depths.map(k=>[`rerank@${k}`,[]]))},latencies=[];
    let fresh=0;
    for(let i=0;i<queries.length;i++) {
      const query=queries[i];
      const lexical=titleBM25.search(query.text,101).filter(r=>r.id!==query._id).slice(0,100);
      const dense=index.search(queryVectors[i],101).filter(r=>r.id!==query._id).slice(0,100);
      const first=reciprocalRankFusion([lexical,dense],100).map(r=>r.id);
      const candidates=first.slice(0,depth),missing=candidates.filter(id=>cache[`${query._id}\t${id}`]===undefined);
      if(missing.length) {
        const before=performance.now(),scores=await reranker.score(query.text,missing.map(id=>textById.get(id)));
        // Only a full-depth batch counts as a per-query latency sample.
        if(missing.length===candidates.length)latencies.push(performance.now()-before);
        missing.forEach((id,j)=>{cache[`${query._id}\t${id}`]=scores[j];});fresh+=missing.length;
      }
      const rerank=k=>[...candidates.slice(0,k)].sort((a,b)=>cache[`${query._id}\t${b}`]-cache[`${query._id}\t${a}`]).concat(first.slice(k));
      runs.first.push(evaluate(first,qrels[query._id]));
      for(const k of depths)runs[`rerank@${k}`].push(evaluate(rerank(k),qrels[query._id]));
      if((i+1)%50===0) {console.log(`${dataset}: ${i+1}/${queries.length} queries`);await saveScores();}
    }
    await saveScores();
    const result={queries:queries.length,newPairsScored:fresh,latencyMs:latencies.length?{p50:percentile(latencies,.5),p95:percentile(latencies,.95),samples:latencies.length}:null,modes:{}};
    for(const [mode,metrics] of Object.entries(runs)) result.modes[mode]={
      ndcg10:average(metrics.map(m=>m.ndcg10)),recall10:average(metrics.map(m=>m.recall10)),mrr10:average(metrics.map(m=>m.mrr10)),
      gainOverFirstCI95:mode==='first'?null:pairedBootstrap(metrics.map((m,q)=>m.ndcg10-runs.first[q].ndcg10))};
    summary.datasets[dataset]=result;
    console.log(dataset,Object.entries(result.modes).map(([m,r])=>`${m} ${r.ndcg10.toFixed(4)}${r.gainOverFirstCI95?` [${r.gainOverFirstCI95.map(x=>(x>=0?'+':'')+x.toFixed(3)).join(',')}]`:''}`).join('  '),
      result.latencyMs?`| ${depth} pairs/query p50 ${result.latencyMs.p50.toFixed(0)} ms`:'');
  }
} finally {await reranker.dispose();}
let all={};
try {all=JSON.parse(await readFile('benchmarks/results/rerank.json','utf8'));} catch {}
// A rerun answered entirely from cached scores measures no latency; keep the earlier measurement.
const previous=all[`${rerankerSpec.key}-${weights}`];
for(const [dataset,result] of Object.entries(summary.datasets))
  if(!result.latencyMs && previous?.datasets?.[dataset]?.latencyMs)result.latencyMs=previous.datasets[dataset].latencyMs;
all[`${rerankerSpec.key}-${weights}`]=summary;
await writeFile('benchmarks/results/rerank.json',JSON.stringify(all,null,2)+'\n');
