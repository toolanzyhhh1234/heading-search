// Compares candidate embedding models with bge-small-en-v1.5 (the first shipped model) on the full SciFact/NFCorpus test sets.
// Each candidate must already be scored: node scripts/benchmark.mjs --model <spec> --embeddings <key>-fp32 --out summary-<key>-fp32.json
// Usage: node scripts/compare-models.mjs benchmarks/models/a.json [benchmarks/models/b.json ...]
// The baseline is bge-small at the same fp32 precision (summary-native-fp32.json), so only the model differs.
import { build } from 'esbuild';
import { readFile, writeFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

await build({entryPoints:['src/core/metrics.ts'],outfile:'dist/bench/metrics.mjs',bundle:true,platform:'node',format:'esm',logLevel:'error'});
const { pairedBootstrap }=await import(pathToFileURL(resolve('dist/bench/metrics.mjs')).href);
const json=async file=>JSON.parse(await readFile(file,'utf8'));
const shipped=await json('src/model-spec.json'),baseline=await json('benchmarks/models/bge-small.json');
const sizeOf=async(spec,weights)=>{
  const variant=spec.weights[weights];
  return [variant,...(variant.extra??[])].reduce((sum,file)=>sum+file.bytes,0);
};
const rows=[{key:'native',spec:baseline,label:'bge-small-en-v1.5 (previous default)'}];
for(const file of process.argv.slice(2)) {const spec=await json(file);rows.push({key:spec.key,spec,label:`${spec.upstream.split('/')[1]}${spec.id===shipped.id?' (shipped)':''}`});}

const datasets=['scifact','nfcorpus'],modes=['int8','hybrid-int8'];
const perQuery={};
for(const row of rows) {
  row.summary=await json(`benchmarks/results/summary-${row.key}-fp32.json`);
  perQuery[row.key]={};
  for(const dataset of datasets)perQuery[row.key][dataset]=await json(`benchmarks/results/${dataset}-${row.key}-fp32-per-query.json`);
}
const results={createdAt:new Date().toISOString(),baseline:'bge-small-en-v1.5 with fp32 weights',
  note:'Dense rows use int8 vector storage, as the plugin ships. Confidence intervals are paired bootstrap 95% intervals of per-query nDCG@10 differences against the baseline.',models:[]};
for(const row of rows) {
  const entry={key:row.key,model:row.spec.upstream??row.spec.id,label:row.label,parameters:row.spec.parameters,dimensions:row.spec.dimensions,
    license:row.spec.license,fp32MB:Math.round(await sizeOf(row.spec,'fp32')/1e6),int8MB:row.spec.weights.int8?Math.round(await sizeOf(row.spec,'int8')/1e6):null,
    encodeMsPerDocument:{},scores:{}};
  for(const dataset of datasets) {
    entry.encodeMsPerDocument[dataset]=row.summary.datasets[dataset].documentEmbedding.perTextMs;
    for(const mode of modes) {
      const score=row.summary.datasets[dataset].modes[mode].ndcg10;
      const ours=perQuery[row.key][dataset][mode],base=perQuery.native[dataset][mode];
      if(ours.some((q,i)=>q.queryId!==base[i].queryId))throw new Error(`Query order differs for ${row.key} ${dataset}`);
      entry.scores[`${dataset}/${mode}`]={ndcg10:score,
        vsBaselineCI95:row.key==='native'?null:pairedBootstrap(ours.map((q,i)=>q.ndcg10-base[i].ndcg10))};
    }
  }
  results.models.push(entry);
}
await writeFile('benchmarks/results/models.json',JSON.stringify(results,null,2)+'\n');
const pad=(text,width)=>String(text).padEnd(width);
console.log(pad('model',36)+datasets.flatMap(d=>modes.map(m=>pad(`${d}/${m}`,26))).join(''));
for(const m of results.models)console.log(pad(m.label,36)+datasets.flatMap(d=>modes.map(mode=>{
  const s=m.scores[`${d}/${mode}`];return pad(`${s.ndcg10.toFixed(4)}${s.vsBaselineCI95?` [${s.vsBaselineCI95.map(x=>(x>=0?'+':'')+x.toFixed(3)).join(',')}]`:''}`,26);
})).join(''));
