// Full-corpus encoding with the shipping native engine (batched, DirectML or multi-threaded CPU).
// Usage: node scripts/embed-benchmarks-native.mjs --weights int8|fp32 --runtime <dir containing node_modules/onnxruntime-node> [--device directml|native-cpu] [--model <spec.json>]
// --model defaults to the shipped src/model-spec.json; candidate specs live in benchmarks/models/ (see scripts/pin-model.mjs).
// The device defaults to the one the plugin pairs with the weights: DirectML for fp32, CPU for int8.
import { build } from 'esbuild';
import { readFile, writeFile, stat, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { cpus } from 'node:os';
import { performance } from 'node:perf_hooks';

const arg=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const weights=arg('weights')??'fp32',runtime=resolve(arg('runtime')??'.cache/runtime-dev');
const spec=JSON.parse(await readFile(arg('model')??'src/model-spec.json','utf8'));
// Cache names: the shipped model keeps its original 'native-<weights>' profile; candidates use their key.
const profile=`${spec.key??'native'}-${weights}`;
const variant=spec.weights[weights];
if(!variant)throw new Error(`Unknown weights ${weights}; expected one of ${Object.keys(spec.weights).join(', ')}`);
const directory=resolve('.cache/models',spec.id,spec.revision);
for(const file of [...spec.files,variant,...(variant.extra??[])]) {
  const bytes=await readFile(resolve(directory,file.path));
  if(bytes.length!==file.bytes || createHash('sha256').update(bytes).digest('hex')!==file.sha256)throw new Error(`Benchmark asset does not match its pinned hash: ${file.path}`);
}
await mkdir('dist/bench',{recursive:true});await mkdir('benchmarks/cache',{recursive:true});
await build({entryPoints:['src/embedding/backends.ts'],outfile:'dist/bench/backends.cjs',bundle:true,platform:'node',format:'cjs',logLevel:'error'});
globalThis.require=createRequire(import.meta.url);
const { openNative }=globalThis.require(resolve('dist/bench/backends.cjs'));
const threads=Math.max(1,Math.min(4,Math.floor(cpus().length/2)));
const {engine,info}=await openNative({runtimeDirectory:runtime.replaceAll('\\','/'),modelPath:resolve(directory,variant.path),
  tokenizer:JSON.parse(await readFile(resolve(directory,'tokenizer.json'),'utf8')),config:JSON.parse(await readFile(resolve(directory,'tokenizer_config.json'),'utf8')),
  threads,device:arg('device')??(weights==='fp32'?'directml':'native-cpu'),
  model:{dimensions:spec.dimensions,maxTokens:spec.maxTokens,pooling:spec.pooling,queryPrefix:spec.queryPrefix,documentPrefix:spec.documentPrefix??''}});
console.log(`Engine: ${info.label}, ${spec.id} ${weights} weights → profile ${profile}`);
try {
  for(const dataset of ['scifact','nfcorpus']) {
    const base=`benchmarks/data/${dataset}`;
    const corpus=(await readFile(`${base}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
    const qrels=new Set((await readFile(`${base}/qrels/test.tsv`,'utf8')).trim().split('\n').slice(1).map(line=>line.split('\t')[0]));
    const queries=(await readFile(`${base}/queries.jsonl`,'utf8')).trim().split('\n').map(JSON.parse).filter(q=>qrels.has(q._id));
    for(const [kind,texts,query] of [['queries',queries.map(q=>q.text),true],['corpus',corpus.map(d=>`${d.title}\n${d.text}`.trim()),false]]) {
      // The digest covers model revision, inputs and role, so every profile embeds exactly the same inputs.
      const digest=createHash('sha256').update(spec.revision+JSON.stringify(texts)+String(query)).digest('hex').slice(0,16);
      const destination=`benchmarks/cache/${dataset}-${kind}-${digest}.${profile}.f32`;
      try {if((await stat(destination)).size===texts.length*spec.dimensions*4){console.log(`${dataset}-${kind}: cached`);continue;}}catch{}
      const all=new Float32Array(texts.length*spec.dimensions),before=performance.now(),step=256;
      for(let i=0;i<texts.length;i+=step) {
        const vectors=await engine.embed(texts.slice(i,i+step),query);
        vectors.forEach((vector,j)=>all.set(vector,(i+j)*spec.dimensions));
        if((i/step)%4===3)console.log(`${dataset}-${kind}: ${Math.min(i+step,texts.length)}/${texts.length}, ${((performance.now()-before)/1000).toFixed(1)}s`);
      }
      await writeFile(destination,new Uint8Array(all.buffer));
      const totalMs=performance.now()-before;
      await writeFile(destination+'.json',JSON.stringify({totalMs,perTextMs:totalMs/texts.length,backend:`onnxruntime-node ${info.label}, ${spec.id} (${weights} weights)`}));
      console.log(`${dataset}-${kind}: ${texts.length} texts in ${(totalMs/1000).toFixed(1)}s (${(totalMs/texts.length).toFixed(1)} ms/text)`);
    }
  }
} finally {await engine.dispose();}
