// Builds the engine probe (a development-only Obsidian plugin) into a test vault and enables it.
// Needs: the pinned model in .cache/models (node scripts/pin-model.mjs src/model-spec.json), the native engine in
// .cache/runtime-dev, and onnxruntime-web outside package.json:
//   npm install --prefix .cache/ortweb-dev --ignore-scripts --no-save onnxruntime-web@1.31.1
// Usage: node scripts/engine-probe/build.mjs [--vault .test-vault/<name>] [--suite engines|lowbit]
// --suite engines (default): native CPU/DirectML against every onnxruntime-web build with fp32/int8 weights.
// --suite lowbit: WebGPU (JSPI build) with fp32, fp16, q4, and q4f16 weights plus kernel profiles; needs those variants
//   pinned from benchmarks/models/granite-small-r2.json (node scripts/pin-model.mjs benchmarks/models/granite-small-r2.json).
//        node scripts/engine-probe/build.mjs --portable [<out dir>]   (default dist/engine-probe-portable)
// --portable copies the model, WASM builds, and the native engine (this platform only) into the plugin folder, with
// relative paths, so the folder can be copied into any vault's .obsidian/plugins/ on another computer.
// Opening the vault (or reloading it) runs the probe; results land in <vault>/.obsidian/plugins/heading-search-engine-probe/results.json.
import { build } from 'esbuild';
import { readFile, writeFile, mkdir, cp, rm } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { readdirSync } from 'node:fs';
const option=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const portable=process.argv.includes('--portable');
const portableArg=process.argv[process.argv.indexOf('--portable')+1];
const portableRoot=portable?(portableArg && !portableArg.startsWith('--')?portableArg:'dist/engine-probe-portable'):null;
const id='heading-search-engine-probe';
const vault=portable?null:option('vault')??join('.test-vault',readdirSync('.test-vault',{withFileTypes:true}).find(e=>e.isDirectory()).name);
const target=portable?join(portableRoot,id):join(vault,'.obsidian','plugins',id);
if(portable) await rm(target,{recursive:true,force:true});
const ortWeb=resolve('.cache/ortweb-dev/node_modules/onnxruntime-web/dist');
const ortVersion=JSON.parse(await readFile('.cache/ortweb-dev/node_modules/onnxruntime-web/package.json','utf8')).version;
const suite=option('suite')??'engines';
if(!['engines','lowbit'].includes(suite)) throw new Error('--suite must be engines or lowbit');
const shipped=JSON.parse(await readFile('src/model-spec.json','utf8'));
// The low-bit variants are benchmark candidates; the shipped spec contributes the bundled DirectML graph.
const spec=suite==='lowbit' ? JSON.parse(await readFile('benchmarks/models/granite-small-r2.json','utf8')) : shipped;
if(suite==='lowbit') spec.weights.fp32.gpu=shipped.weights.fp32.gpu;
const suiteWeights=suite==='lowbit' ? ['fp32','int8','fp16','q4','q4f16'] : ['fp32','int8'];
const suiteWasm=suite==='lowbit' ? ['ort-wasm-simd-threaded.jspi.wasm'] : ['ort-wasm-simd-threaded.wasm','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.wasm','ort-wasm-simd-threaded.jspi.wasm'];
await mkdir(target,{recursive:true});

const bundles={wasm:'ort.wasm.bundle.min.mjs',jsep:'ort.bundle.min.mjs',webgpu:'ort.webgpu.bundle.min.mjs',jspi:'ort.jspi.bundle.min.mjs'};
for(const [name,bundle] of Object.entries(bundles))
  await build({entryPoints:['scripts/engine-probe/worker-web.ts'],outfile:join(target,`worker-${name}.js`),bundle:true,format:'esm',platform:'browser',
    alias:{'ort-variant':join(ortWeb,bundle)},logLevel:'error',supported:{'top-level-await':true}});
await build({entryPoints:['scripts/engine-probe/worker-native.ts'],outfile:join(target,'worker-native.js'),bundle:true,format:'iife',platform:'browser',logLevel:'error'});
await build({entryPoints:['scripts/engine-probe/plugin.ts'],outfile:join(target,'main.js'),bundle:true,format:'cjs',platform:'node',external:['obsidian','electron'],logLevel:'error'});

// The same texts as scripts/benchmark-engines.mjs uses for speed, cut to 64 documents to keep the probe short.
const corpus=(await readFile('benchmarks/data/scifact/corpus.jsonl','utf8')).trim().split('\n').map(JSON.parse);
const docs=Array.from({length:64},(_,i)=>corpus[Math.floor(i*corpus.length/64)]).map(d=>`${d.title}\n${d.text}`.trim());
const queries=(await readFile('benchmarks/data/scifact/queries.jsonl','utf8')).trim().split('\n').slice(0,16).map(l=>JSON.parse(l).text);
const weights=Object.fromEntries(Object.entries(spec.weights).filter(([k])=>suiteWeights.includes(k)).map(([k,w])=>[k,{path:w.path,external:w.extra[0].path,gpu:w.gpu?.path}]));
const modelDirectory=resolve(`.cache/models/${spec.id}/${spec.revision}`),runtimeDirectory=resolve('.cache/runtime-dev');
let paths={runtimeDirectory,ortWeb,modelDirectory};
if(portable) {
  // Only the files the probe opens: tokenizer, both weight variants, the DirectML graph, the four WASM builds,
  // and onnxruntime-node with this platform's binaries.
  const modelFiles=['tokenizer.json','tokenizer_config.json',...Object.entries(spec.weights).filter(([k])=>suiteWeights.includes(k)).map(([,w])=>w).flatMap(w=>[w.path,...w.extra.map(e=>e.path),...(w.gpu?[w.gpu.path]:[])])];
  for(const file of modelFiles) await cp(join(modelDirectory,file),join(target,'model',file));
  for(const file of suiteWasm)
    await cp(join(ortWeb,file),join(target,'ort',file));
  const modules=join(runtimeDirectory,'node_modules'),out=join(target,'runtime','node_modules');
  await cp(join(modules,'onnxruntime-common'),join(out,'onnxruntime-common'),{recursive:true});
  for(const part of ['package.json','dist','lib']) await cp(join(modules,'onnxruntime-node',part),join(out,'onnxruntime-node',part),{recursive:true});
  const binary=join('bin','napi-v6',process.platform,process.arch);
  await cp(join(modules,'onnxruntime-node',binary),join(out,'onnxruntime-node',binary),{recursive:true});
  paths={runtimeDirectory:'runtime',ortWeb:'ort',modelDirectory:'model'};
  await writeFile(join(target,'README.txt'),[
    'Heading Search engine probe (development only). Times embedding engines inside Obsidian; nothing leaves this computer.',
    '',
    `1. Use a current Obsidian installer (Electron 43 or later; Settings > About shows the installer version). Built for ${process.platform}-${process.arch}.`,
    `2. Copy this "${id}" folder into a vault's .obsidian/plugins/ folder. A new empty vault is fine.`,
    '3. Open the vault, turn on community plugins, and enable "Heading Search engine probe (dev)".',
    '   It starts once the vault has loaded; a notice shows progress.',
    '   To run it again, quit Obsidian completely first: reloading the same window leaves the native engine unusable ("OrtEnv already destroyed").',
    '4. Keep the Obsidian window focused for 5-10 minutes, until the notice says it has finished.',
    '5. Send back results.json from this folder.',
  ].join('\n')+'\n');
}
await writeFile(join(target,'probe.json'),JSON.stringify({...paths,suite,ortVersion,docs,queries,weights,
  model:{dimensions:spec.dimensions,maxTokens:spec.maxTokens,pooling:spec.pooling,queryPrefix:spec.queryPrefix,documentPrefix:spec.documentPrefix??''}}));
await writeFile(join(target,'manifest.json'),JSON.stringify({id,name:'Heading Search engine probe (dev)',version:'0.0.1',minAppVersion:'1.8.0',
  description:'Development-only: times embedding engines inside Obsidian.',author:'dev',isDesktopOnly:true},null,2));
if(!portable) {
  const list=join(vault,'.obsidian','community-plugins.json');
  const enabled=JSON.parse(await readFile(list,'utf8').catch(()=>'[]'));
  if(!enabled.includes(id)) {enabled.push(id);await writeFile(list,JSON.stringify(enabled,null,2));}
}
console.log(`Built the engine probe (onnxruntime-web ${ortVersion}) into ${resolve(target)}`);
console.log(portable ? 'Copy that folder into a vault\'s .obsidian/plugins/ on the other computer; see README.txt inside.'
  : 'Open or reload that vault in Obsidian with the window focused; it runs on load (about 5-10 minutes).');
