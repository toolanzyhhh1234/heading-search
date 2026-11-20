// Builds the engine probe (a development-only Obsidian plugin) into a test vault and enables it.
// Needs: the pinned model in .cache/models (node scripts/pin-model.mjs src/model-spec.json), the native engine in
// .cache/runtime-dev, and onnxruntime-web outside package.json:
//   npm install --prefix .cache/ortweb-dev --ignore-scripts --no-save onnxruntime-web@1.31.1
// Usage: node scripts/engine-probe/build.mjs [--vault .test-vault/<name>]
// Opening the vault (or reloading it) runs the probe; results land in <vault>/.obsidian/plugins/heading-search-engine-probe/results.json.
import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { readdirSync } from 'node:fs';
const option=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const vault=option('vault')??join('.test-vault',readdirSync('.test-vault',{withFileTypes:true}).find(e=>e.isDirectory()).name);
const id='heading-search-engine-probe',target=join(vault,'.obsidian','plugins',id);
const ortWeb=resolve('.cache/ortweb-dev/node_modules/onnxruntime-web/dist');
const ortVersion=JSON.parse(await readFile('.cache/ortweb-dev/node_modules/onnxruntime-web/package.json','utf8')).version;
const spec=JSON.parse(await readFile('src/model-spec.json','utf8'));
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
const weights=Object.fromEntries(Object.entries(spec.weights).map(([k,w])=>[k,{path:w.path,external:w.extra[0].path,gpu:w.gpu?.path}]));
await writeFile(join(target,'probe.json'),JSON.stringify({repo:resolve('.'),runtimeDirectory:resolve('.cache/runtime-dev'),ortWeb,ortVersion,
  modelDirectory:resolve(`.cache/models/${spec.id}/${spec.revision}`),docs,queries,weights,
  model:{dimensions:spec.dimensions,maxTokens:spec.maxTokens,pooling:spec.pooling,queryPrefix:spec.queryPrefix,documentPrefix:spec.documentPrefix??''}}));
await writeFile(join(target,'manifest.json'),JSON.stringify({id,name:'Heading Search engine probe (dev)',version:'0.0.1',minAppVersion:'1.8.0',
  description:'Development-only: times embedding engines inside Obsidian.',author:'dev',isDesktopOnly:true},null,2));
const list=join(vault,'.obsidian','community-plugins.json');
const enabled=JSON.parse(await readFile(list,'utf8').catch(()=>'[]'));
if(!enabled.includes(id)) {enabled.push(id);await writeFile(list,JSON.stringify(enabled,null,2));}
console.log(`Built the engine probe (onnxruntime-web ${ortVersion}) into ${resolve(target)}`);
console.log('Open or reload that vault in Obsidian with the window focused; it runs on load (about 5-10 minutes).');
