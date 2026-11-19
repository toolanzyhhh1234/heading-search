// Pins a candidate embedding model for benchmarking: resolves its Hugging Face revision to a commit, downloads the listed
// files into .cache/models/<id>/<revision>/, and writes each file's size and SHA-256 back into the spec.
// Usage: node scripts/pin-model.mjs benchmarks/models/<name>.json
// The spec lists tokenizer files and weight variants; a variant may list "extra" files (ONNX external data).
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const file=process.argv[2];
if(!file)throw new Error('Usage: node scripts/pin-model.mjs <spec.json>');
const spec=JSON.parse(await readFile(file,'utf8'));
if(!/^[0-9a-f]{40}$/.test(spec.revision)) {
  const info=await (await fetch(`https://huggingface.co/api/models/${spec.id}/revision/${spec.revision??'main'}`)).json();
  if(!info.sha)throw new Error(`Could not resolve ${spec.id}@${spec.revision}`);
  spec.revision=info.sha;
}
const directory=resolve('.cache/models',...spec.id.split('/'),spec.revision);
const entries=[...spec.files,...Object.values(spec.weights).flatMap(weights=>[weights,...(weights.extra??[])])];
for(const entry of entries) {
  const target=resolve(directory,entry.path);
  let data;
  try {data=await readFile(target);} catch {}
  if(data && entry.sha256 && (data.length!==entry.bytes || createHash('sha256').update(data).digest('hex')!==entry.sha256))
    throw new Error(`${entry.path} on disk does not match its pinned hash`);
  if(!data) {
    const url=`https://huggingface.co/${spec.id}/resolve/${spec.revision}/${entry.path}`;
    console.log(`Downloading ${entry.path}…`);
    for(let attempt=1;;attempt++) {
      try {
        const response=await fetch(url);
        if(!response.ok)throw new Error(`${url}: HTTP ${response.status}`);
        data=Buffer.from(await response.arrayBuffer());
        break;
      } catch(error) {
        if(attempt===3)throw error;
        console.log(`  retrying after ${error.cause?.code ?? error.message}`);
      }
    }
    await mkdir(dirname(target),{recursive:true});await writeFile(target,data);
  }
  const sha256=createHash('sha256').update(data).digest('hex');
  if(entry.sha256 && entry.sha256!==sha256)throw new Error(`${entry.path} changed upstream; review before re-pinning`);
  Object.assign(entry,{sha256,bytes:data.length});
}
// A variant's GPU graph is generated here, not downloaded (scripts/patch-gpu-graph.py): install the bundled copy beside its weights.
for(const gpu of Object.values(spec.weights).map(weights=>weights.gpu).filter(Boolean)) {
  const data=await readFile(gpu.bundled);
  if(data.length!==gpu.bytes || createHash('sha256').update(data).digest('hex')!==gpu.sha256)throw new Error(`${gpu.bundled} does not match its pinned hash; run python scripts/patch-gpu-graph.py`);
  await mkdir(dirname(resolve(directory,gpu.path)),{recursive:true});await writeFile(resolve(directory,gpu.path),data);
}
await writeFile(file,JSON.stringify(spec,null,2)+'\n');
const total=entries.reduce((sum,entry)=>sum+entry.bytes,0);
console.log(`Pinned ${spec.id}@${spec.revision.slice(0,10)}: ${entries.length} files, ${(total/1e6).toFixed(0)} MB in ${directory}`);
