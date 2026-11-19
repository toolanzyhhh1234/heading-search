// Pins the native ONNX Runtime the plugin installs: npm tarball integrity plus a SHA-256 for every extracted file.
// Usage: node scripts/pin-runtime.mjs [version]. Review the diff of src/runtime-spec.json before committing.
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const version=process.argv[2]??'1.22.0';
await mkdir('dist/bench',{recursive:true});await mkdir('.cache/runtime-pin',{recursive:true});
await build({entryPoints:['src/embedding/tar.ts'],outfile:'dist/bench/tar.mjs',bundle:true,platform:'node',format:'esm',logLevel:'error'});
const { extractTarGz }=await import(pathToFileURL(resolve('dist/bench/tar.mjs')).href);

const platforms={'win32-x64':'directml','win32-arm64':'directml','darwin-arm64':null,'darwin-x64':null,'linux-x64':null,'linux-arm64':null};
// JavaScript entry points and the platform's binary folder; source maps, typings, and install scripts are skipped.
const select={
  'onnxruntime-node':path=>path==='package/package.json' || /^package\/dist\/[^/]+\.js$/.test(path) || /^package\/bin\/napi-v6\/[^/]+\/[^/]+\/[^/]+$/.test(path),
  'onnxruntime-common':path=>path==='package/package.json' || /^package\/dist\/cjs\/[^/]+\.(js|json)$/.test(path),
};
const spec={version,archives:{},shared:[],platforms:{}};
for(const name of Object.keys(select)) {
  const meta=await (await fetch(`https://registry.npmjs.org/${name}/${version}`)).json();
  const [algorithm,expected]=meta.dist.integrity.split('-');
  if(algorithm!=='sha512')throw new Error(`Unexpected integrity algorithm for ${name}`);
  const cached=resolve('.cache/runtime-pin',`${name}-${version}.tgz`);
  let data;
  try {data=await readFile(cached);} catch {data=Buffer.from(await (await fetch(meta.dist.tarball)).arrayBuffer());await writeFile(cached,data);}
  if(createHash('sha512').update(data).digest('base64')!==expected)throw new Error(`${name} tarball does not match the npm registry integrity`);
  spec.archives[name]={url:meta.dist.tarball,sha512:expected,bytes:data.length};
  const files=await extractTarGz(new Uint8Array(data),select[name]);
  for(const [path,bytes] of [...files].sort(([a],[b])=>a.localeCompare(b))) {
    const entry={archive:name,path,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length};
    const binary=path.match(/^package\/bin\/napi-v6\/([^/]+)\/([^/]+)\//);
    if(!binary) {spec.shared.push(entry);continue;}
    const key=`${binary[1]}-${binary[2]}`;
    if(!(key in platforms))continue;
    (spec.platforms[key]??={gpu:platforms[key],files:[]}).files.push(entry);
  }
}
await writeFile('src/runtime-spec.json',JSON.stringify(spec,null,2)+'\n');
for(const [key,platform] of Object.entries(spec.platforms))
  console.log(`${key}: ${platform.files.length} binaries, ${(platform.files.reduce((sum,file)=>sum+file.bytes,0)/1e6).toFixed(1)} MB extracted`);
console.log(`Pinned onnxruntime-node ${version}; download ${(Object.values(spec.archives).reduce((sum,a)=>sum+a.bytes,0)/1e6).toFixed(1)} MB.`);
