// Build two throwaway Obsidian vaults (SciFact, NFCorpus) with the real Omnisearch 1.32.0 plugin and its HTTP API on,
// so scripts/benchmark-omnisearch.mjs --live can check the reproduction against the running plugin.
// Vaults go in .cache/omnisearch-vaults/ (gitignored). Each abstract becomes one note, named as in scripts/beir-notes.mjs.
// Then: open each folder in Obsidian (Open folder as vault), choose "Trust author and enable plugins", leave both open.
import { readFile, writeFile, mkdir, readdir, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { noteNames, LIVE_PORTS } from './beir-notes.mjs';

const VERSION='1.32.0';
// SHA-256 of the GitHub release assets for 1.32.0 (tag commit e4a51ab), recorded when first downloaded.
const ASSETS={
  'main.js':'2734d64d4e483773f7921fd0710073519545f72c89f72caa9e96b3eaeb1e95ea',
  'manifest.json':'0e48f87a7bce226c398782760f52c0b892aaa396cd5a762f48a200c7e779021f',
  'styles.css':'c6a56b1a18ff867f12ebbb3a139060a249704281e1b449003ff487969621a2e6',
};
const sha256=data=>createHash('sha256').update(data).digest('hex');
const releaseDir='.cache/omnisearch-release';
await mkdir(releaseDir,{recursive:true});
const assets={};
for(const [name,hash] of Object.entries(ASSETS)) {
  let data=await readFile(join(releaseDir,name)).catch(()=>null);
  if(!data || sha256(data)!==hash) {
    const response=await fetch(`https://github.com/scambier/obsidian-omnisearch/releases/download/${VERSION}/${name}`);
    if(!response.ok) throw new Error(`Downloading Omnisearch ${name} failed: ${response.status}`);
    data=Buffer.from(await response.arrayBuffer());
    if(sha256(data)!==hash) throw new Error(`Omnisearch ${name} does not match the pinned SHA-256.`);
    await writeFile(join(releaseDir,name),data);
  }
  assets[name]=data;
}

for(const dataset of ['scifact','nfcorpus']) {
  const corpus=(await readFile(`benchmarks/data/${dataset}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
  const names=noteNames(corpus);
  const vault=join('.cache','omnisearch-vaults',dataset);
  await mkdir(vault,{recursive:true});
  // Only this script writes here: replace earlier notes so renamed ones do not linger.
  for(const file of await readdir(vault)) if(file.endsWith('.md')) await rm(join(vault,file));
  for(const [i,d] of corpus.entries()) await writeFile(join(vault,`${names[i]}.md`),d.text);
  const plugin=join(vault,'.obsidian','plugins','omnisearch');
  await mkdir(plugin,{recursive:true});
  for(const [name,data] of Object.entries(assets)) await writeFile(join(plugin,name),data);
  // Settings are merged over Omnisearch's defaults, so only the HTTP API is changed.
  await writeFile(join(plugin,'data.json'),JSON.stringify({httpApiEnabled:true,httpApiPort:String(LIVE_PORTS[dataset]),httpApiNotice:false},null,2));
  await writeFile(join(vault,'.obsidian','community-plugins.json'),JSON.stringify(['omnisearch'],null,2));
  console.log(`${dataset}: ${corpus.length} notes, Omnisearch HTTP API on port ${LIVE_PORTS[dataset]}: ${resolve(vault)}`);
}
console.log('\nOpen each folder in Obsidian (Open folder as vault), trust the author to enable plugins, and keep both vaults open.');
console.log('Then run: node scripts/benchmark-omnisearch.mjs --live');
