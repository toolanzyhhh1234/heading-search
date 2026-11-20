// Builds an Obsidian vault from a BEIR corpus, for timing indexing inside Obsidian on a fixed, shareable set of notes.
// Usage: node scripts/make-benchmark-vault.mjs [--dataset scifact|nfcorpus] [--limit N]   (needs prepare-benchmarks.py)
// Writes .test-vault/<dataset>[-N]/ (gitignored): one note per document, "# <title>" then the abstract, so each note is
// one heading section, like the benchmark's documents. Then `npm run deploy:test-vault` installs the plugin into it.
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { noteNames } from './beir-notes.mjs';
const arg=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const dataset=arg('dataset')??'scifact',limit=arg('limit')?Number(arg('limit')):undefined;
const corpus=(await readFile(`benchmarks/data/${dataset}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse).slice(0,limit);
const vault=join('.test-vault',`${dataset}${limit?`-${limit}`:''}`);
await rm(vault,{recursive:true,force:true});
await mkdir(join(vault,'.obsidian'),{recursive:true});
const names=noteNames(corpus);
for(const [i,d] of corpus.entries()) await writeFile(join(vault,`${names[i]}.md`),`# ${d.title.trim()||names[i]}\n\n${d.text}\n`);
await writeFile(join(vault,'.obsidian','community-plugins.json'),JSON.stringify(['heading-search'],null,2));
console.log(`Wrote ${corpus.length} notes to ${vault}. Next: npm run deploy:test-vault, then open the folder as a vault in Obsidian.`);
