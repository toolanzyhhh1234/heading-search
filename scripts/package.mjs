import { zipSync } from 'fflate';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const manifest=JSON.parse(await readFile('manifest.json','utf8'));
const files=['main.js','manifest.json','styles.css'];
const archive={};const checksums=[];
for(const file of files) {
  const data=new Uint8Array(await readFile(file));archive[`${manifest.id}/${file}`]=data;
  checksums.push(`${createHash('sha256').update(data).digest('hex')}  ${file}`);
}
for(const file of ['LICENSE','THIRD_PARTY_NOTICES.md'])archive[`${manifest.id}/${file}`]=new Uint8Array(await readFile(file));
await mkdir('dist/release',{recursive:true});
for(const file of files)await writeFile(`dist/release/${file}`,await readFile(file));
const name=`dist/${manifest.id}-${manifest.version}.zip`;
await writeFile(name,zipSync(archive,{level:9}));
await writeFile('dist/release/SHA256SUMS',checksums.join('\n')+'\n');
console.log(`Packaged ${name}; the Community Plugins release assets are in dist/release/.`);
