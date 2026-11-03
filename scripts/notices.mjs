import { readFile, readdir, writeFile } from 'node:fs/promises';
const lock=JSON.parse(await readFile('package-lock.json','utf8'));
let notices='# Third-party notices\n\nThe release embeds these notices in main.js so marketplace installs include their license texts. Some transitive notices cover a superset of the runtime features used by this plugin. Model weights are a separate optional download, not part of the plugin ZIP.\n\n';
for(const [directory,info] of Object.entries(lock.packages)) {
  if(!directory || info.dev || info.devOptional)continue;
  const pkg=JSON.parse(await readFile(`${directory}/package.json`,'utf8'));
  notices+=`## ${pkg.name}@${pkg.version}\n\nSource: ${typeof pkg.repository==='string'?pkg.repository:pkg.repository?.url??pkg.homepage??'See package metadata.'}\n\n`;
  const files=(await readdir(directory)).filter(file=>/^(licen[sc]e|copying|notice)(\.|$)/i.test(file));
  if(files.length)for(const file of files)notices+=`${await readFile(`${directory}/${file}`,'utf8')}\n\n`;
  else if(pkg.name==='onnxruntime-web' || pkg.name==='onnxruntime-common')notices+='License: MIT. Full Microsoft license and bundled runtime notices follow below.\n\n';
  else notices+=`Declared license: ${pkg.license??JSON.stringify(pkg.licenses)}.\n\n`;
}
for(const file of (await readdir('third-party')).sort())notices+=`## ${file}\n\n${await readFile(`third-party/${file}`,'utf8')}\n\n`;
await writeFile('THIRD_PARTY_NOTICES.md',notices);
console.log('Generated THIRD_PARTY_NOTICES.md');
