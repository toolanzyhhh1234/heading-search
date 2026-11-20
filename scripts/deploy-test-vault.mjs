// Copy the built plugin into every vault under .test-vault/ (gitignored) and enable it.
// Leaves each vault's index/ and data.json alone. Run `npm run build` first, or `npm run deploy:test-vault`.
import { readFile, writeFile, mkdir, copyFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
const manifest=JSON.parse(await readFile('manifest.json','utf8'));
const root='.test-vault';
const vaults=[];
for(const entry of await readdir(root,{withFileTypes:true}).catch(()=>[])) {
  if(entry.isDirectory() && await stat(join(root,entry.name,'.obsidian')).then(s=>s.isDirectory(),()=>false))
    vaults.push(join(root,entry.name));
}
if(!vaults.length) { console.error(`No vaults found: put each vault in its own folder under ${root}/ and open it in Obsidian once.`); process.exit(1); }
for(const vault of vaults) {
  const target=join(vault,'.obsidian','plugins',manifest.id);
  await mkdir(target,{recursive:true});
  for(const file of ['main.js','manifest.json','styles.css']) await copyFile(file,join(target,file));
  const list=join(vault,'.obsidian','community-plugins.json');
  const enabled=JSON.parse(await readFile(list,'utf8').catch(()=>'[]'));
  if(!enabled.includes(manifest.id)) { enabled.push(manifest.id); await writeFile(list,JSON.stringify(enabled,null,2)); }
  console.log(`Deployed ${manifest.id} ${manifest.version} to ${target}`);
}
console.log('Reload the plugin in Obsidian: disable and re-enable it, or run "Reload app without saving".');
