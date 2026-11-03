import { mkdir, writeFile, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const vault=resolve('.test-vault');
const config=resolve(vault,'.obsidian');
const plugin=resolve(config,'plugins/heading-search');
await mkdir(plugin,{recursive:true});
for(const file of ['main.js','manifest.json','styles.css'])await copyFile(file,resolve(plugin,file));
await writeFile(resolve(config,'community-plugins.json'),JSON.stringify(['heading-search']));
await writeFile(resolve(config,'app.json'),JSON.stringify({showInlineTitle:true}));
await writeFile(resolve(config,'hotkeys.json'),JSON.stringify({'heading-search:search-notes':[{modifiers:['Mod','Shift'],key:'F'}]}));
const fixtures={
  'Storage.md':'# Storage design\n## Backups\nKeep a second copy of important files on another device.\n## Compression\nQuantized vectors need less disk space than floating point vectors.\n',
  'Meetings.md':'# Team meetings\n## Tuesday\nDiscuss the product launch, support rota, and next release.\n',
  'Garden.md':'# A small garden\n## Tomatoes\nWater plants in the early morning.\n## Cats\nA kitten rests on the warm patio.\n',
};
for(const [name,text] of Object.entries(fixtures))await writeFile(resolve(vault,name),text);
const profile=resolve('.cache/obsidian-test-profile');await mkdir(profile,{recursive:true});
await writeFile(resolve(profile,'obsidian.json'),JSON.stringify({vaults:{'a123456789abcdef':{path:vault,ts:Date.now(),open:true}}}));
console.log(JSON.stringify({vault,profile},null,2));
