import { test } from "node:test";
import assert from "node:assert/strict";
import type { App, DataAdapter } from "obsidian";
import { SearchService } from "../src/index/service";
import { IndexStorage } from "../src/index/storage";
import type { EmbeddingClient } from "../src/embedding/client";
import { DEFAULT_SETTINGS } from "../src/settings";

function fixture() {
  const contents=new Map<string,string>(),files=new Map<string,{path:string;stat:{mtime:number;size:number}}>();
  const strings=new Map<string,string>(),binaries=new Map<string,ArrayBuffer>();
  const folders=new Set<string>();
  const adapter={
    exists:async(path:string)=>folders.has(path)||strings.has(path)||binaries.has(path),
    mkdir:async(path:string)=>{folders.add(path);},
    read:async(path:string)=>{if(!strings.has(path))throw new Error("Missing file");return strings.get(path)!;},
    write:async(path:string,value:string)=>{strings.set(path,value);},
    readBinary:async(path:string)=>{if(!binaries.has(path))throw new Error("Missing binary");return binaries.get(path)!;},
    writeBinary:async(path:string,value:ArrayBuffer)=>{binaries.set(path,value);},
    list:async(path:string)=>({files:[...strings.keys(),...binaries.keys()].filter(p=>p.startsWith(path+"/")),folders:[]}),
    remove:async(path:string)=>{strings.delete(path);binaries.delete(path);},
  } as unknown as DataAdapter;
  const app={vault:{adapter,getMarkdownFiles:()=>[...files.values()],getFileByPath:(path:string)=>files.get(path)??null,read:async(file:{path:string})=>contents.get(file.path)!}} as unknown as App;
  const settings={...DEFAULT_SETTINGS,excludedFolders:[] as string[]};
  let embeddings=0;
  const embedding={embed:async(texts:string[])=>{embeddings+=texts.length;return texts.map(()=>{const vector=new Float32Array(384);vector[0]=1;return vector;});},dispose:()=>{}} as unknown as EmbeddingClient;
  const service=()=>new SearchService(app,()=>settings,embedding,".obsidian/plugins/heading-search/index",()=>{});
  const put=(path:string,text:string)=>{contents.set(path,text);files.set(path,{path,stat:{mtime:(files.get(path)?.stat.mtime??0)+1,size:text.length}});};
  return {app,settings,service,put,files,strings,binaries,embeddings:()=>embeddings};
}

test("incremental indexing follows edits, moves, exclusions, deletion and restart",async()=>{
  const f=fixture();f.put("Guide.md","# Storage\nBackups protect your notes.");f.put("Private/Secret.md","# Passwords\nSensitive secrets");
  let service=f.service();await service.load();await service.refresh();
  assert.equal((await service.search("backups"))[0].chunk.path,"Guide.md");assert.equal(f.embeddings(),0);
  f.put("Guide.md","# Garden\nTomatoes grow in sunlight.");await service.refresh();assert.equal((await service.search("backups")).length,0);
  f.settings.excludedFolders=["Private"];assert.equal((await service.search("secrets")).length,0);
  await service.refresh();assert.equal((await service.search("secrets")).length,0);
  f.put("Moved.md","# Garden\nTomatoes grow in sunlight.");f.files.delete("Guide.md");await service.refresh();
  assert.deepEqual((await service.search("tomatoes")).map(r=>r.chunk.path),["Moved.md"]);
  service.dispose();service=f.service();await service.load();assert.equal((await service.search("tomatoes"))[0].chunk.path,"Moved.md");
  f.files.delete("Moved.md");await service.refresh();assert.equal((await service.search("tomatoes")).length,0);
});
test("semantic vectors persist without repeated embedding and changing codec rebuilds them",async()=>{
  const f=fixture();f.settings.semanticEnabled=true;f.put("Note.md","# Ideas\nA useful thought.");
  let service=f.service();await service.load();await service.refresh();assert.equal(f.embeddings(),1);
  await service.refresh();assert.equal(f.embeddings(),1);
  service.dispose();service=f.service();await service.load();await service.refresh();assert.equal(f.embeddings(),1);
  assert.equal((await service.search("useful","vector"))[0].chunk.path,"Note.md");
  const before=f.embeddings();f.settings.codec="sq4";await service.refresh();assert.equal(f.embeddings(),before+1);
});
test("interrupted or corrupt cache generations rebuild from notes",async()=>{
  const f=fixture();f.put("Note.md","# Restore\nRecover from the source note.");
  const original=f.service();await original.load();await original.refresh();
  const binaryPath=[...f.binaries.keys()][0];
  const sameLengthCorruption=Uint8Array.from(new Uint8Array(f.binaries.get(binaryPath)!));sameLengthCorruption[0]^=1;
  f.binaries.set(binaryPath,sameLengthCorruption.buffer);
  assert.equal(await new IndexStorage(f.app.vault.adapter,".obsidian/plugins/heading-search/index").load(),null);
  await original.refresh();
  const binary=[...f.binaries.keys()][0];f.binaries.set(binary,new ArrayBuffer(2));
  const service=f.service();await service.load();await service.refresh();
  assert.equal((await service.search("recover"))[0].chunk.path,"Note.md");
  f.strings.set(".obsidian/plugins/heading-search/index/current.json",'{"generation":"../../untrusted"}');
  assert.equal(await new IndexStorage(f.app.vault.adapter,".obsidian/plugins/heading-search/index").load(),null);
});
