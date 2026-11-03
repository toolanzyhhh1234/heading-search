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
  const settings={...DEFAULT_SETTINGS,codec:"int8" as const as import("../src/core/vectors").VectorCodec,excludedFolders:[] as string[]};
  let embeddings=0;
  const embedding={embed:async(texts:string[])=>{embeddings+=texts.length;return texts.map(()=>{const vector=new Float32Array(384);vector[0]=1;return vector;});},dispose:()=>{}} as unknown as EmbeddingClient;
  const service=()=>new SearchService(app,()=>settings,embedding,".obsidian/plugins/heading-search/index",()=>{});
  const put=(path:string,text:string)=>{contents.set(path,text);files.set(path,{path,stat:{mtime:(files.get(path)?.stat.mtime??0)+1,size:text.length}});};
  return {app,settings,service,put,files,strings,binaries,embedding,embeddings:()=>embeddings};
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
test("a long note cannot crowd other matching notes out of the candidate list",async()=>{
  const f=fixture();
  f.put("Book.md",Array.from({length:140},(_,i)=>`# Backups ${i}\nStorage backup guidance.`).join("\n"));
  f.put("Checklist.md","# Checklist\nKeep backups of storage devices.");
  const service=f.service();await service.load();await service.refresh();
  const paths=(await service.search("backups storage")).map(r=>r.chunk.path);
  assert.ok(paths.includes("Book.md"));assert.ok(paths.includes("Checklist.md"));assert.equal(paths.length,2);
});
test("an edit during embedding supersedes stale work before the index continues",async()=>{
  const f=fixture();f.settings.semanticEnabled=true;f.put("Note.md","# Original\nOld topic.");
  const originalEmbed=f.embedding.embed.bind(f.embedding);
  let release!:()=>void,announce!:()=>void;
  const gate=new Promise<void>(resolve=>{release=resolve;});
  const started=new Promise<void>(resolve=>{announce=resolve;});
  let first=true;
  f.embedding.embed=async(texts:string[])=>{if(first){first=false;announce();await gate;}return originalEmbed(texts);};
  const service=f.service();await service.load();
  const original=service.refresh();await started;
  f.put("Note.md","# Updated\nFresh content.");const updated=service.refresh();release();
  await Promise.all([original,updated]);
  assert.equal((await service.search("old topic")).length,0);
  assert.equal((await service.search("fresh content"))[0].chunk.headings[0].text,"Updated");
  assert.equal(service.vectorCount,1);
});
