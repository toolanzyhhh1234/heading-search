import type { App, TFile } from "obsidian";
import { BM25, chunkDocument, reciprocalRankFusion } from "../core/bm25";
import { chunkMarkdown, embeddingText, type Chunk } from "../core/chunker";
import { joinVectorRows, packVectors, PackedVectors } from "../core/vectors";
import type { EmbeddingClient } from "../embedding/client";
import type { SearchMode, Settings } from "../settings";
import { IndexStorage, type FileRecord } from "./storage";
import spec from "../model-spec.json";

export interface SearchResult {chunk:Chunk;score:number}
export class SearchService {
  private lexical=new BM25([],true);
  private files:Record<string,FileRecord>={};
  private chunks=new Map<string,Chunk>();
  private vectors?:PackedVectors;
  private rows=new Map<string,Uint8Array>();
  private queue:Promise<void>=Promise.resolve();
  private storage:IndexStorage;
  private closed=false;
  private requested=0;
  private rebuildRequested=false;
  paused=false;
  status="Preparing search…";
  constructor(private app:App,private settings:()=>Settings,private embedding:EmbeddingClient,
    directory:string,private changed:()=>void) {this.storage=new IndexStorage(app.vault.adapter,directory);}

  private update(message:string):void {if(this.closed)return;this.status=message;this.changed();}
  get chunkCount():number {return this.chunks.size;}
  get vectorCount():number {return this.rows.size;}

  async load():Promise<void> {
    const saved=await this.storage.load();
    if(saved) {
      try {
        for(const [path,record] of Object.entries(saved.snapshot.files)) {
          if(!record || !Array.isArray(record.chunks) || !record.chunks.every(c=>typeof c.text==="string" && c.path===path && Array.isArray(c.headings)))
            throw new Error("Invalid cached chunks.");
          this.files[path]=record;
          for(const chunk of record.chunks) {this.chunks.set(chunk.id,chunk);this.lexical.add(chunkDocument(chunk));}
        }
        if(saved.snapshot.modelRevision===spec.revision && saved.snapshot.codec===this.settings().codec) {
          this.vectors=new PackedVectors(saved.vectors,saved.snapshot.vectorIds);
          saved.snapshot.vectorIds.forEach((id,row)=>{if(this.chunks.has(id))this.rows.set(id,this.vectors!.rowBytes(row));});
        }
      } catch {
        this.files={};this.chunks.clear();this.rows.clear();this.vectors=undefined;this.lexical=new BM25([],true);
      }
    }
    this.update(`${this.chunks.size} sections ready`);
  }

  refresh(rebuild=false):Promise<void> {
    this.rebuildRequested ||= rebuild;
    const serial=++this.requested;
    const task=this.queue.catch(()=>{}).then(()=>this.synchronize(serial));
    this.queue=task;
    return task;
  }

  private included(file:TFile):boolean {
    const path=file.path;
    return !this.settings().excludedFolders.some(folder=>path===folder || path.startsWith(`${folder}/`));
  }
  private remove(path:string):void {
    for(const chunk of this.files[path]?.chunks??[]) {
      this.lexical.remove(chunk.id);this.chunks.delete(chunk.id);this.rows.delete(chunk.id);
    }
    delete this.files[path];
  }

  private async synchronize(serial:number):Promise<void> {
    if(this.closed || this.paused || serial!==this.requested)return;
    if(this.rebuildRequested) {this.rebuildRequested=false;this.files={};this.chunks.clear();this.rows.clear();this.vectors=undefined;this.lexical=new BM25([],true);}
    const files=this.app.vault.getMarkdownFiles().filter(file=>this.included(file));
    const live=new Set(files.map(file=>file.path));
    for(const path of Object.keys(this.files)) if(!live.has(path))this.remove(path);
    this.update("Updating note sections…");
    for(let i=0;i<files.length;i++) {
      if(this.closed || this.paused || serial!==this.requested)return;
      const file=files[i],previous=this.files[file.path];
      if(previous && previous.mtime===file.stat.mtime && previous.size===file.stat.size)continue;
      // Capture before reading: an edit during indexing is picked up by the next event.
      const mtime=file.stat.mtime,size=file.stat.size;
      const content=await this.app.vault.read(file);
      if(!this.app.vault.getFileByPath(file.path))continue;
      const chunks=chunkMarkdown(file.path,content);
      this.remove(file.path);
      this.files[file.path]={mtime,size,chunks};
      for(const chunk of chunks) {this.chunks.set(chunk.id,chunk);this.lexical.add(chunkDocument(chunk));}
      if(i%20===0)await new Promise(resolve=>setTimeout(resolve,0));
    }
    const settings={...this.settings()};
    if(this.vectors && this.vectors.codec!==settings.codec)this.rows.clear();
    // Keep keyword search recoverable even if model startup fails or takes a long time.
    this.repack(settings);await this.save(settings);
    if(settings.semanticEnabled) {
      const missing=[...this.chunks.values()].filter(chunk=>!this.rows.has(chunk.id));
      for(let i=0;i<missing.length;i+=2) {
        if(this.closed || this.paused || serial!==this.requested || !this.settings().semanticEnabled)break;
        this.update(`Embedding sections ${i+1}–${Math.min(i+2,missing.length)} of ${missing.length}…`);
        const batch=missing.slice(i,i+2);
        const vectors=await this.embedding.embed(batch.map(embeddingText));
        const packed=new PackedVectors(packVectors(vectors,settings.codec),batch.map(chunk=>chunk.id));
        batch.forEach((chunk,row)=>this.rows.set(chunk.id,packed.rowBytes(row)));
        // Yield between small batches so interactive search can enter the worker queue.
        await new Promise(resolve=>setTimeout(resolve,10));
        if(i%100===0) {this.repack(settings);await this.save(settings);}
      }
    }
    if(this.closed)return;
    this.repack(settings);await this.save(settings);
    this.update(this.paused ? "Indexing paused" : `${this.chunks.size} sections ready${settings.semanticEnabled ? ` · ${this.rows.size} embedded` : " · BM25"}`);
  }

  private repack(settings:Settings):void {
    const entries=[...this.rows].filter(([id])=>this.chunks.has(id));
    this.rows=new Map(entries);
    this.vectors=new PackedVectors(joinVectorRows(entries.map(([,row])=>row),settings.codec),entries.map(([id])=>id));
  }
  private async save(settings:Settings):Promise<void> {
    await this.storage.save({version:1,modelRevision:spec.revision,codec:settings.codec,files:this.files,vectorIds:this.vectors?.ids??[]},
      this.vectors?.bytes??packVectors([],settings.codec));
  }

  async search(query:string,mode:SearchMode=this.settings().mode):Promise<SearchResult[]> {
    if(!query.trim())return [];
    const settings=this.settings();
    const group=(id:string)=>this.chunks.get(id)?.path??id;
    const lexical=this.lexical.search(query,100,group);
    let ranking=lexical;
    if(mode!=="bm25") {
      if(!settings.semanticEnabled)throw new Error("Enable semantic search in Heading Search settings, or choose BM25.");
      if(!this.vectors?.count)throw new Error("Semantic indexing is still starting. BM25 search is ready now.");
      const [embedding]=await this.embedding.embed([query],true);
      const dense=this.vectors.search(embedding,100,group);
      if(mode==="hybrid") {
        const sections=new Map(lexical.map(entry=>[group(entry.id),entry.id]));
        for(const entry of dense)if(!sections.has(group(entry.id)))sections.set(group(entry.id),entry.id);
        const notes=(list:typeof lexical)=>list.map(entry=>({id:group(entry.id),score:entry.score}));
        ranking=reciprocalRankFusion([notes(lexical),notes(dense)],100).map(entry=>({id:sections.get(entry.id)!,score:entry.score}));
      } else ranking=dense;
    }
    // Display the strongest section per note. Each result opens at its source line.
    const paths=new Set<string>(),results:SearchResult[]=[];
    for(const entry of ranking) {
      const chunk=this.chunks.get(entry.id);
      if(!chunk || paths.has(chunk.path) || settings.excludedFolders.some(folder=>chunk.path===folder || chunk.path.startsWith(`${folder}/`)) || !this.app.vault.getFileByPath(chunk.path))continue;
      paths.add(chunk.path);results.push({chunk,score:entry.score});
      if(results.length>=settings.resultCount)break;
    }
    return results;
  }
  dispose():void {this.closed=true;this.embedding.dispose();}
}
