import type { DataAdapter } from "obsidian";
import { randomUUID, createHash } from "node:crypto";
import type { Chunk } from "../core/chunker";
import type { VectorCodec } from "../core/vectors";

export interface FileRecord {mtime:number;size:number;chunks:Chunk[]}
export interface Snapshot {
  version:1;
  modelRevision:string;
  codec:VectorCodec;
  files:Record<string,FileRecord>;
  vectorIds:string[];
}

/** Publish a complete immutable generation, then change the pointer. */
export class IndexStorage {
  constructor(private adapter:DataAdapter,private directory:string) {}
  async load():Promise<{snapshot:Snapshot;vectors:Uint8Array}|null> {
    try {
      const pointer=JSON.parse(await this.adapter.read(`${this.directory}/current.json`));
      if(typeof pointer.generation!=="string" || !/^[a-f0-9-]{36}$/.test(pointer.generation)) return null;
      const base=`${this.directory}/${pointer.generation}`;
      const snapshot=JSON.parse(await this.adapter.read(`${base}.json`)) as Snapshot;
      if(snapshot.version!==1 || !snapshot.files || !Array.isArray(snapshot.vectorIds)) return null;
      const vectors=new Uint8Array(await this.adapter.readBinary(`${base}.bin`));
      const expected=(snapshot as Snapshot & {vectorHash?:string}).vectorHash;
      if(expected!==createHash("sha256").update(vectors).digest("hex"))return null;
      return {snapshot,vectors};
    } catch {return null;}
  }
  async save(snapshot:Snapshot,vectors:Uint8Array):Promise<void> {
    if(!await this.adapter.exists(this.directory)) await this.adapter.mkdir(this.directory);
    const generation=randomUUID(),base=`${this.directory}/${generation}`;
    await this.adapter.write(`${base}.json`,JSON.stringify({...snapshot,vectorHash:createHash("sha256").update(vectors).digest("hex")}));
    await this.adapter.writeBinary(`${base}.bin`,Uint8Array.from(vectors).buffer);
    // If a pointer write is interrupted, load() discards the cache and notes rebuild it.
    await this.adapter.write(`${this.directory}/current.json`,JSON.stringify({generation}));
    const listing=await this.adapter.list(this.directory);
    for(const file of listing.files) {
      if(/\/[a-f0-9-]{36}\.(?:json|bin)$/.test(file) && file!==`${base}.json` && file!==`${base}.bin`)
        await this.adapter.remove(file).catch(()=>{});
    }
  }
}
