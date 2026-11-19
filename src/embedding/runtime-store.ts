import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import runtimeSpec from "../runtime-spec.json";
import { ensureDirectory, promote, stampMatches, verifyFiles, withDownloadLock, writeStamp, type Download, type PinnedFile, type Progress } from "./shared-cache";
import { extractTarGz } from "./tar";

export interface RuntimeEntry {archive:string;path:string;sha256:string;bytes:number}
export interface RuntimeSpec {
  version:string;
  archives:Record<string,{url:string;sha512:string;bytes:number}>;
  shared:RuntimeEntry[];
  platforms:Record<string,{gpu:string|null;files:RuntimeEntry[]}>;
}
export const sharedRuntimeRoot=():string=>join(homedir(),".cache","heading-search","runtime");

/**
 * The native ONNX Runtime, from Microsoft's official npm packages. Shared by every vault, like the model.
 * Only this platform's binaries are extracted, and every file is checked against a pinned SHA-256.
 */
export class RuntimeStore {
  readonly directory:string;
  readonly platform:string;
  private stamp:string;
  constructor(private download:Download,root=sharedRuntimeRoot(),private spec:RuntimeSpec=runtimeSpec,
    platform:string=process.platform,arch:string=process.arch) {
    this.platform=`${platform}-${arch}`;
    this.directory=join(root,`onnxruntime-node-${spec.version}-${this.platform}`);
    this.stamp=join(this.directory,".verified.json");
  }
  get supported():boolean {return this.platform in this.spec.platforms;}
  get gpu():string|null {return this.spec.platforms[this.platform]?.gpu ?? null;}
  get downloadBytes():number {return Object.values(this.spec.archives).reduce((sum,archive)=>sum+archive.bytes,0);}
  private entries():RuntimeEntry[] {return [...this.spec.shared,...(this.spec.platforms[this.platform]?.files??[])];}
  private target(base:string,entry:RuntimeEntry):string {return join(base,"node_modules",entry.archive,...entry.path.replace(/^package\//,"").split("/"));}
  private pinned(base=this.directory):PinnedFile[] {return this.entries().map(entry=>({target:this.target(base,entry),bytes:entry.bytes,sha256:entry.sha256}));}

  isInstalled():Promise<boolean> {return this.supported ? stampMatches(this.stamp,this.pinned()) : Promise.resolve(false);}

  async ensure(progress:Progress=()=>{}):Promise<void> {
    if(!this.supported)throw new Error(`The fast engine is not available for ${this.platform}.`);
    if(await verifyFiles(this.pinned())) {await writeStamp(this.stamp,this.pinned());progress("Using the fast engine already on this computer.");return;}
    const root=dirname(this.directory);
    await withDownloadLock(root,progress,"fast engine",async()=>{
      if(await verifyFiles(this.pinned())) {await writeStamp(this.stamp,this.pinned());return;}
      const staging=`${this.directory}.${randomUUID()}.partial`;
      try {
        for(const [name,archive] of Object.entries(this.spec.archives)) {
          const wanted=new Map(this.entries().filter(entry=>entry.archive===name).map(entry=>[entry.path,entry]));
          progress(`Downloading the fast engine: ${name} (${(archive.bytes/1e6).toFixed(0)} MB)…`);
          const data=new Uint8Array(await this.download(archive.url));
          if(data.length!==archive.bytes || createHash("sha512").update(data).digest("base64")!==archive.sha512)
            throw new Error(`Integrity check failed for ${name}. Please retry setup.`);
          progress(`Unpacking ${name}…`);
          const files=await extractTarGz(data,path=>wanted.has(path));
          for(const entry of wanted.values()) {
            const bytes=files.get(entry.path);
            if(!bytes || bytes.length!==entry.bytes || createHash("sha256").update(bytes).digest("hex")!==entry.sha256)
              throw new Error(`Integrity check failed for ${entry.path.split("/").pop()}. Please retry setup.`);
            const target=this.target(staging,entry);
            await ensureDirectory(dirname(target));
            await fs.writeFile(target,bytes,{flag:"wx"});
          }
        }
        if(!await verifyFiles(this.pinned(staging)))throw new Error("The fast engine could not be written completely. Check free disk space and retry.");
        // Replace a damaged copy. Windows refuses while another vault has its libraries loaded.
        await fs.rm(this.directory,{recursive:true,force:true}).catch(()=>{
          throw new Error("The fast engine is in use by another vault. Close other vaults, then retry.");
        });
        await promote(staging,this.directory);
        await writeStamp(this.stamp,this.pinned());
        progress("Fast engine ready.");
      } finally {await fs.rm(staging,{recursive:true,force:true}).catch(()=>{});}
    });
  }
}
