import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import spec from "../model-spec.json";

export type Download = (url: string) => Promise<ArrayBuffer>;
export type ModelProgress = (message: string) => void;
export type ModelDefinition = {id:string;revision:string;files:{path:string;bytes:number;sha256:string}[]};
export const sharedModelRoot = ():string => join(homedir(), ".cache", "heading-search", "models");
const digest = (data:Uint8Array) => createHash("sha256").update(data).digest("hex");
async function ensureDirectory(path:string):Promise<void> {
  for(let attempt=0;attempt<5;attempt++) {
    try {await fs.mkdir(path,{recursive:true});return;}
    catch(error) {
      // Concurrent recursive mkdir can transiently return EPERM on Windows.
      try {if((await fs.stat(path)).isDirectory())return;} catch {}
      if((error as NodeJS.ErrnoException).code!=="EPERM" || attempt===4)throw error;
      await new Promise(resolve=>setTimeout(resolve,25));
    }
  }
}

/** One cache per OS user, independent of vault paths and plugin installations. */
export class ModelStore {
  readonly directory:string;
  constructor(private download:Download, root=sharedModelRoot(),private definition:ModelDefinition=spec) {
    this.directory=join(root,...definition.id.split("/"),definition.revision);
  }

  async isInstalled():Promise<boolean> {
    for(const file of this.definition.files) {
      try {
        const data=await fs.readFile(join(this.directory,file.path));
        if(data.length!==file.bytes || digest(data)!==file.sha256) return false;
      } catch { return false; }
    }
    return true;
  }

  async ensure(progress:ModelProgress=()=>{}):Promise<void> {
    if(await this.isInstalled()) {progress("Using the shared model already on this computer.");return;}
    await ensureDirectory(this.directory);
    const lock=join(this.directory,".download-lock");
    const started=Date.now();
    while(true) {
      try { await fs.mkdir(lock);break; }
      catch(error) {
        if((error as NodeJS.ErrnoException).code!=="EEXIST") throw error;
        // A crashed process stops touching the lease. Active downloads keep it fresh.
        try {
          const stat=await fs.stat(lock);
          if(Date.now()-stat.mtimeMs>120_000) {await fs.rmdir(lock).catch(()=>{});continue;}
        } catch {}
        if(Date.now()-started>600_000) throw new Error("Another vault is still downloading the model. Try again shortly.");
        progress("Waiting for another vault to finish the shared download…");
        await new Promise(resolve=>setTimeout(resolve,500));
      }
    }
    const heartbeat=setInterval(()=>{const now=new Date();void fs.utimes(lock,now,now).catch(()=>{});},10_000);
    try {
      // Recheck after taking the lock: a different vault may have finished first.
      if(await this.isInstalled()) return;
      for(const file of this.definition.files) {
        const target=join(this.directory,file.path);
        try {
          const data=await fs.readFile(target);
          if(data.length===file.bytes && digest(data)===file.sha256) continue;
        } catch {}
        progress(`Downloading ${file.path.split("/").pop()} (${(file.bytes/1e6).toFixed(1)} MB)…`);
        const url=`https://huggingface.co/${this.definition.id}/resolve/${this.definition.revision}/${file.path}`;
        const data=new Uint8Array(await this.download(url));
        if(data.length!==file.bytes || digest(data)!==file.sha256) throw new Error(`Integrity check failed for ${file.path}. Please retry setup.`);
        await ensureDirectory(dirname(target));
        const temporary=`${target}.${randomUUID()}.partial`;
        try {await fs.writeFile(temporary,data,{flag:"wx"});await fs.rename(temporary,target);}
        finally {await fs.unlink(temporary).catch(()=>{});}
      }
      progress("Shared model ready. It can now be used offline in every vault.");
    } finally {clearInterval(heartbeat);await fs.rmdir(lock).catch(()=>{});}
  }

  async read():Promise<{model:ArrayBuffer;tokenizer:object;config:object}> {
    if(!await this.isInstalled()) throw new Error("Set up semantic search in Heading Search settings first.");
    const model=await fs.readFile(join(this.directory,"onnx/model_quantized.onnx"));
    const tokenizer=JSON.parse(await fs.readFile(join(this.directory,"tokenizer.json"),"utf8"));
    const config=JSON.parse(await fs.readFile(join(this.directory,"tokenizer_config.json"),"utf8"));
    return {model:Uint8Array.from(model).buffer,tokenizer,config};
  }
}
