import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { randomUUID } from "node:crypto";
import spec from "../model-spec.json";
import { ensureDirectory, promote, sha256, stampMatches, verifyFiles, withDownloadLock, writeStamp, type Download, type PinnedFile, type Progress } from "./shared-cache";

export type { Download } from "./shared-cache";
export type ModelProgress = Progress;
export type ModelWeights = keyof typeof spec.weights;
type SpecFile = {path:string;bytes:number;sha256:string};
export type ModelDefinition = {id:string;revision:string;files:SpecFile[];model?:string;
  /** ONNX external-data files beside the graph; the native runtime loads them from the model's folder. */
  externalData?:string[]};
/** Identifies the shipped model and revision, for choices that were measured with it. */
export const modelKey=`${spec.id}@${spec.revision}`;
export const sharedModelRoot = ():string => join(homedir(), ".cache", "heading-search", "models");
/** The tokenizer files plus one ONNX weight variant and its external-data files. */
export function modelDefinition(weights:ModelWeights):ModelDefinition {
  const variant:SpecFile&{extra?:SpecFile[]}=spec.weights[weights];
  const extra=variant.extra ?? [];
  return {id:spec.id,revision:spec.revision,files:[...spec.files,variant,...extra],model:variant.path,externalData:extra.map(file=>file.path)};
}
export const weightsBytes=(weights:ModelWeights):number=>modelDefinition(weights).files.reduce((sum,file)=>sum+file.bytes,0);

/** One cache per OS user, independent of vault paths and plugin installations. */
export class ModelStore {
  readonly directory:string;
  private stamp:string;
  constructor(private download:Download, root=sharedModelRoot(),private definition:ModelDefinition=modelDefinition("int8")) {
    this.directory=join(root,...definition.id.split("/"),definition.revision);
    this.stamp=join(this.directory,`.verified-${sha256(new TextEncoder().encode(definition.files.map(f=>f.sha256).join()))}.json`);
  }
  private pinned():PinnedFile[] {return this.definition.files.map(file=>({target:join(this.directory,file.path),bytes:file.bytes,sha256:file.sha256}));}
  get modelPath():string {return join(this.directory,this.definition.model ?? this.definition.files[this.definition.files.length-1].path);}

  /** Cheap startup check against the stamp written after the last full verification. */
  isInstalled():Promise<boolean> {return stampMatches(this.stamp,this.pinned());}

  async ensure(progress:ModelProgress=()=>{}):Promise<void> {
    if(await verifyFiles(this.pinned())) {await writeStamp(this.stamp,this.pinned());progress("Using the shared model already on this computer.");return;}
    await withDownloadLock(this.directory,progress,"model",async()=>{
      for(const file of this.definition.files) {
        const target=join(this.directory,file.path);
        // A different vault may have finished first, and earlier partial setups keep their valid files.
        if(await verifyFiles([{target,bytes:file.bytes,sha256:file.sha256}]))continue;
        progress(`Downloading ${file.path.split("/").pop()} (${(file.bytes/1e6).toFixed(1)} MB)…`);
        const url=`https://huggingface.co/${this.definition.id}/resolve/${this.definition.revision}/${file.path}`;
        const data=new Uint8Array(await this.download(url));
        if(data.length!==file.bytes || sha256(data)!==file.sha256) throw new Error(`Integrity check failed for ${file.path}. Please retry setup.`);
        await ensureDirectory(dirname(target));
        const temporary=`${target}.${randomUUID()}.partial`;
        try {await fs.writeFile(temporary,data,{flag:"wx"});await promote(temporary,target);}
        finally {await fs.unlink(temporary).catch(()=>{});}
      }
      await writeStamp(this.stamp,this.pinned());
      progress("Shared model ready. It can now be used offline in every vault.");
    });
  }

  async read():Promise<{modelPath:string;tokenizer:object;config:object}> {
    if(!await this.isInstalled()) throw new Error("The semantic search model is not installed, or this version of Heading Search uses a new one. Open Heading Search settings and choose Check / repair.");
    const tokenizer=JSON.parse(await fs.readFile(join(this.directory,"tokenizer.json"),"utf8"));
    const config=JSON.parse(await fs.readFile(join(this.directory,"tokenizer_config.json"),"utf8"));
    return {modelPath:this.modelPath,tokenizer,config};
  }
  /** Delete the weights files (not the shared tokenizer). Silently kept if another vault has them open. */
  async removeWeights():Promise<void> {
    await fs.rm(this.stamp,{force:true}).catch(()=>{});
    for(const path of [this.modelPath,...this.externalDataPaths]) await fs.rm(path,{force:true}).catch(()=>{});
  }
  private get externalDataPaths():string[] {return (this.definition.externalData ?? []).map(path=>join(this.directory,path));}
}
