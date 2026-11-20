import { promises as fs } from "node:fs";
import { homedir } from "node:os";
import { basename, join, dirname } from "node:path";
import { randomUUID } from "node:crypto";
import spec from "../model-spec.json";
import gpuGraph from "../assets/granite-small-r2.gpu.onnx";
import { ensureDirectory, promote, sha256, stampMatches, verifyFiles, withDownloadLock, writeStamp, type Download, type PinnedFile, type Progress } from "./shared-cache";

export type { Download } from "./shared-cache";
export type ModelProgress = Progress;
export type ModelWeights = keyof typeof spec.weights;
/** A pinned file. Bundled files ship inside the plugin and are installed beside the downloaded ones. */
type SpecFile = {path:string;bytes:number;sha256:string;bundled?:Uint8Array};
export type ModelDefinition = {id:string;revision:string;files:SpecFile[];model?:string;
  /** The graph DirectML runs when it differs from the CPU's; it shares the CPU graph's external weights. */
  gpuModel?:string;
  /** ONNX external-data files beside the graph; the native runtime loads them from the model's folder. */
  externalData?:string[]};
type Variant = Omit<SpecFile,"bundled">&{extra?:SpecFile[];gpu?:Omit<SpecFile,"bundled">&{bundled:string}};
/** Bundled assets by their spec path. scripts/patch-gpu-graph.py writes them and pins their hashes. */
const BUNDLED:Record<string,Uint8Array>={[spec.weights.fp32.gpu.bundled]:gpuGraph};
/** Identifies the shipped model and revision, for choices that were measured with it. */
export const modelKey=`${spec.id}@${spec.revision}`;
export const sharedModelRoot = ():string => join(homedir(), ".cache", "heading-search", "models");
/** The tokenizer files plus one ONNX weight variant, its external-data files, and its bundled GPU graph if any. */
export function modelDefinition(weights:ModelWeights):ModelDefinition {
  const variant:Variant=spec.weights[weights];
  const extra=variant.extra ?? [];
  const gpu:SpecFile[]=variant.gpu ? [{path:variant.gpu.path,bytes:variant.gpu.bytes,sha256:variant.gpu.sha256,bundled:BUNDLED[variant.gpu.bundled]}] : [];
  if(gpu.some(file=>!file.bundled))throw new Error(`${variant.gpu?.bundled} is not bundled with this build.`);
  return {id:spec.id,revision:spec.revision,files:[...spec.files,variant,...extra,...gpu],model:variant.path,gpuModel:variant.gpu?.path,externalData:extra.map(file=>file.path)};
}
/** Bytes downloaded for a variant; bundled files are already in the plugin. */
export const weightsBytes=(weights:ModelWeights):number=>modelDefinition(weights).files.reduce((sum,file)=>sum+(file.bundled ? 0 : file.bytes),0);

const OFFLINE="needs a download";

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
  get gpuModelPath():string {return this.definition.gpuModel ? join(this.directory,this.definition.gpuModel) : this.modelPath;}

  /** Cheap startup check against the stamp written after the last full verification. */
  isInstalled():Promise<boolean> {return stampMatches(this.stamp,this.pinned());}

  /**
   * Installs whatever is missing or corrupt. Offline, it only installs bundled files: an update can add one (such as
   * the GPU graph) beside weights already on disk, and must not need the network or a repair for that.
   */
  async ensure(progress:ModelProgress=()=>{},offline=false):Promise<void> {
    if(offline) {
      // Fail before touching the cache if a download would be needed, for example when setup never ran.
      for(const file of this.definition.files)
        if(!file.bundled && !(await fs.stat(join(this.directory,file.path)).then(stat=>stat.size===file.bytes,()=>false)))throw new Error(OFFLINE);
    }
    if(await verifyFiles(this.pinned())) {await writeStamp(this.stamp,this.pinned());progress("Using the shared model already on this computer.");return;}
    await withDownloadLock(this.directory,progress,"model",async()=>{
      for(const file of this.definition.files) {
        const target=join(this.directory,file.path);
        // A different vault may have finished first, and earlier partial setups keep their valid files.
        if(await verifyFiles([{target,bytes:file.bytes,sha256:file.sha256}]))continue;
        let data:Uint8Array;
        if(file.bundled) data=file.bundled;
        else {
          if(offline)throw new Error(OFFLINE);
          progress(`Downloading ${file.path.split("/").pop()} (${(file.bytes/1e6).toFixed(1)} MB)…`);
          const url=`https://huggingface.co/${this.definition.id}/resolve/${this.definition.revision}/${file.path}`;
          data=new Uint8Array(await this.download(url));
        }
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
  /** Installed, or completed offline from bundled files (see ensure). */
  async isReady():Promise<boolean> {
    if(await this.isInstalled())return true;
    try {await this.ensure(()=>{},true);return true;} catch {return false;}
  }

  async read():Promise<{modelPath:string;gpuModelPath:string;tokenizer:object;config:object}> {
    if(!await this.isReady()) throw new Error("The semantic search model is not installed, or this version of Heading Search uses a new one. Open Heading Search settings and choose Check / repair.");
    const tokenizer=JSON.parse(await fs.readFile(join(this.directory,"tokenizer.json"),"utf8"));
    const config=JSON.parse(await fs.readFile(join(this.directory,"tokenizer_config.json"),"utf8"));
    return {modelPath:this.modelPath,gpuModelPath:this.gpuModelPath,tokenizer,config};
  }
  /**
   * The graph and its external-data files as bytes, for the built-in engine: its worker cannot read files. External
   * data is named the way the graph refers to it (beside the graph, so by file name).
   */
  async readGraph():Promise<{graph:ArrayBuffer;externalData:{path:string;data:ArrayBuffer}[]}> {
    const bytes=async(path:string)=>{
      const data=await fs.readFile(path);
      return (data.byteOffset===0 && data.byteLength===data.buffer.byteLength ? data.buffer : data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength)) as ArrayBuffer;
    };
    return {graph:await bytes(this.modelPath),externalData:await Promise.all(this.externalDataPaths.map(async path=>({path:basename(path),data:await bytes(path)})))};
  }
  /** Delete the weights files (not the shared tokenizer). Silently kept if another vault has them open. */
  async removeWeights():Promise<void> {
    await fs.rm(this.stamp,{force:true}).catch(()=>{});
    for(const path of new Set([this.modelPath,this.gpuModelPath,...this.externalDataPaths])) await fs.rm(path,{force:true}).catch(()=>{});
  }
  private get externalDataPaths():string[] {return (this.definition.externalData ?? []).map(path=>join(this.directory,path));}
}
