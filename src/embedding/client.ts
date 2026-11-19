import type { ModelStore, ModelWeights } from "./model-store";
import type { RuntimeStore } from "./runtime-store";
import type { BackendInfo, Calibration } from "./backends";
import type { DeviceProfile } from "./device-profile";
import type { ModelConfig } from "./engine";

type Response={info?:BackendInfo;vectors?:ArrayBuffer[];calibration?:Calibration};

/** One embedding worker and its request/response bookkeeping. */
class Channel {
  private worker:Worker;
  private pending=new Map<number,{resolve:(response:Response)=>void;reject:(error:Error)=>void}>();
  private next=0;
  constructor(onCrash:()=>void) {
    const url=URL.createObjectURL(new Blob([EMBEDDING_WORKER_SOURCE],{type:"application/javascript"}));
    try { this.worker=new Worker(url); } finally { URL.revokeObjectURL(url); }
    this.worker.onmessage=(event:MessageEvent)=>{
      const response=event.data,promise=this.pending.get(response.id);
      if(!promise) return;
      this.pending.delete(response.id);
      if(response.error) promise.reject(new Error(response.error));
      else promise.resolve(response);
    };
    this.worker.onerror=()=>{this.close(new Error("The search worker stopped. Try again or use BM25."));onCrash();};
  }
  call(type:string,payload:object,transfer:Transferable[]=[]):Promise<Response> {
    const id=++this.next;
    return new Promise((resolve,reject)=>{
      this.pending.set(id,{resolve,reject});
      this.worker.postMessage({id,type,...payload},transfer);
    });
  }
  /**
   * Ask the worker to release its native sessions before terminating it, so a busy GPU batch finishes cleanly.
   * Callers stop waiting immediately; the worker is terminated on reply or after a few seconds.
   */
  close(error:Error):void {
    for(const promise of this.pending.values()) promise.reject(error);
    this.pending.clear();
    const worker=this.worker,timer=setTimeout(()=>worker.terminate(),3000);
    worker.onmessage=()=>{clearTimeout(timer);worker.terminate();};
    worker.onerror=()=>{clearTimeout(timer);worker.terminate();};
    try {worker.postMessage({id:0,type:"dispose"});} catch {clearTimeout(timer);worker.terminate();}
  }
}

/**
 * ONNX Runtime keeps one environment per process, shared by every thread that loads it, and destroys it for good
 * when the last such thread exits. Obsidian's workers share the window's process, so the window itself keeps a
 * reference: restarting or reloading the engine then never destroys it. Node's module cache makes this a no-op after the first call.
 */
function holdNativeRuntime(directory:string):void {
  const require=(globalThis as {require?:(id:string)=>unknown}).require;
  if(typeof require!=="function")throw new Error("This Obsidian build does not allow native modules.");
  require(`${directory}/node_modules/onnxruntime-node`);
}

export interface ClientOptions {threads:number;profile:()=>DeviceProfile;model:ModelConfig;changed?:()=>void}

/** Runs the native engine in a worker. Semantic search needs it; keyword search works without it. */
export class EmbeddingClient {
  private channel?:Channel;
  private ready?:Promise<void>;
  private stopped=false;
  info?:BackendInfo;
  /** Why the engine is not running as chosen: a GPU that fell back to the CPU, or a native engine that could not start. */
  fallbackReason?:string;
  constructor(private models:(weights:ModelWeights)=>ModelStore,private runtime:RuntimeStore,private options:ClientOptions) {}

  get batchTexts():number {return this.info?.batchTexts ?? 2;}
  /** Forward slashes: the path is used as a require() specifier. */
  private get runtimeDirectory():string {return this.runtime.directory.replaceAll("\\","/");}
  initialize():Promise<void> {
    if(this.stopped) return Promise.reject(new Error("Search has been closed."));
    if(!this.ready) this.ready=this.start().catch(error=>{this.ready=undefined;this.close(error);throw error;});
    return this.ready;
  }
  private open():Channel {
    this.close(new Error("The embedding engine is restarting."));
    const channel=new Channel(()=>{if(this.channel===channel){this.channel=undefined;this.ready=undefined;}});
    this.channel=channel;
    return channel;
  }
  private async start():Promise<void> {
    const profile=this.options.profile(),model=this.models(profile.weights);
    const assets=await model.read();
    if(this.stopped) throw new Error("Search has been closed.");
    const fail=(reason:string):never=>{
      this.fallbackReason=`${reason} Keyword (BM25) search still works.`;this.options.changed?.();
      throw new Error(this.fallbackReason);
    };
    if(!this.runtime.supported) fail(`Semantic search is not available for ${this.runtime.platform}.`);
    if(!await this.runtime.isInstalled()) fail("The fast engine is missing or damaged. Open Heading Search settings and choose Check / repair.");
    try {
      holdNativeRuntime(this.runtimeDirectory);
      const {info}=await this.open().call("initialize-native",{runtimeDirectory:this.runtimeDirectory,modelPath:assets.modelPath,
        tokenizer:assets.tokenizer,config:assets.config,threads:this.options.threads,device:profile.device,model:this.options.model});
      this.info=info;this.fallbackReason=info?.fallback;this.options.changed?.();
    } catch(error) {
      // Security software or a missing system library can block native code.
      const message=error instanceof Error ? error.message : String(error);
      fail(/OrtEnv already destroyed/.test(message) ? "The fast engine was shut down in this Obsidian session. Restart Obsidian to use it again."
        : `The fast engine could not start: ${message}.`);
    }
  }
  async embed(texts:string[],query=false):Promise<Float32Array[]> {
    await this.initialize();
    if(!this.channel) throw new Error("The search worker is unavailable.");
    const {vectors}=await this.channel.call("embed",{texts,query});
    return (vectors??[]).map(buffer=>new Float32Array(buffer));
  }
  /** Times int8 on the CPU against fp32 on DirectML in a separate, short-lived worker. */
  async calibrate():Promise<Calibration> {
    holdNativeRuntime(this.runtimeDirectory);
    const {tokenizer,config}=await this.models("int8").read(),channel=new Channel(()=>{});
    try {
      const {calibration}=await channel.call("calibrate",{runtimeDirectory:this.runtimeDirectory,
        cpuModelPath:this.models("int8").modelPath,gpuModelPath:this.models("fp32").modelPath,threads:this.options.threads,tokenizer,config,model:this.options.model});
      return calibration!;
    } finally {channel.close(new Error("Calibration finished."));}
  }
  /** Start again on the next request, for example after setup changes the device or repairs the fast engine. */
  restart():void {this.ready=undefined;this.info=undefined;this.fallbackReason=undefined;this.close(new Error("The embedding engine is restarting."));this.options.changed?.();}
  private close(error:Error):void {this.channel?.close(error);this.channel=undefined;}
  dispose():void {this.stopped=true;this.ready=undefined;this.close(new Error("Search has been closed."));}
}
