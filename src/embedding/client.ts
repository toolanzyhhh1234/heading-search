import type { ModelStore } from "./model-store";

export class EmbeddingClient {
  private worker?:Worker;
  private pending=new Map<number,{resolve:(vectors:Float32Array[])=>void;reject:(error:Error)=>void}>();
  private next=0;
  private ready?:Promise<void>;
  private stopped=false;
  constructor(private store:ModelStore) {}

  initialize():Promise<void> {
    if(this.stopped) return Promise.reject(new Error("Search has been closed."));
    if(!this.ready) this.ready=this.start().catch(error=>{this.ready=undefined;this.terminate(error);throw error;});
    return this.ready;
  }
  private async start():Promise<void> {
    const payload=await this.store.read();
    if(this.stopped) throw new Error("Search has been closed.");
    const url=URL.createObjectURL(new Blob([EMBEDDING_WORKER_SOURCE],{type:"application/javascript"}));
    try { this.worker=new Worker(url); } finally { URL.revokeObjectURL(url); }
    this.worker.onmessage=(event:MessageEvent)=>{
      const response=event.data,promise=this.pending.get(response.id);
      if(!promise) return;
      this.pending.delete(response.id);
      if(response.error) promise.reject(new Error(response.error));
      else promise.resolve((response.vectors??[]).map((buffer:ArrayBuffer)=>new Float32Array(buffer)));
    };
    this.worker.onerror=()=>{this.ready=undefined;this.terminate(new Error("The CPU search worker stopped. Try again or use BM25."));};
    await this.call("initialize",payload,[payload.model]);
  }
  private call(type:string,payload:object,transfer:Transferable[]=[]):Promise<Float32Array[]> {
    const id=++this.next;
    return new Promise((resolve,reject)=>{
      if(!this.worker) {reject(new Error("The search worker is unavailable."));return;}
      this.pending.set(id,{resolve,reject});
      this.worker.postMessage({id,type,...payload},transfer);
    });
  }
  async embed(texts:string[],query=false):Promise<Float32Array[]> {
    await this.initialize();
    return this.call("embed",{texts,query});
  }
  private terminate(error:Error) {
    this.worker?.terminate();this.worker=undefined;
    for(const promise of this.pending.values()) promise.reject(error);
    this.pending.clear();
  }
  dispose():void {this.stopped=true;this.ready=undefined;this.terminate(new Error("Search has been closed."));}
}
