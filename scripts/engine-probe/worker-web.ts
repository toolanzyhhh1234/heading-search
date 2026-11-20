// One onnxruntime-web build per worker; "ort-variant" is aliased to a dist bundle at build time.
import * as ort from "ort-variant";
import { bench, describeError } from "./common";

self.onmessage=async(event:MessageEvent)=>{
  const {job,wasm,model,external,externalName,threads,ep}=event.data;
  const env:Record<string,unknown>={crossOriginIsolated:(self as {crossOriginIsolated?:boolean}).crossOriginIsolated===true,
    sharedArrayBuffer:typeof SharedArrayBuffer!=="undefined",jspi:typeof (WebAssembly as {Suspending?:unknown}).Suspending==="function",
    webgpu:"gpu" in navigator};
  try {
    const gpu=(navigator as {gpu?:{requestAdapter(o?:object):Promise<{info?:Record<string,string>;features:Set<string>}|null>}}).gpu;
    if(ep==="webgpu" && gpu) {
      const adapter=await gpu.requestAdapter({powerPreference:"high-performance"});
      env.adapter=adapter ? {vendor:adapter.info?.vendor,architecture:adapter.info?.architecture,description:adapter.info?.description,shaderF16:adapter.features.has("shader-f16")} : null;
    }
    const o=ort as unknown as {env:{wasm:Record<string,unknown>;webgpu?:Record<string,unknown>;logLevel?:string};
      InferenceSession:{create(m:Uint8Array,o:object):Promise<{release():Promise<void>}>}};
    o.env.logLevel="error";
    o.env.wasm.wasmBinary=wasm;
    o.env.wasm.numThreads=threads;
    if(ep==="webgpu" && o.env.webgpu) try {o.env.webgpu.powerPreference="high-performance";} catch {}
    const start=performance.now();
    const session=await o.InferenceSession.create(new Uint8Array(model),{executionProviders:[ep],graphOptimizationLevel:"all",
      externalData:[{path:externalName,data:new Uint8Array(external)}]});
    const loadMs=performance.now()-start;
    env.threadsUsed=o.env.wasm.numThreads;
    const result=await bench(ort as never,session as never,job);
    await session.release();
    self.postMessage({ok:true,env,loadMs,...result});
  } catch(error) { self.postMessage({ok:false,env,error:describeError(error)}); }
};
