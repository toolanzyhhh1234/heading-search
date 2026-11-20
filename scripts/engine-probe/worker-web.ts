// One onnxruntime-web build per worker; "ort-variant" is aliased to a dist bundle at build time.
import * as ort from "ort-variant";
import { bench, describeError } from "./common";
import { EmbeddingEngine, createTokenizer } from "../../src/embedding/engine";

self.onmessage=async(event:MessageEvent)=>{
  const {job,wasm,model,external,externalName,threads,ep,profile}=event.data;
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
    // Kernel profile. env.webgpu.profiling only reports for the JSEP engine; the native WebGPU EP reports through the
    // session profiler, which prints a Chrome trace to the console at endProfiling(), one event per line. Each Node event's
    // "dur" is host time inside the kernel's Compute(): for the WebGPU EP that is recording and dispatching GPU work, not
    // the GPU's own execution time. Summed against the pass's wall time, it shows whether dispatch or the GPU is the limit.
    type Agg={ms:number;count:number};
    const byOperator:Record<string,Agg>={},nonJson:string[]=[];
    let traceEvents=0,profiledFromUs=Infinity;
    if(profile) {
      const original=console.log.bind(console);
      console.log=(...args:unknown[])=>{
        for(const line of args.map(String).join(" ").split("\n")) {
          const text=line.trim().replace(/,$/,"");
          if(!text.startsWith("{")) {if(text && text!=="[" && text!=="]" && nonJson.length<10) nonJson.push(text.slice(0,300));continue;}
          try {
            const e=JSON.parse(text) as {cat?:string;name?:string;dur?:number;ts?:number;args?:{op_name?:string;provider?:string}};
            traceEvents++;
            if(e.cat!=="Node" || !e.args?.op_name || (e.ts??0)<profiledFromUs) continue;
            const key=`${e.args.op_name} @ ${(e.args.provider??"?").replace("ExecutionProvider","")}`;
            const a=byOperator[key]??={ms:0,count:0};
            a.ms+=(e.dur??0)/1000;a.count++;
          } catch {if(nonJson.length<10) nonJson.push(text.slice(0,300));}
        }
        if(!args.some(a=>String(a).trimStart().startsWith("{\"cat\""))) original(...args);
      };
    }
    const start=performance.now();
    const session=await o.InferenceSession.create(new Uint8Array(model),{executionProviders:[ep],graphOptimizationLevel:"all",
      externalData:[{path:externalName,data:new Uint8Array(external)}],...(profile?{enableProfiling:true}:{})});
    const loadMs=performance.now()-start;
    env.threadsUsed=o.env.wasm.numThreads;
    if(profile) {
      // One warm-up pass (shader compilation), then one profiled pass. The trace's "ts" counts microseconds from session
      // creation, so events before the profiled pass started are left out of the totals.
      const engine=new EmbeddingEngine(ort as never,session as never,createTokenizer(job.tokenizer,job.config),job.limits,job.model);
      const pass=async()=>{for(let i=0;i<job.docs.length;i+=job.chunk) await engine.embed(job.docs.slice(i,i+job.chunk));};
      await pass();
      const passStart=performance.now();
      profiledFromUs=(passStart-start)*1000;
      await pass();
      const passMs=performance.now()-passStart;
      try {(session as unknown as {endProfiling():void}).endProfiling();} catch {}
      await session.release();
      const hostMs=Object.values(byOperator).reduce((s,a)=>s+a.ms,0);
      const operators=Object.fromEntries(Object.entries(byOperator).sort((a,b)=>b[1].ms-a[1].ms)
        .map(([key,a])=>[key,{hostMs:Math.round(a.ms*10)/10,share:hostMs?Math.round(a.ms/hostMs*1000)/10:null,count:a.count}]));
      self.postMessage({ok:true,env,loadMs,profile:{traceEvents,passMs:Math.round(passMs),passMsPerDoc:Math.round(passMs/job.docs.length*10)/10,
        hostMsInKernels:Math.round(hostMs),hostShareOfPass:passMs?Math.round(hostMs/passMs*1000)/10:null,operators,nonJson}});
      return;
    }
    const result=await bench(ort as never,session as never,job);
    await session.release();
    self.postMessage({ok:true,env,loadMs,...result});
  } catch(error) { self.postMessage({ok:false,env,error:describeError(error)}); }
};
