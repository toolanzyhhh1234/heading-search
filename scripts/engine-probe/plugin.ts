// Development probe, not part of Heading Search: times onnxruntime-web (WASM and WebGPU builds) against the shipped
// native engine inside Obsidian, on the same texts, and checks that the vectors agree. Built by scripts/engine-probe/build.mjs.
import { Notice, Plugin } from "obsidian";

interface Probe {suite?:"engines"|"lowbit";runtimeDirectory:string;ortWeb:string;modelDirectory:string;docs:string[];queries:string[];model:object;
  weights:Record<string,{path:string;external:string;gpu?:string}>}
type Weights="int8"|"fp32"|"fp16"|"q4"|"q4f16";
interface Config {label:string;engine:"native"|"web";worker:string;weights:Weights;graph?:"upstream"|"gpu";device?:string;
  ep?:string;wasm?:string;threads:number;batch:"cpu"|"standard"|"large";
  /** One extra pass with onnxruntime-web's per-kernel GPU timing, summed by operator type. */
  profile?:boolean}
const BATCHES={cpu:{limits:{tokens:512,rows:1},chunk:16},standard:{limits:{tokens:2048,rows:8,fixedShapes:true},chunk:16},
  large:{limits:{tokens:16384,rows:64,fixedShapes:true},chunk:64}};
const WASM={plain:"ort-wasm-simd-threaded.wasm",jsep:"ort-wasm-simd-threaded.jsep.wasm",asyncify:"ort-wasm-simd-threaded.asyncify.wasm",jspi:"ort-wasm-simd-threaded.jspi.wasm"};
const threads=Math.min(4,Math.max(1,navigator.hardwareConcurrency-1));
const jspi=(weights:Weights,profile=false):Config=>({label:`web WebGPU (native EP, JSPI) ${weights}, standard batches${profile?", kernel profile":""}`,
  engine:"web",worker:"jspi",weights,graph:"upstream",ep:"webgpu",wasm:WASM.jspi,threads:1,batch:"standard",profile});
/**
 * Integrated graphics (Iris Xe) ran fp32 WebGPU 3x slower than DirectML: do smaller weights close the gap, and where
 * does the time go? Native DirectML rows of each low-bit variant are the vector references for the WebGPU rows.
 */
const LOWBIT:Config[]=[
  {label:"native CPU fp32 (reference)",engine:"native",worker:"native",weights:"fp32",device:"cpu",threads,batch:"cpu"},
  {label:"native CPU int8 (shipped CPU path)",engine:"native",worker:"native",weights:"int8",device:"cpu",threads,batch:"cpu"},
  {label:"native DirectML fp32, standard batches",engine:"native",worker:"native",weights:"fp32",graph:"gpu",device:"dml",threads,batch:"standard"},
  {label:"native DirectML q4f16, standard batches (reference)",engine:"native",worker:"native",weights:"q4f16",device:"dml",threads,batch:"standard"},
  {label:"native DirectML fp16, standard batches (reference)",engine:"native",worker:"native",weights:"fp16",device:"dml",threads,batch:"standard"},
  jspi("fp32"),jspi("fp16"),jspi("q4"),jspi("q4f16"),jspi("fp32",true),jspi("q4f16",true),
];
const ENGINES:Config[]=[
  {label:"native CPU fp32 (reference)",engine:"native",worker:"native",weights:"fp32",device:"cpu",threads,batch:"cpu"},
  {label:"native CPU int8 (shipped CPU path)",engine:"native",worker:"native",weights:"int8",device:"cpu",threads,batch:"cpu"},
  {label:"native DirectML fp32, standard batches",engine:"native",worker:"native",weights:"fp32",graph:"gpu",device:"dml",threads,batch:"standard"},
  {label:"native DirectML fp32, large batches",engine:"native",worker:"native",weights:"fp32",graph:"gpu",device:"dml",threads,batch:"large"},
  {label:"web WASM int8, 1 thread",engine:"web",worker:"wasm",weights:"int8",ep:"wasm",wasm:WASM.plain,threads:1,batch:"cpu"},
  {label:`web WASM int8, ${threads} threads`,engine:"web",worker:"wasm",weights:"int8",ep:"wasm",wasm:WASM.plain,threads,batch:"cpu"},
  ...(["jsep","webgpu","jspi"] as const).flatMap(build=>(["standard","large"] as const).map(batch=>({
    label:`web WebGPU (${build==="jsep"?"JSEP, ort.bundle":build==="webgpu"?"native EP, asyncify":"native EP, JSPI"}) fp32, ${batch} batches`,
    engine:"web" as const,worker:build,weights:"fp32" as const,graph:"upstream" as const,ep:"webgpu",
    wasm:build==="jsep"?WASM.jsep:build==="webgpu"?WASM.asyncify:WASM.jspi,threads:1,batch}))),
];

export default class EngineProbe extends Plugin {
  private running=false;
  async onload() {
    this.addCommand({id:"run",name:"Run engine probe",callback:()=>void this.run()});
    this.app.workspace.onLayoutReady(()=>void this.run());
  }
  private async run() {
    if(this.running) return;
    this.running=true;
    const fs=require("fs") as typeof import("fs"),path=require("path") as typeof import("path");
    const dir=path.join((this.app.vault.adapter as unknown as {basePath:string}).basePath,this.manifest.dir!);
    const probe=JSON.parse(fs.readFileSync(path.join(dir,"probe.json"),"utf8")) as Probe;
    // A portable build (build.mjs --portable) stores paths relative to this plugin's folder.
    for(const key of ["runtimeDirectory","ortWeb","modelDirectory"] as const)
      if(!path.isAbsolute(probe[key])) probe[key]=path.join(dir,probe[key]);
    const out=path.join(dir,"results.json");
    const p=process as unknown as {versions:Record<string,string>;platform:string};
    const results:{createdAt:string;environment:object;configs:object[];done?:boolean}={createdAt:new Date().toISOString(),environment:{
      electron:p.versions.electron,chrome:p.versions.chrome,platform:p.platform,cpu:require("os").cpus()[0].model,logicalCPUs:navigator.hardwareConcurrency,
      windowCrossOriginIsolated:window.crossOriginIsolated===true,documentFocused:document.hasFocus()},configs:[]};
    const save=()=>fs.writeFileSync(out,JSON.stringify(results,null,2));
    const bytes=(file:string)=>{const b=fs.readFileSync(file);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
    // ONNX Runtime's native environment must outlive the workers (see holdNativeRuntime in src/embedding/client.ts).
    try {require(`${probe.runtimeDirectory}/node_modules/onnxruntime-node`);} catch {}
    const tokenizer=JSON.parse(fs.readFileSync(path.join(probe.modelDirectory,"tokenizer.json"),"utf8"));
    const config=JSON.parse(fs.readFileSync(path.join(probe.modelDirectory,"tokenizer_config.json"),"utf8"));
    const references:Record<string,Float32Array[]>={};
    const CONFIGS=probe.suite==="lowbit" ? LOWBIT : ENGINES;
    (results as {suite?:string}).suite=probe.suite??"engines";
    const notice=new Notice("Engine probe: starting. Keep this window focused.",0);
    try {
      for(const [index,c] of CONFIGS.entries()) {
        notice.setMessage(`Engine probe ${index+1}/${CONFIGS.length}: ${c.label}. Keep this window focused.`);
        // Without cross-origin isolation, threaded WASM deadlocked in Obsidian 1.15 (Electron 34) instead of falling back.
        if(c.engine==="web" && c.threads>1 && !window.crossOriginIsolated) {
          results.configs.push({label:c.label,ok:false,skipped:true,error:"skipped: the window is not cross-origin isolated, and threaded WASM hung there"});
          save();
          continue;
        }
        const w=probe.weights[c.weights],batch=BATCHES[c.batch];
        const job={tokenizer,config,model:probe.model,limits:batch.limits,chunk:batch.chunk,docs:probe.docs,queries:probe.queries};
        const modelFile=path.join(probe.modelDirectory,c.graph==="gpu" && w.gpu ? w.gpu : w.path);
        const message:Record<string,unknown>=c.engine==="native"
          ? {job,runtimeDirectory:probe.runtimeDirectory,modelPath:modelFile,device:c.device,threads:c.threads}
          : {job,threads:c.threads,ep:c.ep,profile:c.profile===true,externalName:path.basename(w.external),wasm:bytes(path.join(probe.ortWeb,c.wasm!)),
            model:bytes(modelFile),external:bytes(path.join(probe.modelDirectory,w.external))};
        const started=performance.now();
        // Windows throttles an unfocused window, so count every focus loss during the step, not just the state at the end.
        let focusLosses=document.hasFocus() ? 0 : 1;
        const lost=()=>{focusLosses++;};
        window.addEventListener("blur",lost);
        let reply:Record<string,unknown>;
        try {reply=await this.inWorker(fs.readFileSync(path.join(dir,`worker-${c.worker}.js`),"utf8"),c.worker!=="native",message);}
        finally {window.removeEventListener("blur",lost);}
        const row:Record<string,unknown>={label:c.label,weights:c.weights,batch:c.batch,wallMs:Math.round(performance.now()-started),
          focused:focusLosses===0 && document.hasFocus(),focusLosses,...reply};
        const vectors=reply.vectors as Float32Array[]|undefined;
        delete row.vectors;
        if(vectors) {
          // The first native row of each weight variant is its reference (fp16 and q4f16 have only DirectML rows).
          if(c.engine==="native" && !references[c.weights] && vectors.every(v=>v.every(Number.isFinite))) references[c.weights]=vectors;
          const dot=(a:Float32Array,b:Float32Array)=>a.reduce((s,v,i)=>s+v*b[i],0);
          // Half-precision activations overflowed to NaN in onnxruntime-node's CPU q4f16 kernel; check every vector.
          row.finite=vectors.every(v=>v.every(Number.isFinite));
          for(const [weights,reference] of Object.entries(references))
            row[`minCosineVsNative${weights}`]=Math.min(...vectors.map((v,i)=>dot(v,reference[i])));
          for(const key of ["firstPassMsPerDoc","msPerDoc","queryMs","loadMs"]) row[key]=Math.round((row[key] as number)*10)/10;
        }
        results.configs.push(row);
        save();
      }
      results.done=true;
      save();
      notice.setMessage("Engine probe finished: results.json is in the probe's plugin folder.");
    } catch(error) {notice.setMessage(`Engine probe stopped: ${error}`);}
    finally {setTimeout(()=>notice.hide(),15000);this.running=false;}
  }
  /** One fresh worker per configuration, so GPU devices and WASM memory never carry over. */
  private inWorker(source:string,module:boolean,message:Record<string,unknown>):Promise<Record<string,unknown>> {
    return new Promise(resolve=>{
      const url=URL.createObjectURL(new Blob([source],{type:"text/javascript"}));
      const worker=new Worker(url,module?{type:"module"}:{});
      const finish=(reply:Record<string,unknown>)=>{clearTimeout(timer);worker.terminate();URL.revokeObjectURL(url);resolve(reply);};
      const timer=setTimeout(()=>finish({ok:false,error:"timed out after 5 minutes"}),300000);
      worker.onmessage=event=>finish(event.data);
      worker.onerror=event=>finish({ok:false,error:`worker error: ${event.message}`});
      const transfer=["wasm","model","external"].map(k=>message[k]).filter(Boolean) as ArrayBuffer[];
      worker.postMessage(message,transfer);
    });
  }
}
