// The shipped native engine (onnxruntime-node), for a same-window comparison. Obsidian's workers have Node's require.
import { bench, describeError } from "./common";

self.onmessage=async(event:MessageEvent)=>{
  const {job,runtimeDirectory,modelPath,device,threads}=event.data;
  try {
    const ort=(globalThis as unknown as {require:(id:string)=>{InferenceSession:{create(p:string,o:object):Promise<{release():Promise<void>}>}}})
      .require(`${runtimeDirectory}/node_modules/onnxruntime-node`);
    const options=device==="dml"
      ? {executionProviders:["dml"],enableMemPattern:false,executionMode:"sequential",graphOptimizationLevel:"all"}
      : {executionProviders:["cpu"],intraOpNumThreads:threads,interOpNumThreads:1,graphOptimizationLevel:"all"};
    const start=performance.now();
    const session=await ort.InferenceSession.create(modelPath,options);
    const loadMs=performance.now()-start;
    const result=await bench(ort as never,session as never,job);
    await session.release();
    self.postMessage({ok:true,env:{},loadMs,...result});
  } catch(error) { self.postMessage({ok:false,env:{},error:describeError(error)}); }
};
