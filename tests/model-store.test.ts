import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, writeFile, mkdir, utimes } from "node:fs/promises";
import { join } from "node:path";
import { resolve, sep } from "node:path";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { ModelStore, modelDefinition, weightsBytes, type ModelDefinition } from "../src/embedding/model-store";

const data=new TextEncoder().encode("test model asset");
const definition:ModelDefinition={id:"test/model",revision:"pinned",files:[{path:"onnx/model.onnx",bytes:data.length,sha256:createHash("sha256").update(data).digest("hex")}]};
const testRoot=resolve(".cache/tests");
async function temporaryRoot():Promise<string> {await mkdir(testRoot,{recursive:true});return mkdtemp(join(testRoot,"model-"));}
async function cleanup(root:string):Promise<void> {assert.ok(resolve(root).startsWith(testRoot+sep));await rm(root,{recursive:true,force:true});}
test("simultaneous vaults download once and a later vault works offline",async()=>{
  const root=await temporaryRoot();
  try {
    let requests=0;
    const download=async(url:string)=>{requests++;assert.equal(url,"https://huggingface.co/test/model/resolve/pinned/onnx/model.onnx");await new Promise(r=>setTimeout(r,30));return Uint8Array.from(data).buffer;};
    const first=new ModelStore(download,root,definition),second=new ModelStore(download,root,definition);
    await Promise.all([first.ensure(),second.ensure()]);assert.equal(requests,1);
    const offline=new ModelStore(async()=>{throw new Error("No network");},root,definition);
    await offline.ensure();assert.equal(await offline.isInstalled(),true);
  } finally {await cleanup(root);}
});
test("a failed integrity check is not installed, retries repair a corrupt asset",async()=>{
  const root=await temporaryRoot();
  try {
    const bad=new ModelStore(async()=>new ArrayBuffer(3),root,definition);
    await assert.rejects(bad.ensure(),/Integrity/);assert.equal(await bad.isInstalled(),false);
    const good=new ModelStore(async()=>Uint8Array.from(data).buffer,root,definition);
    await good.ensure();const target=join(good.directory,definition.files[0].path);
    await writeFile(target,"corrupt");assert.equal(await good.isInstalled(),false);
    await good.ensure();assert.deepEqual(await readFile(target),Buffer.from(data));
  } finally {await cleanup(root);}
});
test("stale download leases from a crashed vault are recoverable",async()=>{
  const root=await temporaryRoot();
  try {
    const store=new ModelStore(async()=>Uint8Array.from(data).buffer,root,definition);
    const lock=join(store.directory,".download-lock");await mkdir(lock,{recursive:true});
    await utimes(lock,new Date(0),new Date(0));await store.ensure();assert.equal(await store.isInstalled(),true);
  } finally {await cleanup(root);}
});
test("transient Windows file handles retry promotion without downloading again",async context=>{
  const root=await temporaryRoot();
  try {
    const original=fs.rename.bind(fs);let attempts=0,downloads=0;
    context.mock.method(fs,"rename",async(...args:Parameters<typeof fs.rename>)=>{
      if(++attempts<=2)throw Object.assign(new Error("File temporarily busy"),{code:"EPERM"});
      return original(...args);
    });
    const store=new ModelStore(async()=>{downloads++;return Uint8Array.from(data).buffer;},root,definition);
    await store.ensure();assert.equal(downloads,1);assert.equal(await store.isInstalled(),true);
  } finally {await cleanup(root);}
});

const bundledData=new TextEncoder().encode("bundled GPU graph");
const withBundled:ModelDefinition={...definition,files:[...definition.files,
  {path:"onnx/model.gpu.onnx",bytes:bundledData.length,sha256:createHash("sha256").update(bundledData).digest("hex"),bundled:bundledData}],
  model:"onnx/model.onnx",gpuModel:"onnx/model.gpu.onnx"};
test("bundled files install without a download, and an update adds them offline beside existing weights",async()=>{
  const root=await temporaryRoot();
  try {
    const urls:string[]=[];
    const store=new ModelStore(async url=>{urls.push(url);return Uint8Array.from(data).buffer;},root,withBundled);
    await store.ensure();
    assert.deepEqual(urls,["https://huggingface.co/test/model/resolve/pinned/onnx/model.onnx"]);
    assert.deepEqual(await readFile(store.gpuModelPath),Buffer.from(bundledData));
    // An installation from before the GPU graph existed: the weights are there, the bundled graph is not.
    await rm(store.gpuModelPath);
    const updated=new ModelStore(async()=>{throw new Error("No network");},root,withBundled);
    assert.equal(await updated.isInstalled(),false);
    assert.equal(await updated.isReady(),true);
    assert.equal(await updated.isInstalled(),true);
    assert.deepEqual(await readFile(updated.gpuModelPath),Buffer.from(bundledData));
    await updated.removeWeights();
    for(const path of [updated.modelPath,updated.gpuModelPath])await assert.rejects(readFile(path));
  } finally {await cleanup(root);}
});
test("offline readiness never downloads and leaves no trace when setup never ran",async()=>{
  const root=await temporaryRoot();
  try {
    let requests=0;
    const store=new ModelStore(async()=>{requests++;return Uint8Array.from(data).buffer;},root,withBundled);
    assert.equal(await store.isReady(),false);
    await assert.rejects(store.read(),/Check \/ repair/);
    assert.equal(requests,0);
    await assert.rejects(fs.stat(store.directory));
  } finally {await cleanup(root);}
});
test("the shipped GPU graph is bundled with its pinned hash, and the CPU keeps the upstream graph",()=>{
  const fp32=modelDefinition("fp32"),gpu=fp32.files.find(file=>file.path===fp32.gpuModel);
  assert.ok(gpu?.bundled && fp32.gpuModel!==fp32.model);
  assert.equal(gpu.bundled.length,gpu.bytes);
  assert.equal(createHash("sha256").update(gpu.bundled).digest("hex"),gpu.sha256);
  assert.equal(weightsBytes("fp32"),fp32.files.reduce((sum,file)=>sum+file.bytes,0)-gpu.bytes);
  assert.equal(modelDefinition("int8").gpuModel,undefined);
});
