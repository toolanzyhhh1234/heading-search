import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readFile, writeFile, mkdir, readdir } from "node:fs/promises";
import { join, resolve, sep } from "node:path";
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { RuntimeStore, type RuntimeSpec } from "../src/embedding/runtime-store";
import { extractTarGz } from "../src/embedding/tar";

/** A minimal ustar writer covering the entry kinds npm tarballs use. */
function tar(entries:{name:string;data?:string;type?:string;link?:string}[]):Uint8Array {
  const blocks:Buffer[]=[];
  const header=(name:string,size:number,type:string,link="")=>{
    const block=Buffer.alloc(512);
    block.write(name.slice(0,100),0);block.write("0000644\0",100);block.write(size.toString(8).padStart(11,"0")+"\0",124);
    block.write(type,156);block.write(link,157);block.write("ustar\0",257);
    return block;
  };
  const body=(data:Buffer)=>[data,Buffer.alloc((512-data.length%512)%512)];
  for(const entry of entries) {
    const data=Buffer.from(entry.data??"");
    if(entry.name.length>100) {const name=Buffer.from(entry.name+"\0");blocks.push(header("././@LongLink",name.length,"L"),...body(name));}
    blocks.push(header(entry.name,data.length,entry.type??"0",entry.link),...body(data));
  }
  blocks.push(Buffer.alloc(1024));
  return gzipSync(Buffer.concat(blocks));
}
const sha256=(text:string)=>createHash("sha256").update(text).digest("hex");
const testRoot=resolve(".cache/tests");
async function temporaryRoot():Promise<string> {await mkdir(testRoot,{recursive:true});return mkdtemp(join(testRoot,"runtime-"));}
async function cleanup(root:string):Promise<void> {assert.ok(resolve(root).startsWith(testRoot+sep));await rm(root,{recursive:true,force:true});}

const longPath=`package/bin/napi-v6/win32/x64/${"a".repeat(90)}.dll`;
const archive=tar([
  {name:"package/package.json",data:'{"name":"onnxruntime-node"}'},
  {name:"package/dist/",type:"5"},
  {name:"package/dist/index.js",data:"module.exports=1;"},
  {name:"package/bin/napi-v6/win32/x64/onnxruntime_binding.node",data:"binary"},
  {name:longPath,data:"long library"},
  {name:"package/bin/napi-v6/linux/x64/libonnxruntime.so.1",data:"other platform"},
  // Links must never be written, even when their name matches.
  {name:"package/bin/napi-v6/win32/x64/escape.dll",type:"2",link:"../../../../../evil"},
]);
const spec:RuntimeSpec={
  version:"0.0.1",
  archives:{"onnxruntime-node":{url:"https://registry.npmjs.org/onnxruntime-node/-/onnxruntime-node-0.0.1.tgz",sha512:createHash("sha512").update(archive).digest("base64"),bytes:archive.length}},
  shared:[
    {archive:"onnxruntime-node",path:"package/package.json",sha256:sha256('{"name":"onnxruntime-node"}'),bytes:27},
    {archive:"onnxruntime-node",path:"package/dist/index.js",sha256:sha256("module.exports=1;"),bytes:17},
  ],
  platforms:{"win32-x64":{gpu:"directml",files:[
    {archive:"onnxruntime-node",path:"package/bin/napi-v6/win32/x64/onnxruntime_binding.node",sha256:sha256("binary"),bytes:6},
    {archive:"onnxruntime-node",path:longPath,sha256:sha256("long library"),bytes:12},
  ]}},
};

test("tar extraction keeps only wanted regular files, including GNU long names",async()=>{
  const files=await extractTarGz(archive,()=>true);
  assert.deepEqual([...files.keys()].sort(),["package/bin/napi-v6/linux/x64/libonnxruntime.so.1","package/bin/napi-v6/win32/x64/onnxruntime_binding.node",longPath,"package/dist/index.js","package/package.json"].sort());
  assert.equal(Buffer.from(files.get(longPath)!).toString(),"long library");
});

test("runtime installs only this platform's verified files, once across vaults, and works offline",async()=>{
  const root=await temporaryRoot();
  try {
    let downloads=0;
    const download=async(url:string)=>{downloads++;assert.equal(url,spec.archives["onnxruntime-node"].url);await new Promise(r=>setTimeout(r,30));return Uint8Array.from(archive).buffer;};
    const first=new RuntimeStore(download,root,spec,"win32","x64"),second=new RuntimeStore(download,root,spec,"win32","x64");
    await Promise.all([first.ensure(),second.ensure()]);
    assert.equal(downloads,1);assert.equal(await first.isInstalled(),true);
    const binaries=await readdir(join(first.directory,"node_modules/onnxruntime-node/bin/napi-v6/win32/x64"));
    assert.deepEqual(binaries.sort(),["onnxruntime_binding.node",`${"a".repeat(90)}.dll`].sort());
    await assert.rejects(readFile(join(first.directory,"node_modules/onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime.so.1")));
    const offline=new RuntimeStore(async()=>{throw new Error("No network");},root,spec,"win32","x64");
    await offline.ensure();assert.equal(await offline.isInstalled(),true);
    // Staging folders are cleaned up.
    assert.deepEqual((await readdir(root)).filter(name=>name.endsWith(".partial")),[]);
  } finally {await cleanup(root);}
});

test("a tampered download or damaged install is rejected, then repaired",async()=>{
  const root=await temporaryRoot();
  try {
    const tampered=Uint8Array.from(archive);tampered[tampered.length-20]^=1;
    const bad=new RuntimeStore(async()=>tampered.buffer,root,spec,"win32","x64");
    await assert.rejects(bad.ensure(),/Integrity/);assert.equal(await bad.isInstalled(),false);
    const good=new RuntimeStore(async()=>Uint8Array.from(archive).buffer,root,spec,"win32","x64");
    await good.ensure();
    const binding=join(good.directory,"node_modules/onnxruntime-node/bin/napi-v6/win32/x64/onnxruntime_binding.node");
    await writeFile(binding,"replaced");assert.equal(await good.isInstalled(),false);
    await good.ensure();assert.equal(await readFile(binding,"utf8"),"binary");assert.equal(await good.isInstalled(),true);
  } finally {await cleanup(root);}
});

test("unsupported platforms report that the fast engine is unavailable",async()=>{
  const store=new RuntimeStore(async()=>new ArrayBuffer(0),testRoot,spec,"freebsd","x64");
  assert.equal(store.supported,false);assert.equal(await store.isInstalled(),false);
  await assert.rejects(store.ensure(),/not available/);
});
