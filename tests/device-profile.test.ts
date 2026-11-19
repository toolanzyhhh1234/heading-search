import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, mkdir, writeFile } from "node:fs/promises";
import { join, resolve, sep } from "node:path";
import { CALIBRATION_VERSION, chooseDevice, chooseGpuBatch, DeviceProfileStore, DEFAULT_PROFILE } from "../src/embedding/device-profile";

const testRoot=resolve(".cache/tests");
async function temporaryRoot():Promise<string> {await mkdir(testRoot,{recursive:true});return mkdtemp(join(testRoot,"device-"));}

test("the GPU is chosen with fp32 weights when allowed and at most twice as slow in the foreground",()=>{
  assert.deepEqual(chooseDevice({cpuMs:500,gpuMs:520},true),{device:"directml",weights:"fp32"});
  // Measured on the development laptop: the CPU is faster in the foreground, but 3x slower in the background.
  assert.deepEqual(chooseDevice({cpuMs:358,gpuMs:590},true),{device:"directml",weights:"fp32"});
  assert.deepEqual(chooseDevice({cpuMs:300,gpuMs:700},true),{device:"native-cpu",weights:"int8"});
  assert.deepEqual(chooseDevice({cpuMs:500,gpuMs:100},false),{device:"native-cpu",weights:"int8"});
  assert.deepEqual(chooseDevice({cpuMs:500,gpuMs:null,gpuError:"No adapter"},true),{device:"native-cpu",weights:"int8"});
});

test("the device profile is shared on disk and invalid files fall back safely",async()=>{
  const root=await temporaryRoot();
  try {
    const store=new DeviceProfileStore(root);
    assert.equal(await store.read(),null);
    const profile={...DEFAULT_PROFILE,device:"directml" as const,weights:"fp32" as const,calibration:{cpuMs:400,gpuMs:300}};
    await store.write(profile);
    assert.deepEqual(await new DeviceProfileStore(root).read(),profile);
    await writeFile(store.path,JSON.stringify({version:1,device:"cuda",weights:"fp32",allowGpu:true}));
    assert.equal(await store.read(),null);
  } finally {assert.ok(resolve(root).startsWith(testRoot+sep));await rm(root,{recursive:true,force:true});}
});

test("large GPU batches are chosen only when clearly faster, and never after failing",()=>{
  assert.equal(chooseGpuBatch({standardMs:1024,largeMs:772}),"large"); // RTX 3070 on the batch probe
  assert.equal(chooseGpuBatch({standardMs:1000,largeMs:900}),"standard");
  assert.equal(chooseGpuBatch({standardMs:1000,largeMs:null}),"standard");
});

test("a saved batch tier round-trips and an unknown one invalidates the profile",async()=>{
  const root=await temporaryRoot();
  try {
    const store=new DeviceProfileStore(root);
    const profile={...DEFAULT_PROFILE,device:"directml" as const,weights:"fp32" as const,gpuBatch:"large" as const,calibrationVersion:CALIBRATION_VERSION};
    await store.write(profile);
    assert.deepEqual(await store.read(),profile);
    await writeFile(store.path,JSON.stringify({...profile,gpuBatch:"huge"}));
    assert.equal(await store.read(),null);
  } finally {assert.ok(resolve(root).startsWith(testRoot+sep));await rm(root,{recursive:true,force:true});}
});
