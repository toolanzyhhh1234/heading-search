import { promises as fs } from "node:fs";
import { join } from "node:path";
import type { NativeDevice, Calibration } from "./backends";
import type { ModelWeights } from "./model-store";
import { ensureDirectory, promote } from "./shared-cache";
import { sharedRuntimeRoot } from "./runtime-store";

/**
 * Which device and weights this computer uses, measured once at setup and shared by every vault.
 * Vectors from different weights are incompatible, so vaults must agree, and the choice must not flip on noisy re-measurement.
 */
export interface DeviceProfile {
  version:1;
  device:NativeDevice;
  weights:ModelWeights;
  /** The person's choice: allow the GPU (see GPU_PREFERENCE). */
  allowGpu:boolean;
  /** The model the measurement was made with (id@revision). A different model is timed again at setup. */
  model?:string;
  measuredAt?:string;
  calibration?:Calibration;
}
export const DEFAULT_PROFILE:DeviceProfile={version:1,device:"native-cpu",weights:"int8",allowGpu:true};
/**
 * How much slower than the CPU the GPU may measure and still be chosen. Setup measures with Obsidian in the
 * foreground, but Windows runs a background Obsidian window at idle CPU priority: CPU embedding then slows about 3x
 * while DirectML barely changes. If half of bulk indexing happens in the background, the CPU's real cost is about
 * (1 + 3) / 2 = 2x its foreground measurement. The GPU also leaves CPU cores to Obsidian.
 */
export const GPU_PREFERENCE=2;
/** int8 runs best on the CPU; fp32 on DirectML. */
export function chooseDevice(calibration:Calibration,allowGpu:boolean):Pick<DeviceProfile,"device"|"weights"> {
  return allowGpu && calibration.gpuMs!==null && calibration.gpuMs<=calibration.cpuMs*GPU_PREFERENCE
    ? {device:"directml",weights:"fp32"} : {device:"native-cpu",weights:"int8"};
}

export class DeviceProfileStore {
  readonly path:string;
  constructor(root=sharedRuntimeRoot()) {this.path=join(root,"device.json");}
  async read():Promise<DeviceProfile|null> {
    try {
      const value=JSON.parse(await fs.readFile(this.path,"utf8")) as DeviceProfile;
      const valid=value.version===1 && ["directml","native-cpu"].includes(value.device) && ["int8","fp32"].includes(value.weights) && typeof value.allowGpu==="boolean";
      return valid ? value : null;
    } catch {return null;}
  }
  async write(profile:DeviceProfile):Promise<void> {
    await ensureDirectory(join(this.path,".."));
    const temporary=`${this.path}.${process.pid}.${Date.now()}.partial`;
    await fs.writeFile(temporary,JSON.stringify(profile,null,2));
    await promote(temporary,this.path);
  }
}
