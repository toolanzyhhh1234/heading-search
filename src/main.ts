import { Plugin, Notice, requestUrl, TFile } from "obsidian";
import { readSettings, type Settings, type SearchMode } from "./settings";
import policy from "./retrieval-policy.json";
import { EmbeddingClient } from "./embedding/client";
import { modelConfig, type ModelConfig } from "./embedding/engine";
import spec from "./model-spec.json";
import { ModelStore, modelDefinition, modelKey, type ModelWeights, type Download } from "./embedding/model-store";
import { RuntimeStore } from "./embedding/runtime-store";
import { CALIBRATION_VERSION, DEFAULT_PROFILE, DeviceProfileStore, chooseDevice, chooseGpuBatch, vectorWeights, type DeviceProfile } from "./embedding/device-profile";
import { checkWebGpu } from "./embedding/web-runtime";
import { SearchService } from "./index/service";
import { SearchModal } from "./ui/search-modal";
import { SearchSettingsTab } from "./ui/settings-tab";

export default class HeadingSearchPlugin extends Plugin {
  declare settings:Settings;
  service!:SearchService;
  runtimeStore!:RuntimeStore;
  embedding!:EmbeddingClient;
  /** Device and weights for this computer, shared by every vault (see DeviceProfile). */
  profile:DeviceProfile=DEFAULT_PROFILE;
  private profiles=new DeviceProfileStore();
  private models=new Map<ModelWeights,ModelStore>();
  private download!:Download;
  private refreshTimer?:ReturnType<typeof setTimeout>;
  private active=true;
  settingUp=false;
  sharedModelAvailable=false;
  setupMessage="";

  async onload():Promise<void> {
    this.settings=readSettings(await this.loadData());
    if(!this.settings.semanticEnabled)this.settings.mode="bm25";
    const status=this.addStatusBarItem();
    status.setAttribute("aria-label","Heading Search status");
    this.download=async(url:string)=>(await requestUrl({url,method:"GET",throw:true})).arrayBuffer;
    this.runtimeStore=new RuntimeStore(this.download);
    this.profile=await this.profiles.read() ?? DEFAULT_PROFILE;
    // Leave half the logical cores for Obsidian itself; four threads already capture most of the speedup.
    const threads=Math.max(1,Math.min(4,Math.floor((navigator.hardwareConcurrency||2)/2)));
    this.embedding=new EmbeddingClient(weights=>this.model(weights),this.runtimeStore,{threads,profile:()=>this.profile,model:modelConfig(spec as ModelConfig)});
    const directory=`${this.app.vault.configDir}/plugins/${this.manifest.id}/index`;
    this.service=new SearchService(this.app,()=>this.settings,this.embedding,directory,()=>status.setText(`Search: ${this.service.status}`),()=>vectorWeights(this.profile.weights));
    await Promise.all([this.service.load(),this.checkShared()]);
    this.addRibbonIcon("search","Search notes by heading and meaning",()=>this.openSearch());
    this.addCommand({id:"search-notes",name:"Search notes",callback:()=>this.openSearch()});
    this.addCommand({id:"rebuild-index",name:"Rebuild search index",callback:()=>this.refresh(true)});
    this.addCommand({id:"toggle-indexing",name:"Pause or resume indexing",callback:()=>this.togglePause()});
    this.addSettingTab(new SearchSettingsTab(this.app,this));
    const changed=(file:import("obsidian").TAbstractFile)=>{if(file instanceof TFile && file.extension==="md")this.scheduleRefresh();};
    this.registerEvent(this.app.vault.on("create",changed));
    this.registerEvent(this.app.vault.on("modify",changed));
    this.registerEvent(this.app.vault.on("delete",changed));
    this.registerEvent(this.app.vault.on("rename",()=>this.scheduleRefresh()));
    this.app.workspace.onLayoutReady(()=>this.refresh());
  }
  model(weights:ModelWeights):ModelStore {
    let store=this.models.get(weights);
    if(!store) {store=new ModelStore(this.download,undefined,modelDefinition(weights));this.models.set(weights,store);}
    return store;
  }
  private async checkShared():Promise<void> {
    // isReady also installs files bundled with a plugin update (the GPU graph) beside weights already on disk.
    const [model,runtime]=await Promise.all([this.model(this.profile.weights).isReady(),this.runtimeStore.isInstalled()]);
    // The built-in engine needs only the model; the native one also needs the downloaded runtime.
    this.sharedModelAvailable=model && (this.profile.device==="webgpu" || runtime || !this.runtimeStore.supported);
  }
  private openSearch():void {new SearchModal(this.app,this.service,{...this.settings}).open();}
  async saveSettings():Promise<void> {await this.saveData(this.settings);}
  scheduleRefresh():void {
    clearTimeout(this.refreshTimer);
    this.refreshTimer=setTimeout(()=>this.refresh(),1000);
  }
  refresh(rebuild=false):void {
    if(!this.active)return;
    void this.service.refresh(rebuild).catch(error=>{
      if(!this.active)return;
      console.error("Heading Search indexing failed",error);
      this.service.status="Indexing needs attention. Open Heading Search settings.";
      new Notice(`Heading Search: ${error instanceof Error ? error.message : "Indexing failed."}`,8000);
    });
  }
  togglePause():void {
    this.service.paused=!this.service.paused;
    new Notice(this.service.paused ? "Heading Search indexing paused." : "Heading Search indexing resumed.");
    if(!this.service.paused)this.refresh();
  }

  /**
   * Pick the device once per computer. When the GPU is allowed, the built-in WebGPU engine with fp16 weights is tried
   * first: it needs no native download and indexed faster than every native engine where measured. Otherwise the native
   * engine is installed: int8 on the CPU, and on Windows fp32 is fetched to time DirectML and kept only if the GPU wins,
   * which then also times its batch tier. A saved choice is reused so re-measurement noise never forces re-embedding;
   * a new model or calibration method is measured again.
   */
  private async chooseProfile(update:(message:string)=>void,allowGpu:boolean,remeasure:boolean):Promise<DeviceProfile> {
    const saved=await this.profiles.read();
    if(saved && saved.allowGpu===allowGpu && saved.model===modelKey && saved.calibrationVersion===CALIBRATION_VERSION && !remeasure
      && await this.model(saved.weights).isReady())return saved;
    let webgpuError:string|undefined;
    if(allowGpu) {
      const webgpu=await this.tryWebGpu(update);
      if("profile" in webgpu) {
        if(this.runtimeStore.gpu)await this.model("fp32").removeWeights();
        await this.profiles.write(webgpu.profile);
        return webgpu.profile;
      }
      webgpuError=webgpu.error;
      update(`The built-in GPU engine cannot be used: ${webgpuError} Setting up the native engine instead…`);
    }
    if(!this.runtimeStore.supported)
      throw new Error(`Semantic search is not available for ${this.runtimeStore.platform}.${webgpuError ? ` The built-in GPU engine cannot be used: ${webgpuError}` : ""} Keyword search still works.`);
    await this.runtimeStore.ensure(update);
    await this.model("int8").ensure(update);
    let profile:DeviceProfile={...DEFAULT_PROFILE,allowGpu,model:modelKey,calibrationVersion:CALIBRATION_VERSION,webgpuError};
    if(allowGpu && this.runtimeStore.gpu) {
      await this.model("fp32").ensure(message=>update(`Testing this computer's graphics: ${message}`));
      update("Comparing the GPU and CPU on this computer…");
      // Stop this vault's engine first so its indexing does not compete with the measurement.
      this.embedding.restart();
      const calibration=await this.embedding.calibrate(),choice=chooseDevice(calibration,allowGpu);
      let gpuBatch:DeviceProfile["gpuBatch"];
      if(choice.device==="directml") {
        update("Choosing how many sections the GPU embeds at once…");
        // The GPU already passed; if this extra measurement fails, small batches are the safe choice.
        calibration.batch=await this.embedding.calibrateBatch().catch(()=>undefined);
        gpuBatch=calibration.batch ? chooseGpuBatch(calibration.batch) : "standard";
      }
      profile={version:1,...choice,gpuBatch,allowGpu,model:modelKey,calibrationVersion:CALIBRATION_VERSION,calibration,webgpuError,measuredAt:new Date().toISOString()};
    }
    if(profile.weights!=="fp32" && this.runtimeStore.gpu)await this.model("fp32").removeWeights();
    await this.profiles.write(profile);
    return profile;
  }
  /** Downloads the fp16 weights only if a hardware adapter exists, then tests the engine; the weights go if it fails. */
  private async tryWebGpu(update:(message:string)=>void):Promise<{profile:DeviceProfile}|{error:string}> {
    const check=await checkWebGpu();
    if(!check.ok)return {error:check.error};
    await this.model("fp16").ensure(update);
    update("Testing the built-in GPU engine on this computer…");
    // Stop this vault's engine first so its indexing does not compete with the measurement.
    this.embedding.restart();
    try {
      const ms=await this.embedding.calibrateWebGpu();
      return {profile:{version:1,device:"webgpu",weights:"fp16",allowGpu:true,model:modelKey,calibrationVersion:CALIBRATION_VERSION,
        webgpu:{adapter:check.adapter,ms},measuredAt:new Date().toISOString()}};
    } catch(error) {
      await this.model("fp16").removeWeights();
      return {error:`Its test failed on ${check.adapter} (${error instanceof Error ? error.message : String(error)}).`};
    }
  }
  private apply(profile:DeviceProfile):void {
    this.profile=profile;
    this.embedding.restart();
    this.sharedModelAvailable=true;
  }
  describeProfile(profile=this.profile):string {
    if(profile.device==="webgpu")
      return `Using the GPU through the built-in engine (WebGPU) with half-precision weights${profile.webgpu ? ` on ${profile.webgpu.adapter}. Measured on 12 sample sections: ${profile.webgpu.ms.toFixed(0)} ms.` : "."}`;
    const webgpu=profile.webgpuError ? ` The built-in GPU engine was not used: ${profile.webgpuError}` : "";
    const c=profile.calibration;
    const timing=c ? ` Measured on 12 sample sections: CPU ${c.cpuMs.toFixed(0)} ms, GPU ${c.gpuMs===null ? `unavailable${c.gpuError ? ` (${c.gpuError})` : ""}` : `${c.gpuMs.toFixed(0)} ms`}.` : "";
    const batch=c?.batch ? ` On 64 sections: small batches ${c.batch.standardMs.toFixed(0)} ms, large batches ${c.batch.largeMs===null ? "unavailable" : `${c.batch.largeMs.toFixed(0)} ms`}.` : "";
    return `${profile.device==="directml" ? `Using the GPU (DirectML) with full-precision weights${profile.gpuBatch==="large" ? " in large batches" : ""}.` : "Using the CPU with 8-bit weights."}${timing}${batch}${webgpu}`;
  }
  async setupSemantic(progress:(message:string)=>void):Promise<void> {
    if(this.settingUp)return;
    this.settingUp=true;
    const wasEnabled=this.settings.semanticEnabled;
    const update=(message:string)=>{this.setupMessage=message;progress(message);};
    try {
      const profile=await this.chooseProfile(update,(await this.profiles.read())?.allowGpu ?? true,false);
      // Check / repair: a saved native choice still needs its runtime, which chooseProfile only installs when measuring.
      if(profile.device!=="webgpu")await this.runtimeStore.ensure(update);
      await this.model(profile.weights).ensure(update);
      this.apply(profile);
      update(`Semantic search is ready. ${this.describeProfile()}`);
      if(!this.active)return;
      this.settings.semanticEnabled=true;
      if(!wasEnabled && ["vector","hybrid"].includes(policy.recommendedMode))this.settings.mode=policy.recommendedMode as SearchMode;
      await this.saveSettings();this.refresh();
    } catch(error) {
      update(error instanceof Error ? error.message : "Setup failed. Please retry.");throw error;
    } finally {this.settingUp=false;}
  }
  /** Computer-wide: every vault re-embeds once with the newly chosen weights if the device changes. */
  async setGpuAllowed(allowGpu:boolean,progress:(message:string)=>void):Promise<void> {
    if(this.settingUp)return;
    this.settingUp=true;
    const update=(message:string)=>{this.setupMessage=message;progress(message);};
    try {
      const before=this.profile.weights,profile=await this.chooseProfile(update,allowGpu,true);
      this.apply(profile);
      update(`${this.describeProfile()}${vectorWeights(profile.weights)!==vectorWeights(before) ? " Notes will be embedded again with the new weights." : ""}`);
      this.refresh();
    } catch(error) {
      update(error instanceof Error ? error.message : "Could not change the device. Please retry.");throw error;
    } finally {this.settingUp=false;}
  }
  onunload():void {this.active=false;clearTimeout(this.refreshTimer);this.service?.dispose();}
}
