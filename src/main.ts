import { Plugin, Notice, requestUrl, TFile } from "obsidian";
import { readSettings, type Settings, type SearchMode } from "./settings";
import policy from "./retrieval-policy.json";
import { EmbeddingClient } from "./embedding/client";
import { ModelStore } from "./embedding/model-store";
import { SearchService } from "./index/service";
import { SearchModal } from "./ui/search-modal";
import { SearchSettingsTab } from "./ui/settings-tab";

export default class HeadingSearchPlugin extends Plugin {
  declare settings:Settings;
  service!:SearchService;
  private modelStore!:ModelStore;
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
    this.modelStore=new ModelStore(async url=>{
      const response=await requestUrl({url,method:"GET",throw:true});
      return response.arrayBuffer;
    });
    const embedding=new EmbeddingClient(this.modelStore);
    const directory=`${this.app.vault.configDir}/plugins/${this.manifest.id}/index`;
    this.service=new SearchService(this.app,()=>this.settings,embedding,directory,()=>status.setText(`Search: ${this.service.status}`));
    await Promise.all([this.service.load(),this.modelStore.isInstalled().then(installed=>{this.sharedModelAvailable=installed;})]);
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
  async setupSemantic(progress:(message:string)=>void):Promise<void> {
    if(this.settingUp)return;
    this.settingUp=true;
    const wasEnabled=this.settings.semanticEnabled;
    const update=(message:string)=>{this.setupMessage=message;progress(message);};
    try {
      await this.modelStore.ensure(update);
      this.sharedModelAvailable=true;
      if(!this.active)return;
      this.settings.semanticEnabled=true;
      if(!wasEnabled && ["vector","hybrid"].includes(policy.recommendedMode))this.settings.mode=policy.recommendedMode as SearchMode;
      await this.saveSettings();this.refresh();
    } catch(error) {
      update(error instanceof Error ? error.message : "Model setup failed. Please retry.");throw error;
    } finally {this.settingUp=false;}
  }
  onunload():void {this.active=false;clearTimeout(this.refreshTimer);this.service?.dispose();}
}
