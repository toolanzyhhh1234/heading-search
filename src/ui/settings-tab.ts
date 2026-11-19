import { PluginSettingTab, Setting, Notice, type App } from "obsidian";
import type HeadingSearchPlugin from "../main";
import type { SearchMode } from "../settings";
import type { VectorCodec } from "../core/vectors";
import { sharedModelRoot, weightsBytes } from "../embedding/model-store";
import { sharedRuntimeRoot } from "../embedding/runtime-store";
import policy from "../retrieval-policy.json";

export class SearchSettingsTab extends PluginSettingTab {
  constructor(app:App,private plugin:HeadingSearchPlugin) {super(app,plugin);}
  display():void {
    const {containerEl}=this;containerEl.empty();
    new Setting(containerEl).setName("Search").setHeading();
    new Setting(containerEl).setName("Search method")
      .setDesc("Keyword search uses your headings immediately. Semantic and hybrid search use the optional local model.")
      .addDropdown(dropdown=>{
        dropdown.addOptions({bm25:"Keyword (BM25)",vector:"Semantic",hybrid:"Hybrid"}).setValue(this.plugin.settings.mode);
        dropdown.onChange(async value=>{
          if(value!=="bm25" && !this.plugin.settings.semanticEnabled) {new Notice("Set up semantic search below first.");dropdown.setValue("bm25");return;}
          this.plugin.settings.mode=value as SearchMode;await this.plugin.saveSettings();
        });
      });
    new Setting(containerEl).setName("Results per search").addSlider(slider=>slider.setLimits(5,50,5)
      .setValue(this.plugin.settings.resultCount).setDynamicTooltip().onChange(async value=>{this.plugin.settings.resultCount=value;await this.plugin.saveSettings();}));
    new Setting(containerEl).setName("Optional semantic search").setHeading();
    const megabytes=(bytes:number)=>Math.round(bytes/1e6);
    const runtime=this.plugin.runtimeStore;
    const engine=runtime.supported ? ` and the fast engine (Microsoft ONNX Runtime, ${megabytes(runtime.downloadBytes)} MB from the npm registry)` : "";
    const gpuTest=runtime.gpu ? ` On Windows, setup also downloads full-precision weights (${megabytes(weightsBytes("fp32"))} MB) to test the GPU, and keeps them only if the GPU is faster.` : "";
    containerEl.createEl("p",{text:`IBM's Granite Embedding small model runs on this computer. Setup downloads the model (${megabytes(weightsBytes("int8"))} MB from Hugging Face)${engine} once per computer user, and every vault shares them.${gpuTest} Your notes and queries stay on this computer. No API key or separate app is needed.`});
    const status=containerEl.createDiv({cls:"heading-search-model-status",attr:{role:"status","aria-live":"polite"}});
    status.setText(this.plugin.setupMessage || (this.plugin.sharedModelAvailable ? "The shared model is already installed. You can use it here without downloading it again." : "The model is shared by all your vaults."));
    new Setting(containerEl).setName(this.plugin.settings.semanticEnabled ? "Shared model enabled" : "Set up the shared model")
      .setDesc("A model and engine already set up by another vault are reused, including when offline.")
      .addButton(button=>{
        button.setButtonText(this.plugin.settings.semanticEnabled ? "Check / repair" : this.plugin.sharedModelAvailable ? "Use shared model" : "Enable semantic search").setCta().setDisabled(this.plugin.settingUp);
        button.onClick(async()=>{
          button.setDisabled(true);
          try {await this.plugin.setupSemantic(message=>status.setText(message));}
          catch(error) {status.setText(error instanceof Error ? error.message : "Setup failed. Please retry.");}
          finally {this.display();}
        });
      });
    if(this.plugin.settings.semanticEnabled) {
      const {info,fallbackReason}=this.plugin.embedding;
      new Setting(containerEl).setName("Embedding engine")
        .setDesc(`${this.plugin.describeProfile()} ${info ? `Running: ${info.label}.${fallbackReason ? ` ${fallbackReason}` : ""}` : "Starts when notes need embedding or you run a semantic search."}`);
      if(runtime.gpu && runtime.supported) new Setting(containerEl).setName("Allow the GPU on this computer")
        .setDesc("When on, the GPU is used unless it measured more than twice as slow as the CPU, because the CPU slows down whenever Obsidian is in the background. This applies to every vault; changing it re-measures, and vaults embed their notes again if the device changes.")
        .addToggle(toggle=>toggle.setValue(this.plugin.profile.allowGpu).setDisabled(this.plugin.settingUp).onChange(async value=>{
          toggle.setDisabled(true);
          try {await this.plugin.setGpuAllowed(value,message=>status.setText(message));}
          catch(error) {status.setText(error instanceof Error ? error.message : "Could not change the device.");}
          finally {this.display();}
        }));
    }
    if(this.plugin.settings.semanticEnabled) new Setting(containerEl).setName("Use semantic search in this vault")
      .setDesc("Turning this off stops new embeddings. The shared model stays available for other vaults.")
      .addToggle(toggle=>toggle.setValue(true).onChange(async()=>{
        this.plugin.settings.semanticEnabled=false;this.plugin.settings.mode="bm25";
        await this.plugin.saveSettings();this.display();
      }));
    new Setting(containerEl).setName("Vector storage")
      .setDesc("8-bit saves about 75% of vector space. 4-bit choices are experimental; changing this rebuilds embeddings in this vault.")
      .addDropdown(dropdown=>{
        const options:Record<string,string>={int8:"8-bit",f32:"Float32 (reference)",sq4:"4-bit scalar (experimental)","rotated-sq4":"4-bit rotated (experimental)"};
        options[policy.recommendedCodec]+=" (recommended)";
        dropdown.addOptions(options).setValue(this.plugin.settings.codec).onChange(async value=>{
          this.plugin.settings.codec=value as VectorCodec;await this.plugin.saveSettings();this.plugin.scheduleRefresh();
        });
      });
    new Setting(containerEl).setName("Indexing").setHeading();
    new Setting(containerEl).setName("Excluded folders")
      .setDesc("One vault-relative folder per line, such as Private or Attachments. Excluded notes are removed from this vault's index.")
      .addTextArea(text=>text.setPlaceholder("Private\nAttachments").setValue(this.plugin.settings.excludedFolders.join("\n")).onChange(async value=>{
        this.plugin.settings.excludedFolders=value.split("\n").map(x=>x.trim().replace(/^\/+|\/+$/g,"")).filter(Boolean);
        await this.plugin.saveSettings();this.plugin.scheduleRefresh();
      }));
    new Setting(containerEl).setName("Index status").setDesc(this.plugin.service.status)
      .addButton(button=>button.setButtonText(this.plugin.service.paused ? "Resume" : "Pause").onClick(()=>{
        this.plugin.togglePause();this.display();
      }))
      .addButton(button=>button.setButtonText("Rebuild index").onClick(()=>{this.plugin.refresh(true);this.display();}));
    const details=containerEl.createEl("details");details.createEl("summary",{text:"Local storage and privacy"});
    details.createEl("p",{text:`Shared model folder: ${sharedModelRoot()}`});
    details.createEl("p",{text:`Shared fast engine folder: ${sharedRuntimeRoot()}. It holds Microsoft's official onnxruntime-node libraries for ${this.plugin.runtimeStore.platform}, checked against pinned SHA-256 hashes when installed, plus the device choice for this computer.`});
    details.createEl("p",{text:"Each vault keeps its own text index and compressed vectors in this plugin's index folder. Notes from different vaults are never combined. Deleting or disabling the plugin in one vault does not delete the shared model. There is no telemetry."});
  }
}
