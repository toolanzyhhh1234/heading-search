import { Modal, Notice, type App } from "obsidian";
import type { SearchMode, Settings } from "../settings";
import type { SearchResult, SearchService } from "../index/service";

export class SearchModal extends Modal {
  private generation=0;
  private timer?:ReturnType<typeof setTimeout>;
  private results:SearchResult[]=[];
  private selected=0;
  private list!:HTMLDivElement;
  private message!:HTMLDivElement;
  private mode:SearchMode;
  constructor(app:App,private service:SearchService,private settings:Settings) {
    super(app);this.mode=settings.mode;
  }
  onOpen():void {
    this.modalEl.addClass("heading-search-modal");
    this.titleEl.setText("Search your notes");
    const controls=this.contentEl.createDiv({cls:"heading-search-controls"});
    const input=controls.createEl("input",{type:"search",placeholder:"Find a topic, idea, or phrase…",attr:{"aria-label":"Search notes"}});
    const select=controls.createEl("select",{attr:{"aria-label":"Search method"}});
    for(const [value,label] of [["bm25","Keyword (BM25)"],["vector","Semantic"],["hybrid","Hybrid"]]) {
      const option=select.createEl("option",{text:label,value});
      option.disabled=value!=="bm25" && !this.settings.semanticEnabled;
    }
    select.value=this.mode;
    this.message=this.contentEl.createDiv({cls:"heading-search-message",attr:{role:"status","aria-live":"polite"}});
    this.message.setText(`${this.service.chunkCount} sections indexed. Type to search.`);
    this.list=this.contentEl.createDiv({cls:"heading-search-results",attr:{role:"listbox","aria-label":"Search results"}});
    const schedule=()=>{
      // Invalidate already running requests as soon as the text changes.
      const generation=++this.generation;
      clearTimeout(this.timer);
      this.timer=setTimeout(()=>{void this.run(input.value,generation);},180);
    };
    input.addEventListener("input",schedule);
    select.addEventListener("change",()=>{this.mode=select.value as SearchMode;schedule();input.focus();});
    input.addEventListener("keydown",event=>{
      if(event.key==="ArrowDown" || event.key==="ArrowUp") {
        event.preventDefault();this.selected=Math.max(0,Math.min(this.results.length-1,this.selected+(event.key==="ArrowDown" ? 1 : -1)));this.highlight();
      } else if(event.key==="Enter" && this.results[this.selected]) {event.preventDefault();void this.openResult(this.results[this.selected],event.ctrlKey||event.metaKey);}
    });
    this.contentEl.createDiv({cls:"heading-search-help",text:"↑ ↓ to select · Enter to open · Ctrl/Cmd + Enter for a new tab"});
    input.focus();
  }
  private async run(query:string,generation:number):Promise<void> {
    if(generation!==this.generation)return;
    if(!query.trim()) {this.results=[];this.list.empty();this.message.setText("Type to search your notes.");return;}
    this.message.setText("Searching…");
    try {
      const results=await this.service.search(query,this.mode);
      if(generation!==this.generation)return;
      this.results=results;this.selected=0;this.list.empty();
      this.message.setText(results.length ? `${results.length} matching notes` : "No matching notes. Try another phrase or search method.");
      for(const result of results) {
        const row=this.list.createEl("button",{cls:"heading-search-result",attr:{role:"option","aria-selected":"false"}});
        row.createDiv({cls:"heading-search-result-title",text:result.chunk.title});
        row.createDiv({cls:"heading-search-breadcrumb",text:result.chunk.headings.map(h=>h.text).join(" › ") || result.chunk.path});
        row.createDiv({cls:"heading-search-snippet",text:result.chunk.text.replace(/\s+/g," ").slice(0,230)});
        row.addEventListener("click",event=>{void this.openResult(result,event.ctrlKey||event.metaKey);});
      }
      this.highlight();
    } catch(error) {if(generation===this.generation){this.list.empty();this.results=[];this.message.setText(error instanceof Error ? error.message : "Search failed. Try again.");}}
  }
  private highlight():void {
    Array.from(this.list.children).forEach((element,index)=>{
      element.toggleClass("is-selected",index===this.selected);
      element.setAttribute("aria-selected",String(index===this.selected));
      if(index===this.selected)element.scrollIntoView({block:"nearest"});
    });
  }
  private async openResult(result:SearchResult,newTab=false):Promise<void> {
    const file=this.app.vault.getFileByPath(result.chunk.path);
    if(!file){new Notice("That note has moved. The index will update shortly.");return;}
    await this.app.workspace.getLeaf(newTab ? "tab" : false).openFile(file,{eState:{line:result.chunk.line}});
    this.close();
  }
  onClose():void {++this.generation;clearTimeout(this.timer);this.contentEl.empty();}
}
