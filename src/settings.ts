import type { VectorCodec } from "./core/vectors";
import policy from "./retrieval-policy.json";
export type SearchMode="bm25"|"vector"|"hybrid";
export interface Settings {
  mode:SearchMode;
  semanticEnabled:boolean;
  codec:VectorCodec;
  excludedFolders:string[];
  resultCount:number;
}
const recommendedCodec=(["f32","int8","sq4","rotated-sq4"].includes(policy.recommendedCodec) ? policy.recommendedCodec : "int8") as VectorCodec;
export const DEFAULT_SETTINGS:Settings={mode:"bm25",semanticEnabled:false,codec:recommendedCodec,excludedFolders:[],resultCount:20};
export function readSettings(value:unknown):Settings {
  const data=(value && typeof value==="object" ? value : {}) as Partial<Settings>;
  return {
    mode:["bm25","vector","hybrid"].includes(data.mode??"") ? data.mode! : DEFAULT_SETTINGS.mode,
    semanticEnabled:data.semanticEnabled===true,
    codec:["f32","int8","sq4","rotated-sq4"].includes(data.codec??"") ? data.codec! : DEFAULT_SETTINGS.codec,
    excludedFolders:Array.isArray(data.excludedFolders) ? data.excludedFolders.filter((x):x is string=>typeof x==="string").map(x=>x.trim().replace(/^\/+|\/+$/g,"")).filter(Boolean) : [],
    resultCount:Number.isInteger(data.resultCount) && data.resultCount!>=5 && data.resultCount!<=100 ? data.resultCount! : 20,
  };
}
