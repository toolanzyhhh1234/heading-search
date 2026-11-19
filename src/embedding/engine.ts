import { Tokenizer } from "@huggingface/tokenizers";

/** The subset of onnxruntime-node the engines use; the runtime is downloaded at setup, not bundled. */
export interface OrtTensor {readonly data:unknown;readonly dims:readonly number[];readonly type:string;dispose?():void}
export interface OrtSession {
  readonly inputNames:readonly string[];readonly outputNames:readonly string[];
  run(feeds:Record<string,OrtTensor>):Promise<Record<string,OrtTensor>>;
  release():Promise<void>;
}
export interface OrtModule {
  Tensor:new(type:"int64",data:BigInt64Array,dims:number[])=>OrtTensor;
  InferenceSession:{create(model:string|Uint8Array,options?:object):Promise<OrtSession>};
}
/**
 * tokens/rows bound one inference call. fixedShapes rounds batches to a small set of shapes:
 * DirectML compiles kernels per input shape, so length-sorted batches with ever-new shapes run 3–4x slower.
 */
export interface BatchLimits {tokens:number;rows:number;fixedShapes?:boolean}
/** What the engine needs to know about a model; the rest of a model spec (files, hashes) concerns storage. */
export interface ModelConfig {
  dimensions:number;
  /** Inputs are truncated to this many tokens, keeping the final special token. */
  maxTokens:number;
  pooling:"cls"|"mean";
  queryPrefix:string;
  documentPrefix?:string;
}
export const modelConfig=(spec:ModelConfig):ModelConfig=>
  ({dimensions:spec.dimensions,maxTokens:spec.maxTokens,pooling:spec.pooling,queryPrefix:spec.queryPrefix,documentPrefix:spec.documentPrefix??""});
const bucketWidth=(width:number,max:number):number=>Math.min(max,width<=128 ? Math.ceil(width/32)*32 : Math.ceil(width/64)*64);
const bucketRows=(rows:number):number=>2**Math.ceil(Math.log2(rows));

function halfToFloat(h:number):number {
  const sign=h&0x8000 ? -1 : 1,exponent=(h>>10)&31,mantissa=h&1023;
  if(exponent===0)return sign*mantissa*2**-24;
  if(exponent===31)return mantissa ? NaN : sign*Infinity;
  return sign*(1+mantissa/1024)*2**(exponent-15);
}

export interface Embedder {embed(texts:string[],query?:boolean):Promise<Float32Array[]>;dispose():Promise<void>}
export const createTokenizer=(tokenizer:object,config:object):Tokenizer=>new Tokenizer(tokenizer,config);

/** Tokenizes, groups similar lengths into padded batches, and pools each row into a normalized vector. */
export class EmbeddingEngine implements Embedder {
  constructor(private ort:OrtModule,private session:OrtSession,private tokenizer:Tokenizer,readonly limits:BatchLimits,readonly model:ModelConfig) {}

  encode(text:string,query:boolean):number[] {
    const tokens=this.tokenizer.encode((query ? this.model.queryPrefix : this.model.documentPrefix ?? "")+text).ids,max=this.model.maxTokens;
    // Truncation retains the leading and closing special tokens ([CLS] … [SEP]).
    return tokens.length<=max ? tokens : [...tokens.slice(0,max-1),tokens[tokens.length-1]];
  }

  async embed(texts:string[],query=false):Promise<Float32Array[]> {
    const order=texts.map((text,index)=>({index,ids:this.encode(text,query)})).sort((a,b)=>a.ids.length-b.ids.length);
    const vectors=new Array<Float32Array>(texts.length);
    const fixed=this.limits.fixedShapes===true,width=(length:number)=>fixed ? bucketWidth(length,this.model.maxTokens) : length;
    for(let start=0;start<order.length;) {
      // Sorted ascending, so the newest row sets the padded width of the whole batch.
      let end=start+1;
      while(end<order.length && end-start<this.limits.rows && (end-start+1)*width(order[end].ids.length)<=this.limits.tokens)end++;
      const batch=order.slice(start,end);
      const pooled=await this.run(batch.map(entry=>entry.ids),fixed);
      batch.forEach((entry,row)=>{vectors[entry.index]=pooled[row];});
      start=end;
    }
    return vectors;
  }

  private async run(texts:number[][],fixed:boolean):Promise<Float32Array[]> {
    const longest=Math.max(...texts.map(row=>row.length));
    // Filler rows are just the special tokens of an empty text; their outputs are discarded.
    const rows=fixed ? [...texts,...Array.from({length:bucketRows(texts.length)-texts.length},()=>this.tokenizer.encode("").ids)] : texts;
    const width=fixed ? bucketWidth(longest,this.model.maxTokens) : longest,count=rows.length,dims=[count,width];
    const ids=new BigInt64Array(count*width),mask=new BigInt64Array(count*width);
    rows.forEach((row,r)=>row.forEach((id,c)=>{ids[r*width+c]=BigInt(id);mask[r*width+c]=1n;}));
    const feeds:Record<string,OrtTensor>={input_ids:new this.ort.Tensor("int64",ids,dims),attention_mask:new this.ort.Tensor("int64",mask,dims)};
    if(this.session.inputNames.includes("token_type_ids"))feeds.token_type_ids=new this.ort.Tensor("int64",new BigInt64Array(count*width),dims);
    const outputs=await this.session.run(feeds);
    try {
      const name=this.session.outputNames.includes("last_hidden_state") ? "last_hidden_state" : this.session.outputNames[0];
      const output=outputs[name],data=output.data as Float32Array|Uint16Array,half=output.type==="float16";
      const value=(index:number)=>half ? halfToFloat(data[index]) : data[index];
      // Token states are [rows, width, hidden]; an already pooled output is [rows, hidden].
      const tokenStates=output.dims.length===3,hidden=output.dims[output.dims.length-1],dimensions=this.model.dimensions;
      if(hidden<dimensions)throw new Error(`The model outputs ${hidden} dimensions, fewer than the ${dimensions} expected.`);
      const stride=tokenStates ? output.dims[1]*hidden : hidden;
      return texts.map((tokens,r)=>{
        const vector=new Float32Array(dimensions);
        if(!tokenStates || this.model.pooling==="cls") {
          // Padding is masked, so the first token's state is unaffected by batch width.
          for(let i=0;i<dimensions;i++)vector[i]=value(r*stride+i);
        } else {
          // Mean over this row's real tokens only; padded positions are excluded.
          for(let t=0;t<tokens.length;t++)for(let i=0;i<dimensions;i++)vector[i]+=value(r*stride+t*hidden+i);
          for(let i=0;i<dimensions;i++)vector[i]/=tokens.length;
        }
        let norm=0;
        for(const component of vector)norm+=component*component;
        norm=Math.sqrt(norm)||1;
        for(let i=0;i<vector.length;i++)vector[i]/=norm;
        return vector;
      });
    } finally {for(const output of Object.values(outputs))output.dispose?.();}
  }

  dispose():Promise<void> {return this.session.release();}
}

/**
 * Cross-encoder reranker: scores (query, passage) pairs with one forward pass each. Shares the embedding engine's
 * batching, so CPU runs pairs one at a time and DirectML runs fixed-shape batches.
 */
export class RerankEngine {
  constructor(private ort:OrtModule,private session:OrtSession,private tokenizer:Tokenizer,readonly limits:BatchLimits,readonly maxTokens:number,private head?:ScoringHead) {}

  /** Token ids and segment ids for a pair; only the passage is shortened, so the whole query is always scored. */
  encodePair(query:string,passage:string):{ids:number[];types:number[]} {
    let text=passage;
    for(let attempt=0;attempt<6;attempt++) {
      const encoding=this.tokenizer.encode(query,{text_pair:text,return_token_type_ids:true});
      if(encoding.ids.length<=this.maxTokens || !text)return {ids:encoding.ids,types:encoding.token_type_ids};
      // Cut the passage in proportion to the overflow, with a little margin, and re-encode.
      const keep=Math.floor(text.length*(this.maxTokens/encoding.ids.length)*0.97);
      text=text.slice(0,Math.max(0,Math.min(keep,text.length-1)));
    }
    const encoding=this.tokenizer.encode(query,{text_pair:"",return_token_type_ids:true});
    return {ids:encoding.ids.slice(0,this.maxTokens),types:encoding.token_type_ids.slice(0,this.maxTokens)};
  }

  async score(query:string,passages:string[]):Promise<number[]> {
    const order=passages.map((passage,index)=>({index,...this.encodePair(query,passage)})).sort((a,b)=>a.ids.length-b.ids.length);
    const scores=new Array<number>(passages.length);
    const fixed=this.limits.fixedShapes===true,width=(length:number)=>fixed ? bucketWidth(length,this.maxTokens) : length;
    for(let start=0;start<order.length;) {
      let end=start+1;
      while(end<order.length && end-start<this.limits.rows && (end-start+1)*width(order[end].ids.length)<=this.limits.tokens)end++;
      const batch=order.slice(start,end);
      const results=await this.run(batch,fixed);
      batch.forEach((entry,row)=>{scores[entry.index]=results[row];});
      start=end;
    }
    return scores;
  }

  private async run(pairs:{ids:number[];types:number[]}[],fixed:boolean):Promise<number[]> {
    const filler=()=>{const e=this.tokenizer.encode("",{text_pair:"",return_token_type_ids:true});return {ids:e.ids,types:e.token_type_ids};};
    const rows=fixed ? [...pairs,...Array.from({length:bucketRows(pairs.length)-pairs.length},filler)] : pairs;
    const longest=Math.max(...pairs.map(pair=>pair.ids.length));
    const width=fixed ? bucketWidth(longest,this.maxTokens) : longest,count=rows.length,dims=[count,width];
    const ids=new BigInt64Array(count*width),mask=new BigInt64Array(count*width),types=new BigInt64Array(count*width);
    rows.forEach((row,r)=>row.ids.forEach((id,c)=>{ids[r*width+c]=BigInt(id);mask[r*width+c]=1n;types[r*width+c]=BigInt(row.types[c]??0);}));
    const feeds:Record<string,OrtTensor>={input_ids:new this.ort.Tensor("int64",ids,dims),attention_mask:new this.ort.Tensor("int64",mask,dims)};
    if(this.session.inputNames.includes("token_type_ids"))feeds.token_type_ids=new this.ort.Tensor("int64",types,dims);
    const outputs=await this.session.run(feeds);
    try {
      const output=outputs[this.session.outputNames.includes("logits") ? "logits" : this.session.outputNames[0]];
      const data=output.data as Float32Array|Uint16Array,half=output.type==="float16";
      const value=(index:number)=>half ? halfToFloat(data[index]) : data[index];
      if(output.dims.length===3) {
        // Encoder-only export: score the CLS state with the model's separately stored head.
        if(!this.head)throw new Error("This reranker outputs token states but has no scoring head.");
        const hidden=output.dims[2],stride=output.dims[1]*hidden;
        return pairs.map((_,r)=>applyHead(this.head!,Float32Array.from({length:hidden},(_,i)=>value(r*stride+i))));
      }
      // One relevance logit per pair; a two-label head scores with its "relevant" (last) column.
      const labels=output.dims.length>1 ? output.dims[output.dims.length-1] : 1;
      return pairs.map((_,r)=>value(r*labels+labels-1));
    } finally {for(const output of Object.values(outputs))output.dispose?.();}
  }

  dispose():Promise<void> {return this.session.release();}
}

/** Reads float32 tensors from a .safetensors file (8-byte little-endian header length, JSON header, raw data). */
export function parseSafetensors(bytes:Uint8Array):Record<string,{shape:number[];data:Float32Array}> {
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),length=Number(view.getBigUint64(0,true));
  const header=JSON.parse(new TextDecoder().decode(bytes.subarray(8,8+length))) as Record<string,{dtype:string;shape:number[];data_offsets:[number,number]}>;
  const tensors:Record<string,{shape:number[];data:Float32Array}>={};
  for(const [name,entry] of Object.entries(header)) {
    if(name==="__metadata__")continue;
    if(entry.dtype!=="F32")throw new Error(`Unsupported tensor type ${entry.dtype} for ${name}.`);
    const [start,end]=entry.data_offsets;
    // Copy: a Node Buffer's slice is a view on a shared pool, and its .buffer is the whole pool.
    tensors[name]={shape:entry.shape,data:new Float32Array(Uint8Array.prototype.slice.call(bytes,8+length+start,8+length+end).buffer)};
  }
  return tensors;
}

/** sentence-transformers cross-encoder head: CLS → Dense + GELU → LayerNorm → Dense to one score. */
export interface ScoringHead {dense:Float32Array;hidden:number;normWeight:Float32Array;normBias:Float32Array;scoreWeight:Float32Array;scoreBias:number}
export function scoringHead(dense:Uint8Array,norm:Uint8Array,score:Uint8Array):ScoringHead {
  const d=parseSafetensors(dense)["linear.weight"],n=parseSafetensors(norm),s=parseSafetensors(score);
  return {dense:d.data,hidden:d.shape[1],normWeight:n["norm.weight"].data,normBias:n["norm.bias"].data,scoreWeight:s["linear.weight"].data,scoreBias:s["linear.bias"].data[0]};
}
/** Exact GELU, x·Φ(x), with erf from Abramowitz–Stegun 7.1.26 (error below 1.5e-7). */
function gelu(x:number):number {
  const z=Math.abs(x)/Math.SQRT2,t=1/(1+0.3275911*z);
  const erf=1-((((1.061405429*t-1.453152027)*t+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-z*z);
  return 0.5*x*(1+(x<0 ? -erf : erf));
}
export function applyHead(head:ScoringHead,cls:Float32Array):number {
  const size=head.hidden,h=new Float32Array(size);
  for(let o=0;o<size;o++){let sum=0;for(let i=0;i<size;i++)sum+=head.dense[o*size+i]*cls[i];h[o]=gelu(sum);}
  let mean=0;for(const x of h)mean+=x;mean/=size;
  let variance=0;for(const x of h)variance+=(x-mean)**2;variance/=size;
  const inv=1/Math.sqrt(variance+1e-5);
  let score=head.scoreBias;
  for(let i=0;i<size;i++)score+=head.scoreWeight[i]*((h[i]-mean)*inv*head.normWeight[i]+head.normBias[i]);
  return score;
}
