import type { Ranked } from "./bm25";
export type VectorCodec = "f32" | "int8" | "sq4" | "rotated-sq4";
const CODECS: VectorCodec[] = ["f32", "int8", "sq4", "rotated-sq4"];
// Lloyd-Max centroids for a standard Gaussian; fixed, not fitted on test data.
const CENTERS = [-2.7325896,-2.0690172,-1.6180464,-1.2562312,-0.9423403,-0.6567591,-0.3880483,-0.1283950,
  0.1283950,0.3880483,0.6567591,0.9423403,1.2562312,1.6180464,2.0690172,2.7325896];

function paddedDimensions(dim: number): number { return 2 ** Math.ceil(Math.log2(dim)); }
export function rotate(vector: Float32Array): Float32Array {
  const output = new Float32Array(paddedDimensions(vector.length));
  let seed = 0x12345678;
  for (let i=0;i<output.length;i++) {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    output[i] = (vector[i] ?? 0) * ((seed & 1) ? 1 : -1);
  }
  for (let width=1;width<output.length;width*=2) {
    for (let block=0;block<output.length;block+=2*width) {
      for (let offset=0;offset<width;offset++) {
        const a=output[block+offset], b=output[block+offset+width];
        output[block+offset]=a+b; output[block+offset+width]=a-b;
      }
    }
  }
  const scale=1/Math.sqrt(output.length);
  for(let i=0;i<output.length;i++) output[i]*=scale;
  return output;
}

function strideFor(codec: VectorCodec, dims: number): number {
  return codec === "f32" ? dims*4 : codec === "int8" ? dims+8 : codec === "sq4" ? Math.ceil(dims/2)+8 : paddedDimensions(dims)/2+4;
}

/** Compact binary data: no JSON decimal arrays or base64 expansion on disk. */
export function packVectors(vectors: Float32Array[], codec: VectorCodec, dimensions = vectors[0]?.length ?? 384): Uint8Array {
  if (!CODECS.includes(codec) || dimensions < 1 || dimensions > 8192) throw new Error("Invalid vector format.");
  const stride = strideFor(codec,dimensions);
  const bytes = new Uint8Array(16 + stride*vectors.length);
  const view=new DataView(bytes.buffer);
  bytes.set([72,86,83,49]); view.setUint8(4,CODECS.indexOf(codec)); view.setUint16(6,dimensions,true);
  view.setUint32(8,vectors.length,true); view.setUint32(12,stride,true);
  vectors.forEach((original,row)=>{
    if(original.length!==dimensions || original.some(x=>!Number.isFinite(x))) throw new Error("Invalid embedding dimensions or values.");
    const base=16+row*stride;
    if(codec==="f32") {
      for(let i=0;i<dimensions;i++) view.setFloat32(base+4*i,original[i],true);
      return;
    }
    const vector=codec==="rotated-sq4" ? rotate(original) : original;
    let max=0;
    for(const value of vector) max=Math.max(max,Math.abs(value));
    const scale=codec==="rotated-sq4" ? 1/Math.sqrt(vector.length) : (max || 1)/(codec==="int8" ? 127 : 7);
    let norm=0;
    const offset=codec==="rotated-sq4" ? 4 : 8;
    if(offset===8) view.setFloat32(base,scale,true);
    for(let i=0;i<vector.length;i++) {
      let code:number, reconstructed:number;
      if(codec==="rotated-sq4") {
        const value=vector[i]/scale;
        code=0;
        while(code<15 && value>(CENTERS[code]+CENTERS[code+1])/2) code++;
        reconstructed=CENTERS[code]*scale;
      } else {
        const quantized=Math.round(vector[i]/scale);
        code=codec==="int8" ? quantized : quantized+7;
        reconstructed=quantized*scale;
      }
      norm+=reconstructed*reconstructed;
      if(codec==="int8") view.setInt8(base+offset+i,code);
      else bytes[base+offset+(i>>1)] |= code << ((i&1)*4);
    }
    view.setFloat32(base+(offset===8 ? 4 : 0),norm ? 1/Math.sqrt(norm) : 1,true);
  });
  return bytes;
}

export class PackedVectors {
  readonly codec:VectorCodec;
  readonly dimensions:number;
  readonly count:number;
  readonly stride:number;
  private view:DataView;
  constructor(readonly bytes:Uint8Array, readonly ids:string[]) {
    this.view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    if(bytes.length<16 || bytes.slice(0,4).toString()!=="72,86,83,49") throw new Error("Invalid vector index header.");
    this.codec=CODECS[this.view.getUint8(4)];
    this.dimensions=this.view.getUint16(6,true); this.count=this.view.getUint32(8,true); this.stride=this.view.getUint32(12,true);
    if(!this.codec || this.dimensions<1 || this.dimensions>8192 || this.stride!==strideFor(this.codec,this.dimensions) ||
      bytes.length!==16+this.stride*this.count || ids.length!==this.count) throw new Error("Vector index is truncated or incompatible.");
  }

  search(query:Float32Array, limit=20):Ranked[] {
    if(query.length!==this.dimensions || query.some(x=>!Number.isFinite(x))) throw new Error("Query dimensions do not match the vector index.");
    const q=this.codec==="rotated-sq4" ? rotate(query) : query;
    const result:Ranked[]=[];
    for(let row=0;row<this.count;row++) {
      const score=this.scoreRow(row,q);
      if(result.length===limit && score<=result[result.length-1].score) continue;
      result.push({id:this.ids[row],score}); result.sort((a,b)=>b.score-a.score || a.id.localeCompare(b.id));
      if(result.length>limit) result.pop();
    }
    return result;
  }

  /** Copy encoded rows when incrementally updating a vault; never retain float32 copies. */
  rowBytes(row:number):Uint8Array { return this.bytes.slice(16+row*this.stride,16+(row+1)*this.stride); }

  private scoreRow(row:number,query:Float32Array):number {
    const base=16+row*this.stride;
    let score=0;
    if(this.codec==="f32") {
      for(let i=0;i<query.length;i++) score+=query[i]*this.view.getFloat32(base+4*i,true);
      return score;
    }
    const offset=this.codec==="rotated-sq4" ? 4 : 8;
    const scale=offset===4 ? 1/Math.sqrt(query.length) : this.view.getFloat32(base,true);
    for(let i=0;i<query.length;i++) {
      const code=this.codec==="int8" ? this.view.getInt8(base+offset+i) : (this.bytes[base+offset+(i>>1)]>>((i&1)*4))&15;
      score+=query[i]*(this.codec==="rotated-sq4" ? CENTERS[code] : this.codec==="sq4" ? code-7 : code);
    }
    return score*scale*this.view.getFloat32(base+(offset===8 ? 4 : 0),true);
  }
}

export function joinVectorRows(rows:Uint8Array[], codec:VectorCodec, dims=384):Uint8Array {
  const header=packVectors([],codec,dims);
  const stride=strideFor(codec,dims);
  const bytes=new Uint8Array(16+rows.length*stride); bytes.set(header);
  new DataView(bytes.buffer).setUint32(8,rows.length,true);
  rows.forEach((row,i)=>{
    if(row.length!==stride) throw new Error("Incompatible encoded vector row.");
    bytes.set(row,16+i*stride);
  });
  return bytes;
}
