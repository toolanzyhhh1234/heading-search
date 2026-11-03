import { test } from "node:test";
import assert from "node:assert/strict";
import { packVectors, PackedVectors, rotate, joinVectorRows, type VectorCodec } from "../src/core/vectors";

function normalized(values:number[]):Float32Array {
  const norm=Math.hypot(...values);return Float32Array.from(values.map(x=>x/(norm||1)));
}
for(const codec of ["f32","int8","sq4","rotated-sq4"] as VectorCodec[]) {
  test(`${codec}: packed retrieval preserves nearby vs opposite vectors`,()=>{
    const vector=normalized(Array.from({length:384},(_,i)=>Math.sin(i*0.7)));
    const other=normalized(Array.from(vector,x=>-x));
    const zero=new Float32Array(384);
    const bytes=packVectors([vector,other,zero],codec);
    const index=new PackedVectors(bytes,["near","opposite","zero"]);
    const result=index.search(vector);
    assert.equal(result[0].id,"near");assert.ok(result[0].score>0.97);assert.equal(result[2].id,"opposite");
    const rebuilt=new PackedVectors(joinVectorRows([index.rowBytes(1),index.rowBytes(0)],codec),["opposite","near"]);
    assert.equal(rebuilt.search(vector)[0].id,"near");
  });
}
test("binary sizes include per-vector metadata and the header",()=>{
  const vectors=[new Float32Array(384),new Float32Array(384)];
  assert.equal(packVectors(vectors,"f32").length,16+2*1536);
  assert.equal(packVectors(vectors,"int8").length,16+2*392);
  assert.equal(packVectors(vectors,"sq4").length,16+2*200);
  assert.equal(packVectors(vectors,"rotated-sq4").length,16+2*260);
});
test("rotation preserves dot products despite padding and is deterministic",()=>{
  const a=normalized([1,2,3]),b=normalized([2,-1,4]);
  const ra=rotate(a),rb=rotate(b);
  const dot=(x:Float32Array,y:Float32Array)=>x.reduce((sum,value,i)=>sum+value*y[i],0);
  assert.ok(Math.abs(dot(a,b)-dot(ra,rb))<1e-6);assert.deepEqual(ra,rotate(a));
});
test("invalid or truncated storage and mismatched embeddings fail safely",()=>{
  const bytes=packVectors([normalized([1,2,3])],"int8");
  assert.throws(()=>new PackedVectors(bytes.slice(0,-1),["x"]));
  assert.throws(()=>new PackedVectors(bytes,[]));
  assert.throws(()=>new PackedVectors(bytes,["x"]).search(new Float32Array(4)));
  assert.throws(()=>packVectors([Float32Array.from([NaN])],"sq4"));
});
