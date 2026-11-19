import { createGunzip } from "node:zlib";

/**
 * Extracts selected regular files from a gzipped tar archive (npm package format).
 * Only matching entries are buffered, so a large multi-platform package never sits fully in memory.
 * Links, directories, and metadata entries are never materialized.
 */
export function extractTarGz(archive:Uint8Array,wanted:(path:string)=>boolean):Promise<Map<string,Uint8Array>> {
  return new Promise((resolve,reject)=>{
    const files=new Map<string,Uint8Array>();
    const gunzip=createGunzip();
    let pending:Buffer=Buffer.alloc(0),remaining=0,padding=0,ended=false;
    let sink:"skip"|"file"|"longname"="skip",name="",parts:Buffer[]=[],longName:string|null=null;
    const text=(header:Buffer,start:number,length:number)=>{
      const slice=header.subarray(start,start+length),end=slice.indexOf(0);
      return slice.toString("utf8",0,end<0?length:end);
    };
    const complete=()=>{
      if(sink==="file")files.set(name,Uint8Array.from(Buffer.concat(parts)));
      else if(sink==="longname")longName=Buffer.concat(parts).toString("utf8").replace(/\0+$/,"");
      sink="skip";parts=[];
    };
    gunzip.on("data",(chunk:Buffer)=>{
      if(ended)return;
      pending=pending.length ? Buffer.concat([pending,chunk]) : chunk;
      let offset=0;
      while(offset<pending.length) {
        if(remaining>0) {
          const take=Math.min(remaining,pending.length-offset);
          if(sink!=="skip")parts.push(pending.subarray(offset,offset+take));
          offset+=take;remaining-=take;
          if(remaining===0)complete();
          continue;
        }
        if(padding>0) {const take=Math.min(padding,pending.length-offset);offset+=take;padding-=take;continue;}
        if(pending.length-offset<512)break;
        const header=pending.subarray(offset,offset+512);offset+=512;
        if(header.every(byte=>byte===0)) {ended=true;break;}
        const size=parseInt(text(header,124,12).trim()||"0",8),type=header[156]===0 ? "0" : String.fromCharCode(header[156]);
        if(!Number.isFinite(size) || size<0) {reject(new Error("Corrupt runtime archive."));gunzip.destroy();return;}
        const prefix=text(header,345,155),path=longName ?? (prefix ? `${prefix}/${text(header,0,100)}` : text(header,0,100));
        remaining=size;padding=(512-size%512)%512;
        if(type==="L") sink="longname";
        else {longName=null;name=path;sink=type==="0" && wanted(path) ? "file" : "skip";}
        if(size===0)complete();
      }
      pending=Buffer.from(pending.subarray(offset));
    });
    gunzip.on("end",()=>resolve(files));
    gunzip.on("error",reject);
    gunzip.end(Buffer.from(archive.buffer,archive.byteOffset,archive.byteLength));
  });
}
