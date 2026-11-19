import { promises as fs } from "node:fs";
import { createHash } from "node:crypto";
import { join } from "node:path";

export type Download = (url: string) => Promise<ArrayBuffer>;
export type Progress = (message: string) => void;
export interface PinnedFile {target:string;bytes:number;sha256:string}

export const sha256=(data:Uint8Array):string=>createHash("sha256").update(data).digest("hex");
export const sleep=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms));

export async function ensureDirectory(path:string):Promise<void> {
  for(let attempt=0;attempt<5;attempt++) {
    try {await fs.mkdir(path,{recursive:true});return;}
    catch(error) {
      // Concurrent recursive mkdir can transiently return EPERM on Windows.
      try {if((await fs.stat(path)).isDirectory())return;} catch {}
      if((error as NodeJS.ErrnoException).code!=="EPERM" || attempt===4)throw error;
      await sleep(25);
    }
  }
}
export async function promote(source:string,target:string):Promise<void> {
  for(let attempt=0;attempt<8;attempt++) {
    try {await fs.rename(source,target);return;}
    catch(error) {
      // Virus scanners and other vaults can briefly hold a Windows file handle.
      if(!["EPERM","EACCES","EBUSY"].includes((error as NodeJS.ErrnoException).code??"") || attempt===7)throw error;
      await sleep(25*2**attempt);
    }
  }
}

/** Full integrity check: every byte of every pinned file. Used during setup and repair. */
export async function verifyFiles(files:PinnedFile[]):Promise<boolean> {
  for(const file of files) {
    try {
      const data=await fs.readFile(file.target);
      if(data.length!==file.bytes || sha256(data)!==file.sha256)return false;
    } catch {return false;}
  }
  return true;
}

/**
 * Startup check: compare size and modification time with the stamp written after a full verification.
 * Hashing ~200 MB of weights and runtime on every Obsidian launch would stall the interface.
 */
export async function stampMatches(stamp:string,files:PinnedFile[]):Promise<boolean> {
  try {
    const recorded=JSON.parse(await fs.readFile(stamp,"utf8")) as Record<string,{bytes:number;mtimeMs:number;sha256:string}>;
    for(const file of files) {
      const entry=recorded[file.target],stat=await fs.stat(file.target);
      if(!entry || entry.sha256!==file.sha256 || stat.size!==file.bytes || Math.abs(stat.mtimeMs-entry.mtimeMs)>1)return false;
    }
    return true;
  } catch {return false;}
}
export async function writeStamp(stamp:string,files:PinnedFile[]):Promise<void> {
  const recorded:Record<string,{bytes:number;mtimeMs:number;sha256:string}>={};
  for(const file of files)recorded[file.target]={bytes:file.bytes,mtimeMs:(await fs.stat(file.target)).mtimeMs,sha256:file.sha256};
  const temporary=`${stamp}.${process.pid}.${Date.now()}.partial`;
  await fs.writeFile(temporary,JSON.stringify(recorded));
  await promote(temporary,stamp);
}

/** A cross-vault lease: simultaneous setup requests in several vaults download once. */
export async function withDownloadLock<T>(directory:string,progress:Progress,what:string,work:()=>Promise<T>):Promise<T> {
  await ensureDirectory(directory);
  const lock=join(directory,".download-lock"),started=Date.now();
  while(true) {
    try {await fs.mkdir(lock);break;}
    catch(error) {
      if((error as NodeJS.ErrnoException).code!=="EEXIST")throw error;
      // A crashed process stops touching the lease. Active downloads keep it fresh.
      try {
        const stat=await fs.stat(lock);
        if(Date.now()-stat.mtimeMs>120_000) {await fs.rmdir(lock).catch(()=>{});continue;}
      } catch {}
      if(Date.now()-started>600_000)throw new Error(`Another vault is still downloading the ${what}. Try again shortly.`);
      progress(`Waiting for another vault to finish downloading the ${what}…`);
      await sleep(500);
    }
  }
  const heartbeat=setInterval(()=>{const now=new Date();void fs.utimes(lock,now,now).catch(()=>{});},10_000);
  try {return await work();}
  finally {clearInterval(heartbeat);await fs.rmdir(lock).catch(()=>{});}
}
