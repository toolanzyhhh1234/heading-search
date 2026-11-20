// Note file names for BEIR documents, shared by the Omnisearch reproduction and the live Omnisearch vaults,
// so both index exactly the same notes. Obsidian file names cannot contain \ / : * ? " < > | # ^ [ ];
// Windows drops trailing dots and spaces; long titles are cut to stay well inside Windows path limits.
export const MAX_NAME=150;
// Omnisearch HTTP API port per benchmark vault (51361 is Omnisearch's default).
export const LIVE_PORTS={scifact:51361,nfcorpus:51362};
export function noteNames(corpus) {
  const used=new Set();
  return corpus.map(d=>{
    let name=d.title.replace(/[\\/:*?"<>|#^[\]]/g,' ').replace(/\s+/g,' ').trim();
    if(name.length>MAX_NAME) { name=name.slice(0,MAX_NAME); const cut=name.lastIndexOf(' '); if(cut>MAX_NAME*0.6) name=name.slice(0,cut); }
    name=name.replace(/[. ]+$/,'')||String(d._id);
    let unique=name;
    for(let n=2;used.has(unique.toLowerCase());n++) unique=`${name} (${n})`;
    used.add(unique.toLowerCase());
    return unique;
  });
}
