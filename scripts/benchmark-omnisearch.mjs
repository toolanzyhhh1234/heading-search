// Compare Heading Search with Obsidian's core search and Omnisearch on SciFact and NFCorpus.
// Curiosity benchmark, not a decision gate: BEIR documents are single abstracts without headings,
// so this measures whole-note ranking only, which is what Omnisearch and the core search do.
//
// Omnisearch is reproduced from its source (v1.32.0, commit e4a51ab) with its own libraries, installed outside package.json:
//   npm install --prefix .cache/omnisearch-bench --ignore-scripts --no-save minisearch@7.1.0 search-query-parser@1.6.0 markdown-link-extractor@4.0.4
// Each abstract is treated as a note named after its title (scripts/beir-notes.mjs), with the abstract as its content.
// Usage: node scripts/benchmark-omnisearch.mjs [--embeddings granite-small-r2-fp32] [--live]
// --live also queries the real Omnisearch plugin over its HTTP API and checks the reproduction against it:
// run node scripts/setup-omnisearch-vaults.mjs, open both vaults in Obsidian, and wait for indexing.
import { build } from 'esbuild';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { noteNames, LIVE_PORTS } from './beir-notes.mjs';

const require=createRequire(new URL('../.cache/omnisearch-bench/package.json',import.meta.url));
const MiniSearch=require('minisearch').default??require('minisearch');
const { parse }=require('search-query-parser');
const markdownLinkExtractor=require('markdown-link-extractor');

await mkdir('dist',{recursive:true});
await build({entryPoints:['src/core/bm25.ts','src/core/vectors.ts','src/core/metrics.ts'],outdir:'dist/core',bundle:true,platform:'node',format:'esm'});
const { BM25, reciprocalRankFusion }=await import('../dist/core/bm25.js');
const { packVectors, PackedVectors }=await import('../dist/core/vectors.js');
const { evaluate, pairedBootstrap }=await import('../dist/core/metrics.js');
const option=name=>{const i=process.argv.indexOf(`--${name}`);return i>0?process.argv[i+1]:undefined;};
const spec=JSON.parse(await readFile('src/model-spec.json','utf8'));
const profile=option('embeddings')??`${spec.key}-fp32`;

// ---- Omnisearch 1.32.0, default settings (src/search/{search-engine,tokenizer,query}.ts, src/globals.ts) ----
const SEPARATORS=/[|\t\n\r^"= -#%-*,.`/<>:;?@[-\]_{}\u00A0\u00A1\u00A7\u00AB\u00B6\u00B7\u00BB\u00BF\u037E\u0387\u055A-\u055F\u0589\u058A\u05BE\u05C0\u05C3\u05C6\u05F3\u05F4\u0609\u060A\u060C\u060D\u061B\u061E\u061F\u066A-\u066D\u06D4\u0700-\u070D\u07F7-\u07F9\u0830-\u083E\u085E\u0964\u0965\u0970\u09FD\u0A76\u0AF0\u0C77\u0C84\u0DF4\u0E4F\u0E5A\u0E5B\u0F04-\u0F12\u0F14\u0F3A-\u0F3D\u0F85\u0FD0-\u0FD4\u0FD9\u0FDA\u104A-\u104F\u10FB\u1360-\u1368\u1400\u166E\u1680\u169B\u169C\u16EB-\u16ED\u1735\u1736\u17D4-\u17D6\u17D8-\u17DA\u1800-\u180A\u1944\u1945\u1A1E\u1A1F\u1AA0-\u1AA6\u1AA8-\u1AAD\u1B5A-\u1B60\u1BFC-\u1BFF\u1C3B-\u1C3F\u1C7E\u1C7F\u1CC0-\u1CC7\u1CD3\u2000-\u200A\u2010-\u2029\u202F-\u2043\u2045-\u2051\u2053-\u205F\u207D\u207E\u208D\u208E\u2308-\u230B\u2329\u232A\u2768-\u2775\u27C5\u27C6\u27E6-\u27EF\u2983-\u2998\u29D8-\u29DB\u29FC\u29FD\u2CF9-\u2CFC\u2CFE\u2CFF\u2D70\u2E00-\u2E2E\u2E30-\u2E4F\u3000-\u3003\u3008-\u3011\u3014-\u301F\u3030\u303D\u30A0\u30FB\uA4FE\uA4FF\uA60D-\uA60F\uA673\uA67E\uA6F2-\uA6F7\uA874-\uA877\uA8CE\uA8CF\uA8F8-\uA8FA\uA8FC\uA92E\uA92F\uA95F\uA9C1-\uA9CD\uA9DE\uA9DF\uAA5C-\uAA5F\uAADE\uAADF\uAAF0\uAAF1\uABEB\uFD3E\uFD3F\uFE10-\uFE19\uFE30-\uFE52\uFE54-\uFE61\uFE63\uFE68\uFE6A\uFE6B\uFF01-\uFF03\uFF05-\uFF0A\uFF0C-\uFF0F\uFF1A\uFF1B\uFF1F\uFF20\uFF3B-\uFF3D\uFF3F\uFF5B\uFF5D\uFF5F-\uFF65]/.toString().slice(1,-1);
const SPACE_OR_PUNCTUATION=new RegExp(`${SEPARATORS}+`,'u');
const BRACKETS_AND_SPACE=/[|()<>{}[\] \t\n\r]/u;
// Omnisearch's defaults as shipped (getDefaultSettings in src/settings/index.ts); its online docs list older values.
const OMNI={ignoreDiacritics:true,splitCamelCase:false,tokenizeUrls:false,simpleSearch:false,fuzziness:'1',
  weightBasename:10,weightDirectory:7,weightH1:6,weightH2:5,weightH3:4,weightUnmarkedTags:2};
const splitHyphens=text=>text.includes('-')?text.split('-').filter(t=>t):[];
const splitCamelCase=text=>/[a-z][A-Z]/.test(text)?text.replace(/([a-z](?=[A-Z]))/g,'$1 ').split(' ').filter(t=>t):[];
const removeBase64Images=text=>text.replace(/data:[^;)]*;base64,[A-Za-z0-9+/=_-]*/g,'');
const japaneseDiacritics=[0x30FC,0x309A,0x3099].map(c=>String.fromCharCode(c)).join('|');
const diacriticsRegex=new RegExp(`(?!${japaneseDiacritics})\\p{Diacritic}`,'gu');
const removeDiacritics=str=>str.replaceAll('`','[__omnisearch__backtick__]').replaceAll('^','[__omnisearch__caret__]')
  .normalize('NFD').replace(diacriticsRegex,'').normalize('NFC')
  .replaceAll('[__omnisearch__backtick__]','`').replaceAll('[__omnisearch__caret__]','^');
const stripMarkdownCharacters=text=>text.replace(/(\*|_)+(.+?)(\*|_)+/g,(_m,_p1,p2)=>p2);
const normalizeExactMatchContent=text=>stripMarkdownCharacters(removeDiacritics(text)).toLowerCase();
function tokenizeForIndexing(text) {
  text=removeBase64Images(text);
  const words=text.split(BRACKETS_AND_SPACE);
  let urls=[];
  if(OMNI.tokenizeUrls) try{urls=markdownLinkExtractor(text);}catch{}
  let tokens=text.split(SPACE_OR_PUNCTUATION);
  tokens=[...tokens.flatMap(t=>[t,...splitHyphens(t),...(OMNI.splitCamelCase?splitCamelCase(t):[])]),...words];
  if(urls.length) tokens=[...tokens,...urls];
  return tokens.filter(Boolean);
}
function tokenizeForSearch(text,combine='AND') {
  const urls=markdownLinkExtractor(text);
  text=urls.reduce((acc,url)=>acc.replace(url,''),text);
  const tokens=[...text.split(SPACE_OR_PUNCTUATION),...urls].filter(Boolean);
  return {combineWith:'OR',queries:[
    {combineWith:combine,queries:tokens},
    {combineWith:combine,queries:text.split(BRACKETS_AND_SPACE).filter(Boolean)},
    {combineWith:combine,queries:tokens.flatMap(splitHyphens)},
    ...(OMNI.splitCamelCase?[{combineWith:combine,queries:tokens.flatMap(splitCamelCase)}]:[]),
  ]};
}
function parseQuery(text) {
  if(OMNI.ignoreDiacritics) text=removeDiacritics(text);
  const parsed=parse(text.toLowerCase(),{tokenize:true,keywords:['ext','path']});
  const words=(typeof parsed==='string'?[parsed]:parsed.text??[]).filter(o=>!o.startsWith('.'));
  let exclude=typeof parsed==='string'?[]:parsed.exclude?.text??[];
  if(!Array.isArray(exclude)) exclude=[exclude];
  const quoted=text.match(/"([^"]+)"/g)?.map(o=>o.replace(/"/g,''))??[];
  return {words,exclude:exclude.filter(o=>o.length),exact:[...new Set([...words.filter(o=>o.split(' ').length>1),...quoted].map(s=>s.toLowerCase()))]};
}
const omniWeights={basename:OMNI.weightBasename,aliases:OMNI.weightBasename,displayTitle:OMNI.weightBasename,directory:OMNI.weightDirectory,
  headings1:OMNI.weightH1,headings2:OMNI.weightH2,headings3:OMNI.weightH3,unmarkedTags:OMNI.weightUnmarkedTags};
const fuzziness={'0':0,'1':0.1}[OMNI.fuzziness]??0.2;
// combine='OR' is a diagnostic, not an Omnisearch setting: the same ranking without requiring every word.
function omnisearchIndex(notes,combine='AND') {
  const engine=new MiniSearch({
    tokenize:tokenizeForIndexing,
    extractField:(doc,field)=>field==='directory'?doc.path.split('/').slice(0,-1).join('/'):doc[field],
    processTerm:term=>(OMNI.ignoreDiacritics?removeDiacritics(term):term).toLowerCase(),
    idField:'path',
    fields:['basename','directory','aliases','content','headings1','headings2','headings3','unmarkedTags'],
  });
  engine.addAll(notes);
  const byPath=new Map(notes.map(n=>[n.path,n]));
  return query=>{
    const q=parseQuery(query);
    if(!q.words.length) return [];
    const str=q.words.join(' ');
    let results=engine.search(tokenizeForSearch(str,combine),{
      prefix:term=>term.length>=(OMNI.simpleSearch?3:1),
      fuzzy:term=>term.length<=3?0:term.length<=5?fuzziness/2:fuzziness,
      boost:omniWeights, tokenize:text=>[text],
    });
    const queryStr=str.toLowerCase();
    for(const r of results) {
      const title=(r.id.split('/').pop()??'').replace(/\.[^.]+$/,'');
      if(title.toLowerCase().startsWith(queryStr)) r.score*=100;
    }
    results=results.sort((a,b)=>b.score-a.score).slice(0,50);
    if(q.exact.length) results=results.filter(r=>{const content=normalizeExactMatchContent(byPath.get(r.id).content);
      return q.exact.every(e=>content.includes(e)||removeDiacritics(r.id.toLowerCase()).includes(e));});
    if(q.exclude.length) results=results.filter(r=>q.exclude.every(e=>!byPath.get(r.id).content.toLowerCase().includes(e)));
    return results.map(r=>({id:byPath.get(r.id).id,score:r.score}));
  };
}

// ---- Obsidian core search, default settings: every space-separated word must occur (case-insensitive substring)
// in the file name or content; results sorted by file name A to Z. Quotes and operators are not used by these queries.
function coreSearch(notes) {
  const prepared=notes.map(n=>({id:n.id,name:n.basename,haystack:`${n.basename}\n${n.content}`.toLowerCase()}))
    .sort((a,b)=>a.name.localeCompare(b.name,'en'));
  return query=>{
    const words=query.toLowerCase().split(/\s+/).filter(Boolean);
    return prepared.filter(n=>words.every(w=>n.haystack.includes(w))).map(n=>({id:n.id,score:0}));
  };
}

async function cachedEmbeddings(key,texts,query=false) {
  const digest=createHash('sha256').update(spec.revision+JSON.stringify(texts)+String(query)).digest('hex').slice(0,16);
  const bytes=await readFile(`benchmarks/cache/${key}-${digest}.${profile}.f32`).catch(()=>{throw new Error(`Missing ${profile} embedding cache for ${key}. Run node scripts/embed-benchmarks-native.mjs first.`);});
  const values=new Float32Array(Uint8Array.from(bytes).buffer);
  return texts.map((_,i)=>values.slice(i*spec.dimensions,(i+1)*spec.dimensions));
}
const average=values=>values.reduce((a,b)=>a+b,0)/values.length;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function liveSearch(port,q) {
  const response=await fetch(`http://localhost:${port}/search?q=${encodeURIComponent(q)}`);
  if(!response.ok) throw new Error(`Omnisearch on port ${port} answered ${response.status}`);
  return response.json();
}
// Omnisearch answers before its index is complete, so wait until spread-out notes are all findable by name.
async function waitForLive(port,dataset,notes) {
  const probes=[0,0.25,0.5,0.75,1].map(f=>notes[Math.min(notes.length-1,Math.floor(f*notes.length))]);
  for(let attempt=0;;attempt++) {
    try {
      const found=await Promise.all(probes.map(async n=>(await liveSearch(port,n.basename)).some(r=>r.path===n.path)));
      if(found.every(Boolean)) return;
    } catch(error) { if(attempt===0) console.log(`Waiting for Omnisearch (${dataset} vault, port ${port}): ${error.cause?.code??error.message}`); }
    if(attempt>300) throw new Error(`Omnisearch on port ${port} did not finish indexing the ${dataset} vault.`);
    if(attempt%10===0) console.log(`Waiting for Omnisearch to index the ${dataset} vault...`);
    await sleep(2000);
  }
}

const live=process.argv.includes('--live');
const methods=['obsidian-core','omnisearch',...(live?['omnisearch-live']:[]),'omnisearch-any-word','bm25-title','dense-int8','hybrid-int8'];
const results={createdAt:new Date().toISOString(),embeddings:profile,omnisearch:'1.32.0 (e4a51ab), default settings, reproduced with minisearch 7.1.0',datasets:{}};
// Keep the last live check; report-benchmark.mjs only cites it while its reproduction score still matches.
results.live=JSON.parse(await readFile('benchmarks/results/omnisearch-comparison.json','utf8').catch(()=>'{}')).live??{};
for(const dataset of ['scifact','nfcorpus']) {
  const base=`benchmarks/data/${dataset}`;
  const corpus=(await readFile(`${base}/corpus.jsonl`,'utf8')).trim().split('\n').map(JSON.parse);
  const qrels={};
  for(const line of (await readFile(`${base}/qrels/test.tsv`,'utf8')).trim().split('\n').slice(1)) {const [q,id,g]=line.split('\t');(qrels[q]??={})[id]=Number(g);}
  const queries=(await readFile(`${base}/queries.jsonl`,'utf8')).trim().split('\n').map(JSON.parse).filter(q=>qrels[q._id]);
  const names=noteNames(corpus);
  const notes=corpus.map((d,i)=>({id:d._id,path:`${names[i]}.md`,basename:names[i],content:d.text,aliases:'',headings1:'',headings2:'',headings3:'',unmarkedTags:[]}));
  const idByPath=new Map(notes.map(n=>[n.path,n.id]));
  if(live) await waitForLive(LIVE_PORTS[dataset],dataset,notes);
  const agreement={identicalTop10:0,sameTop10Set:0,sameResultCount:0,mismatches:[]};
  const search={'obsidian-core':coreSearch(notes),omnisearch:omnisearchIndex(notes),'omnisearch-any-word':omnisearchIndex(notes,'OR')};
  const bm25=new BM25(corpus.map(d=>({id:d._id,text:d.text,title:d.title})),true);
  const texts=corpus.map(d=>`${d.title}\n${d.text}`.trim());
  const dense=new PackedVectors(packVectors(await cachedEmbeddings(`${dataset}-corpus`,texts),'int8'),corpus.map(d=>d._id));
  const queryVectors=await cachedEmbeddings(`${dataset}-queries`,queries.map(q=>q.text),true);
  const quality=Object.fromEntries(methods.map(m=>[m,[]])),returned=Object.fromEntries(methods.map(m=>[m,[]]));
  for(const [i,query] of queries.entries()) {
    const rankings={};
    for(const m of ['obsidian-core','omnisearch','omnisearch-any-word']) rankings[m]=search[m](query.text);
    if(live) {
      const raw=await liveSearch(LIVE_PORTS[dataset],query.text);
      const unknown=raw.filter(r=>!idByPath.has(r.path));
      if(unknown.length) throw new Error(`Omnisearch returned notes this corpus does not have, e.g. ${unknown[0].path}. Is port ${LIVE_PORTS[dataset]} the ${dataset} vault?`);
      rankings['omnisearch-live']=raw.map(r=>({id:idByPath.get(r.path),score:r.score}));
      const a=rankings.omnisearch.slice(0,10).map(r=>r.id),b=rankings['omnisearch-live'].slice(0,10).map(r=>r.id);
      const identical=a.length===b.length&&a.every((id,k)=>id===b[k]);
      if(identical) agreement.identicalTop10++;
      if(a.length===b.length&&a.every(id=>b.includes(id))) agreement.sameTop10Set++;
      if(rankings.omnisearch.length===rankings['omnisearch-live'].length) agreement.sameResultCount++;
      if(!identical&&agreement.mismatches.length<20) agreement.mismatches.push({query:query._id,text:query.text,
        reproduction:{results:rankings.omnisearch.length,top3:a.slice(0,3)},live:{results:raw.length,top3:b.slice(0,3)}});
    }
    rankings['bm25-title']=bm25.search(query.text,101);
    rankings['dense-int8']=dense.search(queryVectors[i],101);
    for(const m of Object.keys(rankings)) rankings[m]=rankings[m].filter(r=>r.id!==query._id).slice(0,100);
    rankings['hybrid-int8']=reciprocalRankFusion([rankings['bm25-title'],rankings['dense-int8']],100);
    for(const m of methods) {quality[m].push(evaluate(rankings[m].map(r=>r.id),qrels[query._id]));returned[m].push(rankings[m].length);}
  }
  const summary={corpus:corpus.length,queries:queries.length,meanQueryWords:average(queries.map(q=>q.text.split(/\s+/).length)),methods:{}};
  for(const m of methods) summary.methods[m]={
    ...Object.fromEntries(['ndcg10','recall10','mrr10'].map(k=>[k,average(quality[m].map(q=>q[k]))])),
    noResults:returned[m].filter(n=>n===0).length/queries.length,
    ndcgVsOmnisearchCI95:m==='omnisearch'?undefined:pairedBootstrap(quality[m].map((q,i)=>q.ndcg10-quality.omnisearch[i].ndcg10)),
  };
  if(live) {
    const {mismatches,...counts}=agreement;
    results.live[dataset]={checkedAt:new Date().toISOString(),queries:queries.length,...counts,
      ndcg10:{reproduction:summary.methods.omnisearch.ndcg10,live:summary.methods['omnisearch-live'].ndcg10},mismatches};
    console.log(`Reproduction vs live Omnisearch on ${dataset}: identical top 10 for ${counts.identicalTop10}/${queries.length} queries, same top-10 set ${counts.sameTop10Set}, same result count ${counts.sameResultCount}.`);
  }
  results.datasets[dataset]=summary;
  console.log(`\n${dataset}: ${corpus.length} notes, ${queries.length} queries, ${summary.meanQueryWords.toFixed(1)} words per query`);
  console.table(Object.fromEntries(methods.map(m=>{const s=summary.methods[m];return [m,{'nDCG@10':s.ndcg10.toFixed(4),'Recall@10':s.recall10.toFixed(4),'MRR@10':s.mrr10.toFixed(4),'no results':`${(s.noResults*100).toFixed(1)}%`,'Δ nDCG vs Omnisearch (95% CI)':s.ndcgVsOmnisearchCI95?s.ndcgVsOmnisearchCI95.map(v=>v.toFixed(3)).join(' to '):''}];})));
}
await mkdir('benchmarks/results',{recursive:true});
await writeFile('benchmarks/results/omnisearch-comparison.json',JSON.stringify(results,null,2)+'\n');
console.log('Wrote benchmarks/results/omnisearch-comparison.json');
