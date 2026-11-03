import { readFile, writeFile } from 'node:fs/promises';
const summary=JSON.parse(await readFile('benchmarks/results/summary.json','utf8'));
if(!summary.recommendation || Object.keys(summary.datasets).length!==2)throw new Error('The full benchmark has not completed.');
const percent=x=>(x*100).toFixed(2);
const modes=['bm25','bm25-title','f32','int8','sq4','rotated-sq4','hybrid-int8'];
let report=`# Measured retrieval results\n\nRun: ${new Date(summary.createdAt).toLocaleString('en-AU',{timeZone:'Australia/Sydney'})} Australia/Sydney. CPU: ${summary.hardware.cpu}; ${summary.hardware.logicalCPUs} logical CPUs. Model: ${summary.model.id} at revision \`${summary.model.revision}\`.\n\n${summary.runtime}\n\n`;
report+='## Retrieval quality\n\nnDCG@10 uses BEIR/trec_eval linear graded gains. Higher is better. All test queries search the full corpus.\n\n| Method | SciFact nDCG@10 | NFCorpus nDCG@10 | Macro average |\n| --- | ---: | ---: | ---: |\n';
for(const mode of modes) {
  const a=summary.datasets.scifact.modes[mode].ndcg10,b=summary.datasets.nfcorpus.modes[mode].ndcg10;
  report+=`| ${mode} | ${a.toFixed(4)} | ${b.toFixed(4)} | ${((a+b)/2).toFixed(4)} |\n`;
}
report+='\n`bm25-title` uses the shipped title weight (2x) with no synthetic section headings. These datasets contain titles and abstracts, not Markdown heading hierarchies. Both lexical baselines must be beaten before recommending a vector-based default.\n';
report+='\n| Method | SciFact Recall@10 | NFCorpus Recall@10 | SciFact MRR@10 | NFCorpus MRR@10 |\n| --- | ---: | ---: | ---: | ---: |\n';
for(const mode of modes) {
  const a=summary.datasets.scifact.modes[mode],b=summary.datasets.nfcorpus.modes[mode];
  report+=`| ${mode} | ${a.recall10.toFixed(4)} | ${b.recall10.toFixed(4)} | ${a.mrr10.toFixed(4)} | ${b.mrr10.toFixed(4)} |\n`;
}
report+='\n## Compression\n\nThe binary sizes include per-vector scale/norm values and one 16-byte header. Text metadata, BM25 postings, and model weights are separate. Top-10 overlap measures agreement with float32, not relevance.\n\n| Codec | Bytes/vector | Space saved | SciFact top-10 overlap | NFCorpus top-10 overlap |\n| --- | ---: | ---: | ---: | ---: |\n';
for(const mode of ['f32','int8','sq4','rotated-sq4']) {
  const a=summary.datasets.scifact,b=summary.datasets.nfcorpus;
  report+=`| ${mode} | ${a.storage[mode].bytesPerVector} | ${percent(1-a.storage[mode].bytes/a.storage.f32.bytes)}% | ${percent(a.modes[mode].top10OverlapWithFloat)}% | ${percent(b.modes[mode].top10OverlapWithFloat)}% |\n`;
}
report+='\n## CPU latency\n\nShipping runtime: one-thread SIMD WASM. Sample sizes: 32 evenly spaced official queries and 12 documents per dataset. These latency samples are not full-corpus inference throughput. Warm vector scans include all corpus documents. All final quality embeddings also use this shipping engine; four independent workers accelerate corpus preparation only. Native CPU embeddings were excluded after a strict runtime agreement check failed.\n\n| Dataset | WASM query p50/p95 (ms) | WASM document p50/p95 (ms) | int8 scan p50/p95 (ms) | BM25 p50/p95 (ms) |\n| --- | ---: | ---: | ---: | ---: |\n';
for(const [name,d] of Object.entries(summary.datasets))report+=`| ${name} | ${d.wasmRuntime.queryP50Ms.toFixed(1)} / ${d.wasmRuntime.queryP95Ms.toFixed(1)} | ${d.wasmRuntime.documentP50Ms.toFixed(1)} / ${d.wasmRuntime.documentP95Ms.toFixed(1)} | ${d.modes.int8.searchP50Ms.toFixed(1)} / ${d.modes.int8.searchP95Ms.toFixed(1)} | ${d.modes.bm25.searchP50Ms.toFixed(1)} / ${d.modes.bm25.searchP95Ms.toFixed(1)} |\n`;
report+=`\nCold worker/model startup: ${summary.modelLoadMs.toFixed(0)} ms. Timings depend on this machine and background load; no GPU was used. First-time vault indexing can take minutes, especially for long documents. BM25 remains available during embedding.\n\n`;
report+='## Decision\n\n';
report+=`Recommended retrieval: **${summary.recommendation.defaultMode}**. Recommended vector codec: **${summary.recommendation.codec}**.\n\n${summary.recommendation.reason}\n\n`;
report+='| Candidate | SciFact nDCG gain 95% paired CI | NFCorpus nDCG gain 95% paired CI |\n| --- | ---: | ---: |\n';
for(const mode of ['int8','hybrid-int8'])report+=`| ${mode} | ${summary.datasets.scifact.modes[mode].ndcgGainOverBM25CI95.map(x=>x.toFixed(4)).join(' to ')} | ${summary.datasets.nfcorpus.modes[mode].ndcgGainOverBM25CI95.map(x=>x.toFixed(4)).join(' to ')} |\n`;
report+='\nThe plugin initially opens with BM25 because it needs no model. This recommendation applies after optional model setup. Two English scientific datasets do not establish a universal winner for personal notes, multilingual vaults, or heading weights. The rotated-SQ4 experiment is not full TurboQuant. A reranker was not benchmarked or shipped. See [methodology](README.md).\n';
await writeFile('benchmarks/RESULTS.md',report);
await writeFile('src/retrieval-policy.json',JSON.stringify({recommendedMode:summary.recommendation.defaultMode==='hybrid-int8'?'hybrid':summary.recommendation.defaultMode==='int8'?'vector':'bm25',recommendedCodec:summary.recommendation.codec,benchmarkDate:summary.createdAt,scope:'SciFact and NFCorpus full test splits only'},null,2)+'\n');
console.log('Generated benchmarks/RESULTS.md and src/retrieval-policy.json');
