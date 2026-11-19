import { readFile, writeFile } from 'node:fs/promises';
const summary=JSON.parse(await readFile('benchmarks/results/summary.json','utf8'));
const optional=async file=>{try{return JSON.parse(await readFile(file,'utf8'));}catch{return null;}};
// Same corpus, queries, and metrics with the shipped model's other weights, from scripts/embed-benchmarks-native.mjs.
const shipped=JSON.parse(await readFile('src/model-spec.json','utf8'));
const weightRuns=Object.fromEntries((await Promise.all(['fp32','fp16'].map(async w=>[w,await optional(`benchmarks/results/summary-${shipped.key}-${w}.json`)]))).filter(([,run])=>run));
const engines=await optional('benchmarks/results/engines.json');
// bge-small-en-v1.5, the model earlier versions shipped, measured the same way.
const previous=await optional('benchmarks/results/summary-bge-small-int8.json'),previousEngines=await optional('benchmarks/results/engines-bge-small.json');
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
report+='\nAll dense rows above use the int8 (quantized) ONNX model weights. The f32/int8/sq4 labels describe stored vector precision, not different model weights.\n';
if(Object.keys(weightRuns).length) {
  report+='\n## Model weight precision\n\nThe plugin pairs int8 weights with the CPU and full-precision (fp32) weights with DirectML, so each precision was scored on the same full test sets. Rows use int8 vector storage, as shipped.\n\n| Weights | SciFact dense | NFCorpus dense | SciFact hybrid | NFCorpus hybrid | Macro hybrid |\n| --- | ---: | ---: | ---: | ---: | ---: |\n';
  for(const [label,run] of [['int8',summary],...Object.entries(weightRuns)]) {
    const m=(set,mode)=>run.datasets[set].modes[mode].ndcg10;
    report+=`| ${label} | ${m('scifact','int8').toFixed(4)} | ${m('nfcorpus','int8').toFixed(4)} | ${m('scifact','hybrid-int8').toFixed(4)} | ${m('nfcorpus','hybrid-int8').toFixed(4)} | ${((m('scifact','hybrid-int8')+m('nfcorpus','hybrid-int8'))/2).toFixed(4)} |\n`;
  }
  report+='\nDifferences between weight precisions are within ±0.005 nDCG@10, far inside the paired confidence intervals against BM25, so precision is chosen for speed and determinism rather than quality. int8 weights quantize activations with one scale per input tensor, so batching texts together changes each vector (cosine 0.996 against one-at-a-time); the plugin runs int8 one text at a time, which is also faster on the CPU. fp32 vectors match between DirectML and the native CPU (cosine ≥ 0.99999), so documents can be embedded on the GPU and queries on the CPU.\n';
}
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
if(engines) {
  report+=`\n## Embedding engines\n\n${engines.sample}. Hardware: ${engines.hardware.cpu}, ${engines.hardware.logicalCPUs} logical CPUs, ${engines.hardware.platform}. Runtime: ${engines.runtime}. Measured through the plugin's worker bundle with \`node scripts/benchmark-engines.mjs\`.\n\n| Engine | Weights | Indexing (ms/document) | Query p50 / p95 (ms) | Startup (ms) |\n| --- | --- | ---: | ---: | ---: |\n`;
  for(const e of Object.values(engines.engines))report+=`| ${e.label} | ${e.weights} | ${e.msPerDocument.toFixed(1)} | ${e.queryP50Ms.toFixed(1)} / ${e.queryP95Ms.toFixed(1)} | ${e.loadMs.toFixed(0)} |\n`;
  if(previousEngines) {
    const compare=key=>engines.engines[key] && previousEngines.engines[key] ? `${previousEngines.engines[key].msPerDocument.toFixed(1)} ms/document (this model takes ${(engines.engines[key].msPerDocument/previousEngines.engines[key].msPerDocument).toFixed(2)}x as long)` : 'n/a';
    report+=`\nThe same sample with bge-small-en-v1.5, the previous model, measured back to back: CPU int8 ${compare('native-cpu-int8')}; DirectML fp32 ${compare('directml-fp32')}.\n`;
  }
  report+='\nSetup times the CPU (int8) against DirectML (fp32) once per computer on twelve sample sections of typical lengths and keeps the GPU unless it measures more than twice as slow. Windows runs a background Obsidian window at idle CPU priority, which slows CPU embedding about 3x (358 → 1040 ms on the calibration sample) while DirectML barely changes (590 → 620 ms); the factor of two assumes half of bulk indexing happens in the background. On this Iris Xe laptop the CPU is faster in the foreground, but the GPU is chosen. DirectML runs single short inputs slowly, so in GPU mode queries run on the CPU with the same fp32 weights. These Iris Xe timings predate the DirectML GPU graph (next section) and have not been measured again.\n';
}
const gpu=await optional('benchmarks/results/gpu-rtx3070.json');
if(gpu) {
  const run=async file=>(await optional(`benchmarks/results/${file}`))?.engines;
  const [patched,upstream]=await Promise.all([run(gpu.engines.gpuGraph),run(gpu.engines.upstreamGraph)]);
  const ms=(engines,key)=>engines?.[key] ? engines[key].msPerDocument.toFixed(1) : 'n/a';
  const pair=([a,b])=>`${a.toFixed(1)} / ${b.toFixed(1)}`;
  report+=`\n## DirectML on a discrete GPU\n\nHardware: ${gpu.hardware.gpu}; ${gpu.hardware.cpu}. Runtime: ${gpu.runtime}. ${gpu.gelu.finding} The DirectML graph bundled with the plugin fixes this: ${gpu.gelu.fix} ${gpu.gelu.agreement} The CPU keeps the upstream graph, and both load the same downloaded weights.\n\n`;
  report+='Setup also chooses DirectML\'s batch size once the GPU wins: 16 sections per call in batches of up to 8 (tuned on Iris Xe), or 64 per call in batches of up to 64. Large batches must measure at least 1.2x faster on 64 probe sections skewed towards short ones, as vaults are. Twelve evenly spread lengths, as in the CPU/GPU comparison, favour small batches because a single large batch pads them all to 510 tokens. Closing batches early to limit padding was slower at every threshold tried: on this GPU the cost is per call.\n\n';
  report+='| DirectML graph | Small batches (ms/document) | Large batches (ms/document) |\n| --- | ---: | ---: |\n';
  report+=`| Upstream (Gelu on the CPU) | ${ms(upstream,'directml-fp32')} | ${ms(upstream,'directml-fp32-large')} |\n| Bundled GPU graph | ${ms(patched,'directml-fp32')} | ${ms(patched,'directml-fp32-large')} |\n\n`;
  report+=`Measured with \`node scripts/benchmark-engines.mjs\` on the sample above. ${gpu.throttling} Setup's calibration in the same session: CPU ${gpu.calibration.cpuMs.toFixed(0)} ms against GPU ${gpu.calibration.gpuMs.toFixed(0)} ms, then small batches ${gpu.calibration.batch.standardMs.toFixed(0)} ms against large ${gpu.calibration.batch.largeMs.toFixed(0)} ms, choosing the GPU with large batches.\n\n`;
  report+=`**CUDA.** ${gpu.cuda.note} With the bundled graph and large batches, DirectML comes within 1.1–1.4x of CUDA, so the plugin does not ship a CUDA engine (a separate ONNX Runtime build plus over 1 GB of NVIDIA libraries).\n\n| Model | DirectML, upstream graph | DirectML, GPU graph | CUDA |\n| --- | ---: | ---: | ---: |\n`;
  for(const row of gpu.cuda.rows)report+=`| ${row.model} | ${pair(row.directmlUpstream)} | ${pair(row.directmlGpuGraph)} | ${pair(row.cuda)} |\n`;
  report+=`\nCells are ms/document with small / large batches.\n\n`;
  const ci=s=>`${s.gain>=0?'+':''}${s.gain.toFixed(4)} [${s.ci95.map(x=>(x>=0?'+':'')+x.toFixed(3)).join(', ')}]`,v=gpu.largerModel.vsShipped;
  report+=`**A larger model for strong GPUs.** ${gpu.largerModel.model}: ${gpu.largerModel.note} SciFact dense ${ci(v['scifact/int8'])}, hybrid ${ci(v['scifact/hybrid-int8'])}; NFCorpus dense ${ci(v['nfcorpus/int8'])}, hybrid ${ci(v['nfcorpus/hybrid-int8'])}. Every interval includes zero, while the model downloads 605 MB instead of 195 MB, stores vectors twice as large, has no int8 weights for the CPU, and indexes about 2.4x slower even with the GPU graph. No larger-model tier is offered.\n`;
}
const wasmLatency=previous && Object.values(previous.datasets).every(d=>d.wasmRuntime);
if(wasmLatency) report+='\n## WebAssembly engine latency (removed)\n\nEarlier releases fell back to a one-thread WebAssembly engine when the native engine could not load. It was removed: it ran about 10x slower than the native engine and could not run the newer ModernBERT models. Measured with bge-small int8, whose quality scores were produced with this engine. Sample sizes: 32 evenly spaced official queries and 12 documents per dataset. These latency samples are not full-corpus inference throughput. Warm vector scans include all corpus documents.\n\n| Dataset | WASM query p50/p95 (ms) | WASM document p50/p95 (ms) | int8 scan p50/p95 (ms) | BM25 p50/p95 (ms) |\n| --- | ---: | ---: | ---: | ---: |\n';
if(wasmLatency) for(const [name,d] of Object.entries(previous.datasets))report+=`| ${name} | ${d.wasmRuntime.queryP50Ms.toFixed(1)} / ${d.wasmRuntime.queryP95Ms.toFixed(1)} | ${d.wasmRuntime.documentP50Ms.toFixed(1)} / ${d.wasmRuntime.documentP95Ms.toFixed(1)} | ${d.modes.int8.searchP50Ms.toFixed(1)} / ${d.modes.int8.searchP95Ms.toFixed(1)} | ${d.modes.bm25.searchP50Ms.toFixed(1)} / ${d.modes.bm25.searchP95Ms.toFixed(1)} |\n`;
report+=`\n${summary.modelLoadMs ? `Cold worker/model startup: ${summary.modelLoadMs.toFixed(0)} ms. ` : ''}Timings depend on this machine and background load. First-time vault indexing can take minutes, especially for long documents. BM25 remains available during embedding.\n\n`;
const models=await optional('benchmarks/results/models.json');
if(models) {
  report+='## Candidate embedding models\n\nEach model embedded the same full test sets with fp32 weights through the plugin\'s native engine (`scripts/embed-benchmarks-native.mjs --model`), scored with `scripts/benchmark.mjs`, and compared by `scripts/compare-models.mjs`. Intervals are paired bootstrap 95% CIs of the per-query nDCG@10 difference against bge-small-en-v1.5 (the previous default) at the same precision. Dense rows use int8 vector storage.\n\n| Model | Parameters | Dimensions | SciFact dense | SciFact hybrid | NFCorpus dense | NFCorpus hybrid | GPU encode (ms/document) |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n';
  const cell=s=>`${s.ndcg10.toFixed(4)}${s.vsBaselineCI95?` [${s.vsBaselineCI95.map(x=>(x>=0?'+':'')+x.toFixed(3)).join(', ')}]`:''}`;
  for(const m of models.models)report+=`| ${m.label} | ${m.parameters} | ${m.dimensions} | ${cell(m.scores['scifact/int8'])} | ${cell(m.scores['scifact/hybrid-int8'])} | ${cell(m.scores['nfcorpus/int8'])} | ${cell(m.scores['nfcorpus/hybrid-int8'])} | ${((m.encodeMsPerDocument.scifact+m.encodeMsPerDocument.nfcorpus)/2).toFixed(0)} |\n`;
  report+='\nGranite small English r2 improves on bge-small-en-v1.5 on both datasets for dense and hybrid search, with every interval above zero, and is now the shipped model. Its int8 weights score within 0.005 nDCG@10 of fp32 (see model weight precision) and index somewhat slower than bge-small on the CPU (see embedding engines). Granite English r2 is slightly better again but 2.5x slower to encode. The multilingual model is worse on these English tasks. IBM describes its training data only as permissively licensed public and IBM-generated pairs; training-split exposure to SciFact or NFCorpus cannot be ruled out.\n\n';
}
const rerank=await optional('benchmarks/results/rerank.json');
if(rerank) {
  const entries=Object.values(rerank).sort((a,b)=>parseFloat(a.reranker.parameters)-parseFloat(b.reranker.parameters));
  report+=`## Rerankers\n\nCross-encoders rerank the top 20 first-stage results for every test query (\`scripts/benchmark-rerank.mjs\`). First stage: ${entries[0].firstStage}. Intervals are paired bootstrap 95% CIs of the nDCG@10 change against the first stage. Latency is the median time to score 20 pairs on ${entries[0].engine} (Intel Iris Xe).\n\n| Reranker | Parameters | SciFact first → rerank@20 | NFCorpus first → rerank@20 | Latency per query |\n| --- | ---: | ---: | ---: | ---: |\n`;
  const cell=d=>`${d.modes.first.ndcg10.toFixed(4)} → ${d.modes['rerank@20'].ndcg10.toFixed(4)} [${d.modes['rerank@20'].gainOverFirstCI95.map(x=>(x>=0?'+':'')+x.toFixed(3)).join(', ')}]`;
  // Runs at K=50 time 50 pairs per query; scale to the 20-pair budget reported here.
  const latency=e=>{const l=[e.datasets.scifact.latencyMs,e.datasets.nfcorpus.latencyMs].filter(Boolean).map(x=>x.p50*20/e.depth);return l.length?`${(l.reduce((a,b)=>a+b)/l.length/1000).toFixed(1)} s`:'n/a';};
  for(const e of entries)report+=`| ${e.reranker.id.split('/')[1]} | ${e.reranker.parameters} | ${cell(e.datasets.scifact)} | ${cell(e.datasets.nfcorpus)} | ${latency(e)} |\n`;
  report+='\nNo reranker improved both datasets: they help slightly on SciFact and hurt on NFCorpus, significantly so for the 17M and 32M models. With the GPU already slower than interactive search allows at 68M parameters, no reranker is shipped. A 150M reference reranker was dropped from the run for time.\n\n';
}
report+='## Decision\n\n';
report+=`Recommended retrieval: **${summary.recommendation.defaultMode}**. Recommended vector codec: **${summary.recommendation.codec}**.\n\n${summary.recommendation.reason}\n\n`;
if(summary.recommendation.defaultMode==='bm25')report+='Dense-only gains have confidence intervals spanning zero, so the pure-vector gate did not pass. Optional hybrid search scored highest and had positive paired intervals against both BM25 baselines on both tasks. BM25 remains the initial recommendation under the requested conservative pivot rule, with no model download or embedding latency.\n\n';
report+='| Candidate | SciFact nDCG gain 95% paired CI | NFCorpus nDCG gain 95% paired CI |\n| --- | ---: | ---: |\n';
for(const mode of ['int8','hybrid-int8'])for(const [label,key] of [['plain BM25','ndcgGainOverBM25CI95'],['title-weighted BM25','ndcgGainOverTitleBM25CI95']])
  report+=`| ${mode} vs ${label} | ${summary.datasets.scifact.modes[mode][key].map(x=>x.toFixed(4)).join(' to ')} | ${summary.datasets.nfcorpus.modes[mode][key].map(x=>x.toFixed(4)).join(' to ')} |\n`;
report+=`\nCompression acceptance requires no more than ${summary.policy.maxQuantizationNdcgLoss} absolute nDCG loss on either dataset, at least 95% top-10 overlap, and scan p95 no slower than ${summary.policy.maxQuantizationScanSlowdown}x int8.\n`;
report+='\nThe plugin initially opens with BM25 because it needs no model. This recommendation applies after optional model setup. Two English scientific datasets do not establish a universal winner for personal notes, multilingual vaults, or heading weights. The rotated-SQ4 experiment is not full TurboQuant. Rerankers were benchmarked (above) and are not shipped. See [methodology](README.md).\n';
await writeFile('benchmarks/RESULTS.md',report);
await writeFile('src/retrieval-policy.json',JSON.stringify({recommendedMode:summary.recommendation.defaultMode==='hybrid-int8'?'hybrid':summary.recommendation.defaultMode==='int8'?'vector':'bm25',recommendedCodec:summary.recommendation.codec,benchmarkDate:summary.createdAt,scope:'SciFact and NFCorpus full test splits only'},null,2)+'\n');
console.log('Generated benchmarks/RESULTS.md and src/retrieval-policy.json');
