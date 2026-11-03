# Retrieval benchmark

This suite compares BM25, BGE-small CPU embeddings, 8-bit/4-bit vector compression, and reciprocal-rank fusion on two public [BEIR](https://github.com/beir-cellar/beir) collections:

| Dataset | Full corpus | Test queries |
| --- | ---: | ---: |
| SciFact | 5,183 | 300 |
| NFCorpus | 3,633 | 323 |

The full document corpus is searched for every official test query. These are not sampled-corpus results. Download URLs and checksums are recorded in `sources.json`; the model revision, graph, tokenizer hashes, hardware, runtime, and settings are recorded in `results/summary.json`.

## Reproduce

```sh
npm ci
python -m pip install -r benchmarks/requirements.txt
python scripts/prepare-benchmarks.py
node scripts/embed-benchmarks-wasm.mjs
npm run benchmark
node scripts/report-benchmark.mjs
```

The preparation script downloads only public benchmark data and a pinned model into this workspace. To change the pinned model revision deliberately, rerun preparation after reviewing the new revision. The plugin never performs this update automatically. Generated embeddings are keyed by a hash of model revision, query/document role, and exact input text. Delete `benchmarks/cache` to measure cold encoding again.

## Method

- BM25: fixed Unicode word analyzer, English stopword removal, Porter stemming, `k1=1.2`, `b=0.75`; title and body have equal weight for the canonical baseline. This is this repository's reproducible BM25 implementation, not a claim to have reproduced a specific Elasticsearch leaderboard score.
- Embedding: the pinned quantized `Xenova/bge-small-en-v1.5` ONNX graph, CLS pooling, L2 normalization, 384 dimensions, 512-token truncation retaining the closing SEP token, and BGE's recommended query instruction. Documents use title plus text.
- Final quality scores use the exact shipping SIMD WASM engine for **every** document and query. Four independent one-thread workers accelerate benchmark preparation. Separately measured plugin latency uses one worker on 12 evenly spaced documents and 32 queries per dataset. The earlier native ONNX reference run failed a strict 0.999 minimum-cosine agreement check, so those native embeddings are not used for the final quality scores. The native scripts remain as reproducible diagnostics.
- Exact search: no approximate nearest-neighbor index or candidate filtering. The same corpus is used for BM25, dense search, and compression comparisons.
- Hybrid: RRF with `k=60`, top 100 candidates from each method. No cross-encoder reranker.
- Metrics: graded nDCG@10 with linear relevance gains, matching BEIR/trec_eval `ndcg_cut`; Recall@10 and MRR@10. Unjudged documents are non-relevant and identical query/document IDs are excluded. Fixed-seed paired bootstrap confidence intervals compare per-query nDCG with BM25. Search timings exclude query encoding; WASM query encoding latency is reported separately. One startup model load is separate again. The [trec_eval implementation](https://github.com/usnistgov/trec_eval/blob/master/m_ndcg_cut.c) defines the gain convention.
- Compression: compare actual binary byte length (including headers/scales), nDCG changes, and top-10 overlap with float32. Note and heading text metadata is not included in vector-only savings.

The policy is fixed before inspecting results: recommend vector or hybrid only with at least **0.01 absolute macro nDCG@10 improvement**, **no dataset regression**, and a **positive lower paired-bootstrap bound on each dataset**. Otherwise keep heading-aware BM25. Prefer smaller vector storage only with no more than **0.005 nDCG loss on either dataset** and **at least 95% top-10 overlap** with float32.

## Limits

SciFact and NFCorpus are English scientific/medical retrieval tasks, not an Obsidian usage study. Canonical scores retrieve whole documents and do not invent headings for these datasets. They test model choice and compression; unit/integration tests separately verify real Markdown section boundaries, heading weights, source links, edits, moves, deletions, and vault isolation. Do not treat these two tasks as proof that a model wins on all personal vaults or languages. BGE is a pretrained model; this is not a claim of decontaminated training data.

The rotated scalar experiment is inspired by TurboQuant but is not the paper's full algorithm. It uses a fixed randomized Hadamard transform padded to 512 dimensions, a fixed 16-value Gaussian codebook, and norm correction. It has neither learned per-corpus codebooks nor QJL residual correction. See the [TurboQuant paper](https://arxiv.org/abs/2504.19874) for the full method; published GPU/KV-cache speedups do not transfer automatically to this CPU plugin.

Results and a readable report are generated after the full run. A reranker is intentionally absent until it shows a worthwhile CPU latency/quality tradeoff.
