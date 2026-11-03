# Measured retrieval results

Run: 03/11/2026, 4:29:33 pm Australia/Sydney. CPU: 11th Gen Intel(R) Core(TM) i7-11390H @ 3.40GHz; 8 logical CPUs. Model: Xenova/bge-small-en-v1.5 at revision `ea104dacec62c0de699686887e3f920caeb4f3e3`.

Full quality and sampled latency: exact shipping onnxruntime-web 1.17.3 SIMD WASM engine, one CPU thread per inference, no GPU. Four independent workers accelerate corpus preparation only; latency samples use one worker.

## Retrieval quality

nDCG@10 uses BEIR/trec_eval linear graded gains. Higher is better. All test queries search the full corpus.

| Method | SciFact nDCG@10 | NFCorpus nDCG@10 | Macro average |
| --- | ---: | ---: | ---: |
| bm25 | 0.6894 | 0.3290 | 0.5092 |
| bm25-title | 0.6901 | 0.3304 | 0.5102 |
| f32 | 0.7104 | 0.3448 | 0.5276 |
| int8 | 0.7079 | 0.3446 | 0.5262 |
| sq4 | 0.6829 | 0.3314 | 0.5071 |
| rotated-sq4 | 0.7032 | 0.3419 | 0.5225 |
| hybrid-int8 | 0.7216 | 0.3666 | 0.5441 |

`bm25-title` uses the shipped title weight (2x) with no synthetic section headings. These datasets contain titles and abstracts, not Markdown heading hierarchies. Both lexical baselines must be beaten before recommending a vector-based default.

All dense rows use the same quantized ONNX model weights. The f32/int8/sq4 labels describe stored vector precision, not different model weights.

| Method | SciFact Recall@10 | NFCorpus Recall@10 | SciFact MRR@10 | NFCorpus MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.8302 | 0.1563 | 0.6506 | 0.5339 |
| bm25-title | 0.8206 | 0.1569 | 0.6544 | 0.5358 |
| f32 | 0.8412 | 0.1621 | 0.6768 | 0.5345 |
| int8 | 0.8379 | 0.1619 | 0.6743 | 0.5355 |
| sq4 | 0.8096 | 0.1576 | 0.6509 | 0.5103 |
| rotated-sq4 | 0.8446 | 0.1620 | 0.6654 | 0.5317 |
| hybrid-int8 | 0.8579 | 0.1734 | 0.6848 | 0.5756 |

## Compression

The binary sizes include per-vector scale/norm values and one 16-byte header. Text metadata, BM25 postings, and model weights are separate. Top-10 overlap measures agreement with float32, not relevance.

| Codec | Bytes/vector | Space saved | SciFact top-10 overlap | NFCorpus top-10 overlap |
| --- | ---: | ---: | ---: | ---: |
| f32 | 1536 | 0.00% | 100.00% | 100.00% |
| int8 | 392 | 74.48% | 98.43% | 97.74% |
| sq4 | 200 | 86.98% | 74.90% | 71.55% |
| rotated-sq4 | 260 | 83.07% | 92.37% | 90.68% |

## CPU latency

Shipping runtime: one-thread SIMD WASM. Sample sizes: 32 evenly spaced official queries and 12 documents per dataset. These latency samples are not full-corpus inference throughput. Warm vector scans include all corpus documents. All final quality embeddings also use this shipping engine; four independent workers accelerate corpus preparation only. Native CPU embeddings were excluded after a strict runtime agreement check failed.

| Dataset | WASM query p50/p95 (ms) | WASM document p50/p95 (ms) | int8 scan p50/p95 (ms) | BM25 p50/p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| scifact | 43.9 / 82.6 | 922.1 / 1138.0 | 7.3 / 11.9 | 1.5 / 3.4 |
| nfcorpus | 32.4 / 45.2 | 818.6 / 1137.0 | 5.0 / 8.3 | 0.2 / 1.4 |

Cold worker/model startup: 1052 ms. Timings depend on this machine and background load; no GPU was used. First-time vault indexing can take minutes, especially for long documents. BM25 remains available during embedding.

## Decision

Recommended retrieval: **bm25**. Recommended vector codec: **int8**.

Dense vectors must first beat both plain and title-weighted BM25 with >= 0.01 absolute macro nDCG@10 gain over the stronger baseline, no dataset regression, and positive paired bootstrap lower bounds on each dataset. A hybrid default must independently clear the same gate. Otherwise default to heading-aware BM25. Quantization is evaluated separately from model quality.

Dense-only gains have confidence intervals spanning zero, so the pure-vector gate did not pass. Optional hybrid search scored highest and had positive paired intervals against both BM25 baselines on both tasks. BM25 remains the initial recommendation under the requested conservative pivot rule, with no model download or embedding latency.

| Candidate | SciFact nDCG gain 95% paired CI | NFCorpus nDCG gain 95% paired CI |
| --- | ---: | ---: |
| int8 vs plain BM25 | -0.0137 to 0.0492 | -0.0032 to 0.0352 |
| int8 vs title-weighted BM25 | -0.0152 to 0.0492 | -0.0045 to 0.0341 |
| hybrid-int8 vs plain BM25 | 0.0124 to 0.0534 | 0.0243 to 0.0510 |
| hybrid-int8 vs title-weighted BM25 | 0.0103 to 0.0543 | 0.0232 to 0.0498 |

Compression acceptance requires no more than 0.005 absolute nDCG loss on either dataset, at least 95% top-10 overlap, and scan p95 no slower than 1.5x int8.

The plugin initially opens with BM25 because it needs no model. This recommendation applies after optional model setup. Two English scientific datasets do not establish a universal winner for personal notes, multilingual vaults, or heading weights. The rotated-SQ4 experiment is not full TurboQuant. A reranker was not benchmarked or shipped. See [methodology](README.md).
