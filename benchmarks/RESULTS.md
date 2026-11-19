# Measured retrieval results

Run: 19/11/2026, 7:59:55 pm Australia/Sydney. CPU: 11th Gen Intel(R) Core(TM) i7-11390H @ 3.40GHz; 8 logical CPUs. Model: onnx-community/granite-embedding-small-english-r2-ONNX at revision `1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8`.

Full quality: the shipping native onnxruntime-node engine with int8 weights, batched exactly as the plugin indexes. See each dataset's documentEmbedding.backend for the device.

## Retrieval quality

nDCG@10 uses BEIR/trec_eval linear graded gains. Higher is better. All test queries search the full corpus.

| Method | SciFact nDCG@10 | NFCorpus nDCG@10 | Macro average |
| --- | ---: | ---: | ---: |
| bm25 | 0.6894 | 0.3290 | 0.5092 |
| bm25-title | 0.6901 | 0.3304 | 0.5102 |
| f32 | 0.7435 | 0.3676 | 0.5555 |
| int8 | 0.7413 | 0.3682 | 0.5547 |
| sq4 | 0.6853 | 0.3172 | 0.5013 |
| rotated-sq4 | 0.7389 | 0.3683 | 0.5536 |
| hybrid-int8 | 0.7460 | 0.3733 | 0.5597 |

`bm25-title` uses the shipped title weight (2x) with no synthetic section headings. These datasets contain titles and abstracts, not Markdown heading hierarchies. Both lexical baselines must be beaten before recommending a vector-based default.

All dense rows above use the int8 (quantized) ONNX model weights. The f32/int8/sq4 labels describe stored vector precision, not different model weights.

## Model weight precision

The plugin pairs int8 weights with the CPU and full-precision (fp32) weights with DirectML, so each precision was scored on the same full test sets. Rows use int8 vector storage, as shipped.

| Weights | SciFact dense | NFCorpus dense | SciFact hybrid | NFCorpus hybrid | Macro hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| int8 | 0.7413 | 0.3682 | 0.7460 | 0.3733 | 0.5597 |
| fp32 | 0.7463 | 0.3683 | 0.7451 | 0.3763 | 0.5607 |

Differences between weight precisions are within ±0.005 nDCG@10, far inside the paired confidence intervals against BM25, so precision is chosen for speed and determinism rather than quality. int8 weights quantize activations with one scale per input tensor, so batching texts together changes each vector (cosine 0.996 against one-at-a-time); the plugin runs int8 one text at a time, which is also faster on the CPU. fp32 vectors match between DirectML and the native CPU (cosine ≥ 0.99999), so documents can be embedded on the GPU and queries on the CPU.

| Method | SciFact Recall@10 | NFCorpus Recall@10 | SciFact MRR@10 | NFCorpus MRR@10 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 0.8302 | 0.1563 | 0.6506 | 0.5339 |
| bm25-title | 0.8206 | 0.1569 | 0.6544 | 0.5358 |
| f32 | 0.8651 | 0.1768 | 0.7115 | 0.5606 |
| int8 | 0.8651 | 0.1755 | 0.7081 | 0.5608 |
| sq4 | 0.8153 | 0.1528 | 0.6565 | 0.5235 |
| rotated-sq4 | 0.8617 | 0.1775 | 0.7061 | 0.5595 |
| hybrid-int8 | 0.8816 | 0.1796 | 0.7136 | 0.5830 |

## Compression

The binary sizes include per-vector scale/norm values and one 16-byte header. Text metadata, BM25 postings, and model weights are separate. Top-10 overlap measures agreement with float32, not relevance.

| Codec | Bytes/vector | Space saved | SciFact top-10 overlap | NFCorpus top-10 overlap |
| --- | ---: | ---: | ---: | ---: |
| f32 | 1536 | 0.00% | 100.00% | 100.00% |
| int8 | 392 | 74.48% | 97.40% | 96.41% |
| sq4 | 200 | 86.98% | 58.57% | 52.66% |
| rotated-sq4 | 260 | 83.07% | 93.47% | 91.58% |

## Embedding engines

128 evenly spaced SciFact/NFCorpus documents (title + text), sent 16 at a time as the plugin indexes; 32 queries one at a time. Hardware: 11th Gen Intel(R) Core(TM) i7-11390H @ 3.40GHz, 8 logical CPUs, win32-x64. Runtime: onnxruntime-node 1.22.0. Measured through the plugin's worker bundle with `node scripts/benchmark-engines.mjs`.

| Engine | Weights | Indexing (ms/document) | Query p50 / p95 (ms) | Startup (ms) |
| --- | --- | ---: | ---: | ---: |
| Native CPU (4 threads) | int8 | 99.5 | 5.9 / 10.3 | 744 |
| Native CPU (4 threads) | fp32 | 177.0 | 8.9 / 20.5 | 1100 |
| DirectML GPU (queries on the CPU) | fp32 | 141.2 | 9.6 / 18.2 | 1536 |

The same sample with bge-small-en-v1.5, the previous model, measured back to back: CPU int8 77.1 ms/document (this model takes 1.29x as long); DirectML fp32 83.9 ms/document (this model takes 1.68x as long).

Setup times the CPU (int8) against DirectML (fp32) once per computer on twelve sample sections of typical lengths and keeps the GPU unless it measures more than twice as slow. Windows runs a background Obsidian window at idle CPU priority, which slows CPU embedding about 3x (358 → 1040 ms on the calibration sample) while DirectML barely changes (590 → 620 ms); the factor of two assumes half of bulk indexing happens in the background. On this Iris Xe laptop the CPU is faster in the foreground, but the GPU is chosen. DirectML runs single short inputs slowly, so in GPU mode queries run on the CPU with the same fp32 weights.

## WebAssembly engine latency (removed)

Earlier releases fell back to a one-thread WebAssembly engine when the native engine could not load. It was removed: it ran about 10x slower than the native engine and could not run the newer ModernBERT models. Measured with bge-small int8, whose quality scores were produced with this engine. Sample sizes: 32 evenly spaced official queries and 12 documents per dataset. These latency samples are not full-corpus inference throughput. Warm vector scans include all corpus documents.

| Dataset | WASM query p50/p95 (ms) | WASM document p50/p95 (ms) | int8 scan p50/p95 (ms) | BM25 p50/p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| scifact | 43.9 / 82.6 | 922.1 / 1138.0 | 7.3 / 11.9 | 1.5 / 3.4 |
| nfcorpus | 32.4 / 45.2 | 818.6 / 1137.0 | 5.0 / 8.3 | 0.2 / 1.4 |

Timings depend on this machine and background load. First-time vault indexing can take minutes, especially for long documents. BM25 remains available during embedding.

## Candidate embedding models

Each model embedded the same full test sets with fp32 weights through the plugin's native engine (`scripts/embed-benchmarks-native.mjs --model`), scored with `scripts/benchmark.mjs`, and compared by `scripts/compare-models.mjs`. Intervals are paired bootstrap 95% CIs of the per-query nDCG@10 difference against bge-small-en-v1.5 (the previous default) at the same precision. Dense rows use int8 vector storage.

| Model | Parameters | Dimensions | SciFact dense | SciFact hybrid | NFCorpus dense | NFCorpus hybrid | GPU encode (ms/document) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bge-small-en-v1.5 (previous default) | 33M | 384 | 0.7128 | 0.7259 | 0.3430 | 0.3653 | 69 |
| granite-embedding-small-english-r2 (shipped) | 47M | 384 | 0.7463 [+0.012, +0.056] | 0.7451 [+0.004, +0.036] | 0.3683 [+0.009, +0.043] | 0.3763 [+0.001, +0.021] | 93 |
| granite-embedding-97m-multilingual-r2 | 97M | 384 | 0.6837 [-0.053, -0.006] | 0.7324 [-0.010, +0.024] | 0.2974 [-0.068, -0.024] | 0.3499 [-0.028, -0.004] | 73 |
| granite-embedding-english-r2 | 149M | 768 | 0.7546 [+0.020, +0.065] | 0.7551 [+0.014, +0.045] | 0.3773 [+0.017, +0.051] | 0.3792 [+0.003, +0.025] | 232 |

Granite small English r2 improves on bge-small-en-v1.5 on both datasets for dense and hybrid search, with every interval above zero, and is now the shipped model. Its int8 weights score within 0.005 nDCG@10 of fp32 (see model weight precision) and index somewhat slower than bge-small on the CPU (see embedding engines). Granite English r2 is slightly better again but 2.5x slower to encode. The multilingual model is worse on these English tasks. IBM describes its training data only as permissively licensed public and IBM-generated pairs; training-split exposure to SciFact or NFCorpus cannot be ruled out.

## Rerankers

Cross-encoders rerank the top 20 first-stage results for every test query (`scripts/benchmark-rerank.mjs`). First stage: onnx-community/granite-embedding-small-english-r2-ONNX hybrid (title-weighted BM25 + int8 dense, RRF k=60). Intervals are paired bootstrap 95% CIs of the nDCG@10 change against the first stage. Latency is the median time to score 20 pairs on DirectML GPU (Intel Iris Xe).

| Reranker | Parameters | SciFact first → rerank@20 | NFCorpus first → rerank@20 | Latency per query |
| --- | ---: | ---: | ---: | ---: |
| ettin-reranker-17m-v1 | 17M | 0.7451 → 0.7340 [-0.033, +0.011] | 0.3763 → 0.3562 [-0.035, -0.006] | 0.5 s |
| ettin-reranker-32m-v1 | 32M | 0.7451 → 0.7513 [-0.017, +0.029] | 0.3763 → 0.3585 [-0.031, -0.004] | 1.2 s |
| ettin-reranker-68m-v1 | 68M | 0.7451 → 0.7642 [-0.004, +0.043] | 0.3763 → 0.3648 [-0.025, +0.002] | 2.6 s |

No reranker improved both datasets: they help slightly on SciFact and hurt on NFCorpus, significantly so for the 17M and 32M models. With the GPU already slower than interactive search allows at 68M parameters, no reranker is shipped. A 150M reference reranker was dropped from the run for time.

## Decision

Recommended retrieval: **hybrid-int8**. Recommended vector codec: **int8**.

Dense vectors must first beat both plain and title-weighted BM25 with >= 0.01 absolute macro nDCG@10 gain over the stronger baseline, no dataset regression, and positive paired bootstrap lower bounds on each dataset. A hybrid default must independently clear the same gate. Otherwise default to heading-aware BM25. Quantization is evaluated separately from model quality.

| Candidate | SciFact nDCG gain 95% paired CI | NFCorpus nDCG gain 95% paired CI |
| --- | ---: | ---: |
| int8 vs plain BM25 | 0.0181 to 0.0847 | 0.0198 to 0.0580 |
| int8 vs title-weighted BM25 | 0.0175 to 0.0856 | 0.0184 to 0.0574 |
| hybrid-int8 vs plain BM25 | 0.0347 to 0.0793 | 0.0310 to 0.0586 |
| hybrid-int8 vs title-weighted BM25 | 0.0333 to 0.0799 | 0.0293 to 0.0574 |

Compression acceptance requires no more than 0.005 absolute nDCG loss on either dataset, at least 95% top-10 overlap, and scan p95 no slower than 1.5x int8.

The plugin initially opens with BM25 because it needs no model. This recommendation applies after optional model setup. Two English scientific datasets do not establish a universal winner for personal notes, multilingual vaults, or heading weights. The rotated-SQ4 experiment is not full TurboQuant. Rerankers were benchmarked (above) and are not shipped. See [methodology](README.md).
