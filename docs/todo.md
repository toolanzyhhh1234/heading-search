# To do

## Done: switch to Granite small English r2 (2026-11-19)

- [x] **Measure Granite small r2 with int8 weights.** Within 0.005 nDCG@10 of fp32 everywhere: dense SciFact 0.7413 vs 0.7463, NFCorpus 0.3682 vs 0.3683; hybrid SciFact 0.7460 vs 0.7451, NFCorpus 0.3733 vs 0.3763. Hybrid clears the BM25 gate, so setup now switches vaults to Hybrid. CPU int8 indexing is 99.5 ms/document vs bge-small's 77.1 (1.29x), measured back to back on the i7-11390H; queries are slightly faster (5.9 vs 7.0 ms p50). See [RESULTS.md](../benchmarks/RESULTS.md).
- [x] `src/model-spec.json` is the Granite spec. `ModelStore` downloads, verifies, and deletes each variant's ONNX external-data file. The device profile records the model it was measured with, so a model change re-times the GPU at the next setup.
- [x] WebAssembly fallback removed (onnxruntime-web 1.17.3 cannot run ModernBERT; the fallback was about 10x slower). Semantic search now needs the native engine; BM25 works without it.

## Next

- [ ] **Try it in a real Obsidian vault**, including the upgrade path from a bge-small install: semantic search should report that the model changed, **Check / repair** should download Granite (52 MB int8, plus 195 MB fp32 on Windows to time the GPU), re-measure the device, and each vault should re-embed once. Both benchmarks are English scientific/medical abstracts, and IBM's training data may overlap them.
- [ ] Old bge-small files stay in `~/.cache/heading-search/models/Xenova/` after upgrading (another vault on an older plugin version may still use them). Decide whether setup should offer to remove them.

## RTX 3070 desktop: a larger model for strong GPUs

On Windows, the `onnxruntime-node` npm package reaches GPUs only through DirectML (its CUDA provider is Linux x64 only). DirectML runs on NVIDIA cards, so the 3070 needs nothing extra. A CUDA build for Windows would add a separate ONNX Runtime build plus NVIDIA CUDA/cuDNN libraries (over 1 GB), which doesn't fit one-button setup.

- [ ] Set up: `npm ci`, `python scripts/prepare-benchmarks.py`, `node scripts/pin-model.mjs <spec>` for each model (re-downloads the pinned files into `.cache/models`), and a runtime in `.cache/runtime-dev` (or the plugin's shared runtime folder after setup).
- [ ] Time the shipped model and candidates on DirectML: `node scripts/benchmark-engines.mjs --model <spec> --out engines-<key>-3070.json`.
- [ ] Score larger models with `--device directml`: Granite English r2 (149M, already pinned; SciFact/NFCorpus hybrid 0.7551/0.3792 vs small's 0.7451/0.3763), plus other base-size candidates worth adding to `benchmarks/models/`.
- [ ] If a larger model wins clearly at acceptable GPU speed, design a "strong GPU" tier: calibration would also need to pick the model, and vectors from different models can't be mixed, so each vault re-embeds when the tier changes.
