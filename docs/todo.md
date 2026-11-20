# To do

## Done: switch to Granite small English r2 (2026-11-19)

- [x] **Measure Granite small r2 with int8 weights.** Within 0.005 nDCG@10 of fp32 everywhere: dense SciFact 0.7413 vs 0.7463, NFCorpus 0.3682 vs 0.3683; hybrid SciFact 0.7460 vs 0.7451, NFCorpus 0.3733 vs 0.3763. Hybrid clears the BM25 gate, so setup now switches vaults to Hybrid. CPU int8 indexing is 99.5 ms/document vs bge-small's 77.1 (1.29x), measured back to back on the i7-11390H; queries are slightly faster (5.9 vs 7.0 ms p50). See [RESULTS.md](../benchmarks/RESULTS.md).
- [x] `src/model-spec.json` is the Granite spec. `ModelStore` downloads, verifies, and deletes each variant's ONNX external-data file. The device profile records the model it was measured with, so a model change re-times the GPU at the next setup.
- [x] WebAssembly fallback removed (onnxruntime-web 1.17.3 cannot run ModernBERT; the fallback was about 10x slower). Semantic search now needs the native engine; BM25 works without it.

## Done: DirectML GPU graph and batch tiers, measured on an RTX 3070 desktop (2026-11-19)

See [DirectML on a discrete GPU](../benchmarks/RESULTS.md#directml-on-a-discrete-gpu).

- [x] **CUDA.** `onnxruntime-node` has CUDA only on Linux x64; on Windows the GPU path is DirectML, which runs NVIDIA cards. With the fixes below DirectML is within 1.1–1.4x of CUDA (onnxruntime-gpu reference), so no CUDA engine is shipped: it would need a separate ONNX Runtime build plus over 1 GB of NVIDIA libraries.
- [x] **Why DirectML was slow.** No DirectML kernel for the ONNX `Gelu` op (still missing in onnxruntime-node 1.31.1), so all 12 layers' activations ran on the CPU with copies both ways. That also made GPU indexing slow down with the CPU in the background. `scripts/patch-gpu-graph.py` rewrites them as `Erf` (multiplication form; ORT fuses the `Div` form back into `Gelu`). The plugin bundles the 190 KB graph (`src/assets/`), pinned in `src/model-spec.json` under `weights.fp32.gpu`, and still downloads only the unchanged weights. The CPU keeps the upstream graph (its fused `Gelu` is faster there). Vectors are identical (cosine 1.000000), so no re-embedding. `ModelStore` installs bundled files offline, so upgraded installs pick up the graph without the network.
- [x] **Batch tiers.** Once the GPU wins, setup times 16-per-call small batches against 64-per-call large ones on 64 probe sections skewed short, and keeps large if 1.2x faster (RTX 3070: 637 vs 1196 ms). Profiles record `gpuBatch` and `calibrationVersion: 2`, so **Check / repair** re-measures computers that chose the CPU against the old graph.
- [x] **Larger model: no.** Granite English r2 (149M) reproduces its scores on the 3070 but gains only +0.003 to +0.010 nDCG@10 over the shipped model, with every paired interval including zero, at 3x the download, 2x the vector size, no int8 weights, and about 2.4x slower indexing.
- [x] `prepare-benchmarks.py` no longer pins a model (it still hardcoded bge-small and would have overwritten `src/model-spec.json`); `pin-model.mjs` does, and installs bundled GPU graphs.

## In progress: an engine bundled in main.js, for directory review (2026-11-20)

The [developer policies](https://docs.obsidian.md/community-directory/developer-policies) say plugins must not "install or update themselves or their dependencies", and setup downloads `onnxruntime-node`. A WebAssembly or WebGPU engine shipped inside `main.js` would avoid that. `scripts/engine-probe/build.mjs` builds a development-only plugin into a test vault that times every engine inside Obsidian on 64 SciFact abstracts and checks vectors against native fp32. First run: `benchmarks/results/engine-probe-3070-electron34.json` (i5-12600KF, RTX 3070, Obsidian 1.15.1 on the 1.8.10 installer = Electron 34 / Chromium 132, focused).

- [x] **onnxruntime-web 1.31.1 runs Granite** (1.17.3 could not). fp32 vectors are identical to native (cosine 1.000000 in Node); int8 differs slightly (0.997), so int8 indexes are not interchangeable between engines.
- [x] **CPU WebAssembly works but is slow**: 326 ms/document on one thread, against 37 for native int8 and 12 for DirectML; queries 20 ms (fine). Obsidian is not cross-origin isolated, and threaded WASM **hangs** there instead of falling back (SharedArrayBuffer exists), so only one thread is usable. The probe now skips it.
- [x] **WebGPU failed on Electron 34** with every build: JSEP stops at `[Mul] .../k_rotary/term2_mul` ("Can't perform binary op on the given tensors"); the native WebGPU EP (asyncify) fails to compile a shader with D3D (`X3663`, barrier in varying flow control); JSPI is not available. The adapter was found (NVIDIA Ampere, no shader-f16).
- [x] **Re-run on the current installer** (`benchmarks/results/engine-probe-3070-electron43.json`: Obsidian 1.15.1 on the current installer = Electron 43.8.9 / Chromium 150, same desktop). Electron is fixed by the installer, not by in-app updates.
  - **The native WebGPU EP works and beats DirectML:** JSPI build 9.5 ms/document (first pass 11.1, including shader compilation), asyncify build 9.4 (first pass 22.4), against 11.9 for native DirectML with large batches. Standard batches (16 per call) beat large ones on WebGPU (12.1). Vectors are identical to native fp32 (cosine 1.000000), so existing DirectML indexes stay valid.
  - **Queries run well on the GPU:** 8.8 ms, against 40.6 for a DirectML query (which is why DirectML sends queries to the CPU).
  - JSPI is available (not on Electron 34); the adapter now reports shader-f16. JSEP still fails on the rotary `Mul`. CPU WASM is unchanged (336 ms/document, one thread).
  - Two native rows (CPU fp32 reference, DirectML standard) had focus losses during the step; their timings may be slightly high.
- [x] **Bundle cost**, Brotli-compressed and base64-encoded in `main.js` (Obsidian's Node `zlib` decompresses Brotli): JSPI WASM 17.9 MB raw → 3.8 MB; plain CPU WASM 14.8 → 3.3; asyncify 28.7 → 5.8. Today's `main.js` is 1.0 MB.
- [ ] **Decide old installers** (no JSPI, WebGPU shader failures on Electron 34): plain CPU WASM (+3.3 MB, about 28x slower than DirectML to index), keyword search with a prompt to update the installer, or the native download as a manual opt-in.
- [x] **Iris Xe laptop, Electron 43** (i7-11390H, on AC): WebGPU is far slower than native there: JSPI fp32 289.5 ms/document (standard batches; 450 with large ones), against 94.0 for DirectML and 67.5 for native CPU int8. Queries 21 ms. Vectors match native fp32. Results reported from the laptop; `engine-probe-irisxe-electron43.json` is still to be copied into `benchmarks/results/`.
- [x] **Low-bit weights** (onnx-community publishes `model_fp16`, `model_q4`, `model_q4f16` at the pinned revision; added to `benchmarks/models/granite-small-r2.json`, hashes match Hugging Face). Quality, nDCG@10 with paired 95% CIs against fp32 (`summary-granite-small-r2-{fp16,q4,q4f16}.json`):
  - fp16 equals fp32: SciFact dense/hybrid 0.7488/0.7430, NFCorpus 0.3678/0.3767, every interval within ±0.006.
  - q4f16: hybrid 0.7500 / 0.3738 (fine); dense 0.7431 / 0.3633, NFCorpus dense −0.005, at the compression limit. q4 is similar (NFCorpus dense −0.007).
  - onnxruntime-node's **CPU** q4f16 kernel returns NaN for every query; DirectML's is finite. `embed-benchmarks-native.mjs --gpu-queries` keeps queries on DirectML for such variants. Any shipped q4f16 path must check vectors are finite.
- [x] **Desktop low-bit probe** (`engine-probe-3070-lowbit.json`, `--suite lowbit`): WebGPU JSPI fp32 10.0, fp16 8.7, q4 7.9, q4f16 8.8 ms/document; all finite; WebGPU q4f16 matches DirectML q4f16 at cosine 0.9998. Kernel profile (`engine-probe-3070-lowbit-profile.json`; its native rows failed because the window was reloaded in the same process, "OrtEnv already destroyed"): host time inside kernels is only 12–15% of each pass, so the 3070 is not dispatch-bound; MatMul / MatMulNBits is the largest host cost; `Abs`/`Sub` for the attention mask run on the CPU with 4 host-to-GPU copies per call.
- [ ] **Low-bit probe on the laptop** (`node scripts/engine-probe/build.mjs --suite lowbit --portable dist/engine-probe-portable-lowbit`): does fp16 or q4f16 bring Iris Xe WebGPU near native, and is the laptop dispatch-bound (high host share) or GPU-bound?
- [ ] Measure a Mac, and whether WebGPU indexing slows when Obsidian is in the background.
- [ ] Prototype a bundled backend: JSPI WebGPU fp32 for documents and queries, CPU WASM when there is no usable adapter, chosen at setup like the DirectML calibration.

## Next

- [ ] **Try it in a real Obsidian vault**, including the upgrade path from a bge-small install: semantic search should report that the model changed, **Check / repair** should download Granite (52 MB int8, plus 195 MB fp32 on Windows to time the GPU), re-measure the device, and each vault should re-embed once. Both benchmarks are English scientific/medical abstracts, and IBM's training data may overlap them.
- [ ] **Also try the GPU-graph upgrade in Obsidian:** an install with fp32 weights should start without a download (the graph installs offline), keep its vectors, and **Check / repair** should re-measure and may choose large batches.
- [ ] **Re-measure the Iris Xe laptop** with the GPU graph (`node scripts/benchmark-engines.mjs`, then `node scripts/report-benchmark.mjs`): its committed engine and calibration numbers predate it, and it should confirm the batch tier stays small on integrated graphics.
- [ ] **Foreground RTX 3070 timings.** The committed 3070 runs were throttled: Windows 11 moves an unfocused window's process to efficiency cores and lowers its GPU priority, and the session was driven remotely. Unthrottled vs throttled the same day: CPU int8 48 vs 314–353 ms/document, DirectML on the upstream graph 40 vs 162–267. Ratios measured back to back hold; absolute numbers do not. With the terminal focused at the desk (don't switch windows, about 3 minutes), run `node scripts/benchmark-engines.mjs --out engines-3070-gpu-graph.json`, then `node scripts/report-benchmark.mjs`, and update the throttling note in `benchmarks/results/gpu-rtx3070.json` and the README's "267 to 15 ms" sentence. Untested alternative: `powercfg /powerthrottling disable /path <node.exe>` from an elevated prompt.

### Picking this up

Benchmark setup on a fresh clone or a cleared `.cache` (about 1.1 GB of downloads):

```sh
npm ci
python scripts/prepare-benchmarks.py                    # SciFact + NFCorpus into benchmarks/data
node scripts/pin-model.mjs src/model-spec.json          # shipped model, plus the bundled GPU graph
node scripts/pin-model.mjs benchmarks/models/granite-r2.json   # only to revisit the 149M model
npm install --prefix .cache/runtime-dev --ignore-scripts --no-save onnxruntime-node@1.22.0   # the pinned engine, as setup installs it
node scripts/runtime-smoke.mjs --weights fp32           # CPU and DirectML must agree (Windows)
```

`python scripts/patch-gpu-graph.py` needs `onnx==1.18.0`. On Windows, pip into a long path can fail with WinError 206 (onnx ships deeply nested test data); install into a short `--target` folder and set `PYTHONPATH`. The CUDA reference numbers came from a throwaway script with `onnxruntime-gpu==1.22.0` in a separate venv; it is not part of the repo because the plugin does not ship CUDA.
- [ ] Old bge-small files stay in `~/.cache/heading-search/models/Xenova/` after upgrading (another vault on an older plugin version may still use them). Decide whether setup should offer to remove them.
