# Heading Search

Search an Obsidian vault by its Markdown headings, keywords, and optionally by meaning. Everything runs locally on your desktop, on the CPU or a DirectML-capable GPU. No API keys, Python installation, model server, or separate app are required by users.

**Development preview:** the plugin has not been published to Obsidian Community Plugins. This repository includes the build, release assets, benchmark tools, and submission instructions. See [Publishing](docs/publishing.md) for the remaining distribution steps.

## Getting started

Once the plugin is accepted into Community Plugins:

1. Open **Settings → Community plugins → Browse**, search for **Heading Search**, then install and enable it.
2. Click the search icon in the ribbon, or run **Heading Search: Search notes** in the command palette.
3. Type a query. Keyword search works immediately, while notes are indexed in the background.
4. For semantic search, open **Settings → Heading Search → Enable semantic search**. This downloads the shared model and fast engine once per computer. Then choose **Semantic** or **Hybrid** in the search dialog.

Use the arrow keys and Enter to open a result at the matching section. Ctrl/Cmd + Enter opens a new tab. The plugin shows the strongest matching section from each note.

For the development preview, extract `dist/heading-search-0.1.0.zip` into your vault's `.obsidian/plugins/` directory, then enable Heading Search in Community plugins. The resulting folder must contain `main.js`, `manifest.json`, and `styles.css` directly. Do not move `node_modules` or the source code into your vault.

## Headings are part of the search

- CommonMark parsing recognizes ATX (`#`) and setext headings and preserves H1–H6 ancestry.
- Sections do not cross a heading boundary. Large sections split into bounded pieces, with the title and heading context attached to every piece.
- Heading-looking text inside fenced code, HTML comments, Obsidian `%%` comments, or blockquotes does not create a new section. YAML frontmatter and Obsidian comment text are excluded.
- Keyword ranking gives the note title and higher-level headings extra weight. Body text stays searchable.
- New, edited, deleted, moved, and excluded notes update incrementally. Result links use source line positions.

## One model for all your vaults

The optional model is IBM's **Granite Embedding small English r2** (47M parameters, Apache-2.0), a ModernBERT-based English embedding model with 384-dimensional output and pinned ONNX weights. It replaced BGE-small-en-v1.5, which it beats on both benchmark collections for semantic and hybrid search; see [candidate embedding models](benchmarks/RESULTS.md#candidate-embedding-models). Updating from a version that used BGE-small: choose **Check / repair** in settings to download the new model, and each vault embeds its notes again.

### The fast engine

Embeddings run in a background worker on Microsoft's native ONNX Runtime ([`onnxruntime-node`](https://www.npmjs.com/package/onnxruntime-node)), so indexing uses several CPU cores or the GPU. On the development laptop (Intel i7-11390H, Iris Xe), it indexes about 100 ms per document on the CPU (the single-thread WebAssembly engine of earlier versions took about 690 ms with the smaller BGE model); see [embedding engines](benchmarks/RESULTS.md#embedding-engines).

- **Setup installs it with one button.** The plugin downloads Microsoft's official npm packages for `onnxruntime-node` and `onnxruntime-common` (93 MB) from `registry.npmjs.org`, checks them against the registry's pinned SHA-512, keeps only your platform's files (about 60 MB on Windows), and checks each extracted file against a pinned SHA-256 before installing it. Nothing is compiled, and no install scripts run. Supported: Windows, macOS, and Linux on x64 and arm64.
- **Setup picks a device once per computer.** int8 weights (52 MB) run on the CPU. On Windows, setup also downloads full-precision weights (195 MB) and times DirectML against the CPU on typical section lengths. The GPU, which works with Intel, AMD, and NVIDIA graphics, is kept unless it measures more than twice as slow. Otherwise the full-precision weights are deleted again. When the GPU is kept, setup also times larger batches, which index a discrete graphics card about twice as fast.
- **DirectML runs a GPU version of the model graph that ships with the plugin.** DirectML cannot run the model's GELU activation, so in the published graph every layer stepped back to the CPU. The plugin's copy (190 KB, [made by a script](scripts/patch-gpu-graph.py) from the pinned upstream graph) writes it as an equivalent formula, and loads the same downloaded weights. Vectors are unchanged, so existing indexes stay valid. Those CPU steps also made GPU indexing slow down with the CPU while Obsidian was in the background: on an RTX 3070 in a throttled background session, indexing went from 267 to 15 ms per section, measured back to back. NVIDIA cards need no CUDA install: with this graph, DirectML comes within about 1.4x of CUDA. See [DirectML on a discrete GPU](benchmarks/RESULTS.md#directml-on-a-discrete-gpu). Turn **Allow the GPU on this computer** off in settings if graphics drivers misbehave. The allowance exists because Windows runs a background Obsidian window at idle CPU priority, so CPU indexing slows about 3x while Obsidian is not focused, while GPU indexing is barely affected. Setup measures in the foreground.
- **The choice is shared by every vault and is not re-measured on its own**, because different weights produce incompatible vectors. Changing the GPU setting re-measures, and vaults embed their notes once more if the weights change.
- **Semantic search requires the fast engine.** If it cannot start, for example because security software blocks native libraries, settings and the search dialog show why, and keyword search keeps working. **Check / repair** in settings re-verifies and reinstalls it.
- Both weight precisions score the same on the benchmarks (within ±0.005 nDCG@10). See [model weight precision](benchmarks/RESULTS.md#model-weight-precision).

Model assets are stored outside your vaults in:

```text
Windows: %USERPROFILE%\.cache\heading-search\models\   (weights and tokenizer)
         %USERPROFILE%\.cache\heading-search\runtime\  (fast engine and this computer's device choice)
macOS/Linux: ~/.cache/heading-search/models/ and ~/.cache/heading-search/runtime/
```

Every vault under the same computer account uses that location. Simultaneous setup requests use a shared download lock. Assets are checked against pinned SHA-256 hashes and installed with atomic file renames. A new vault can reuse a complete cache without an internet connection. Different machines or computer accounts need their own cache; running the model in two open vaults can still use memory in both windows.

Only model weights, tokenizer data, and the engine are shared. Each vault's note text, index, and compressed vectors stay in that vault's `.obsidian/plugins/heading-search/index/` folder (or its configured equivalent). There is no cross-vault search or shared note index.

## Retrieval and storage decisions

Keyword BM25 is the initial default and needs no model download. Semantic search and reciprocal-rank hybrid search are available after setup. The benchmark decision gate compares them against BM25 and keeps heading-aware BM25 as the recommendation unless vectors demonstrate a reliable improvement. See the [measured results](benchmarks/RESULTS.md) and [methodology](benchmarks/README.md).

With Granite, both semantic and hybrid search clear the gate on both collections, and **Hybrid** scores highest, so setup switches a vault to Hybrid. You can change the mode at any time. Earlier, with BGE-small, pure-vector gains were inconclusive and BM25 stayed the default.

Vector data is stored in a binary file rather than JSON floating-point arrays:

| Format | Bytes per 384-dimensional vector | Reduction vs float32 |
| --- | ---: | ---: |
| Float32 | 1,536 | reference |
| 8-bit scalar | 392 | 74.5% |
| 4-bit scalar | 200 | 87.0% |
| 4-bit rotated scalar | 260 | 83.1% |

These sizes include per-vector scale/norm data; each file also has a 16-byte header. Text and heading metadata are additional. The rotated 4-bit experiment uses a fixed sign/Hadamard rotation and Gaussian scalar codebook, inspired by the rotation-and-quantization approach studied in TurboQuant. It is **not a full TurboQuant implementation**: there is no QJL residual correction, and its padded 512-dimensional representation has measurable overhead. Compression is accepted only when measured retrieval loss is small. The 4-bit choices remain explicitly experimental.

No reranker is included. Small cross-encoders (Ettin 17M–68M) reranking the top 20 hybrid results did not improve both benchmark collections, and they added 0.5–2.6 s per query on an Intel Iris Xe GPU. See [rerankers](benchmarks/RESULTS.md#rerankers).

Vector retrieval currently uses an exact scan. Query time grows with the number of indexed sections. Initial embedding of a large vault can take minutes; keyword search stays available while it runs.

## Privacy and network use

- No telemetry or analytics. Notes and queries are not uploaded.
- Network access occurs only when you click setup/repair (or change the GPU setting) and an asset is missing or corrupt. The plugin requests pinned public files from `huggingface.co` (model) and `registry.npmjs.org` (fast engine); Hugging Face may redirect the file transfer to its CDN. The providers receive normal download request information, including your IP address.
- The fast engine is native code from Microsoft, loaded into Obsidian's process. Every file is verified against pinned hashes when installed. Startup compares sizes and modification times with that verification; **Check / repair** verifies every byte again.
- The plugin accesses the shared model and engine folders outside the vault for reuse across vaults. It does not write note content there.
- Once setup completes, model loading, indexing, and search work offline.
- The plugin is desktop-only because it uses a shared filesystem cache. English is the evaluated language; other languages are not claimed to have equivalent retrieval quality.

Pause/resume indexing and exclude folders in settings. Disabling semantic search in one vault leaves the shared model available to other vaults. To remove all generated data, disable the plugin and remove that vault's plugin index directory; remove the shared `heading-search` cache folder separately when no vault needs it. Vault sync tools may synchronize the vault-local index; configure exclusions if desired.

## Development

Requires Node.js 22+:

```sh
npm ci
npm test
npm run package
```

`npm run package` writes the installable ZIP and the three Community Plugins release assets under `dist/release/`. `npm run dev` watches the main plugin source. Changes to the embedding worker require restarting the build.

Benchmark data preparation additionally uses Python 3.11+ and Requests. The optional Python native-runtime diagnostic uses NumPy, tokenizers, and ONNX Runtime. The plugin's own engines are benchmarked through its worker bundle: `node scripts/embed-benchmarks-native.mjs --weights fp32 --runtime <dir>` scores another weight precision, `node scripts/benchmark-engines.mjs --runtime <dir>` measures speed, and `node scripts/runtime-smoke.mjs --runtime <dir>` checks the engine (add `--weights fp32` on Windows to also check that CPU and DirectML vectors agree). `<dir>` holds `node_modules/onnxruntime-node`, for example `~/.cache/heading-search/runtime/onnxruntime-node-1.22.0-win32-x64` after setup. `node scripts/pin-runtime.mjs <version>` re-pins the engine hashes in `src/runtime-spec.json`. To evaluate another model, describe it in `benchmarks/models/` (embedders) or `benchmarks/rerankers/`, pin it with `node scripts/pin-model.mjs <spec>`, then use `--model <spec>` with the embedding and benchmark scripts and `node scripts/compare-models.mjs`, or `node scripts/benchmark-rerank.mjs --reranker <spec>`. These are development tools, not user requirements.

## Credits

The plugin is MIT licensed. It uses ONNX Runtime (MIT; the fast engine downloads Microsoft's `onnxruntime-node`, which on Windows includes DirectML under Microsoft's DirectML license), Hugging Face Tokenizers.js (Apache-2.0), the unified CommonMark parser (MIT), and porter-stemmer (MIT). Granite Embedding was developed by IBM; the pinned ONNX conversion is published by onnx-community. The model weights are downloaded at setup, not bundled; the plugin bundles only a modified copy of the small fp32 graph file for DirectML. See [third-party notices](THIRD_PARTY_NOTICES.md) for licenses and sources.
