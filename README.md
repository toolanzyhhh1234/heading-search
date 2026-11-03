# Heading Search

Search an Obsidian vault by its Markdown headings, keywords, and optionally by meaning. Everything runs locally on a desktop CPU. No API keys, Python installation, model server, or separate app are required by users.

**Development preview:** the plugin has not been published to Obsidian Community Plugins. This repository includes the build, release assets, benchmark tools, and submission instructions. See [Publishing](docs/publishing.md) for the remaining distribution steps.

## Getting started

Once the plugin is accepted into Community Plugins:

1. Open **Settings → Community plugins → Browse**, search for **Heading Search**, then install and enable it.
2. Click the search icon in the ribbon, or run **Heading Search: Search notes** in the command palette.
3. Type a query. Keyword search works immediately, while notes are indexed in the background.
4. For semantic search, open **Settings → Heading Search → Enable semantic search**. This downloads the shared model once. Then choose **Semantic** or **Hybrid** in the search dialog.

Use the arrow keys and Enter to open a result at the matching section. Ctrl/Cmd + Enter opens a new tab. The plugin shows the strongest matching section from each note.

For the development preview, extract `dist/heading-search-0.1.0.zip` into your vault's `.obsidian/plugins/` directory, then enable Heading Search in Community plugins. The resulting folder must contain `main.js`, `manifest.json`, and `styles.css` directly. Do not move `node_modules` or the source code into your vault.

## Headings are part of the search

- CommonMark parsing recognizes ATX (`#`) and setext headings and preserves H1–H6 ancestry.
- Sections do not cross a heading boundary. Large sections split into bounded pieces, with the title and heading context attached to every piece.
- Heading-looking text inside fenced code, HTML comments, or blockquotes does not create a new section. YAML frontmatter is excluded.
- Keyword ranking gives the note title and higher-level headings extra weight. Body text stays searchable.
- New, edited, deleted, moved, and excluded notes update incrementally. Result links use source line positions.

## One model for all your vaults

The optional model is **BGE-small-en-v1.5**, a small English embedding model with a pinned, 34 MB quantized ONNX graph and 384-dimensional output. It runs in a background worker using CPU SIMD WebAssembly, one thread at a time. The runtime is bundled in `main.js`; the plugin does not download executable code or install dependencies.

Model assets are stored outside your vaults in:

```text
Windows: %USERPROFILE%\.cache\heading-search\models\
macOS/Linux: ~/.cache/heading-search/models/
```

Every vault under the same computer account uses that location. Simultaneous setup requests use a shared download lock. Assets are checked against pinned SHA-256 hashes and installed with atomic file renames. A new vault can reuse a complete cache without an internet connection. Different machines or computer accounts need their own cache; running the model in two open vaults can still use memory in both windows.

Only model weights and tokenizer data are shared. Each vault's note text, index, and compressed vectors stay in that vault's `.obsidian/plugins/heading-search/index/` folder (or its configured equivalent). There is no cross-vault search or shared note index.

## Retrieval and storage decisions

Keyword BM25 is the initial default and needs no model download. Semantic search and reciprocal-rank hybrid search are available after setup. The benchmark decision gate compares them against BM25 and keeps heading-aware BM25 as the recommendation unless vectors demonstrate a reliable improvement. See [benchmark methodology and results](benchmarks/README.md).

Vector data is stored in a binary file rather than JSON floating-point arrays:

| Format | Bytes per 384-dimensional vector | Reduction vs float32 |
| --- | ---: | ---: |
| Float32 | 1,536 | reference |
| 8-bit scalar | 392 | 74.5% |
| 4-bit scalar | 200 | 87.0% |
| 4-bit rotated scalar | 260 | 83.1% |

These sizes include per-vector scale/norm data; each file also has a 16-byte header. Text and heading metadata are additional. The rotated 4-bit experiment uses a fixed sign/Hadamard rotation and Gaussian scalar codebook, inspired by the rotation-and-quantization approach studied in TurboQuant. It is **not a full TurboQuant implementation**: there is no QJL residual correction, and its padded 512-dimensional representation has measurable overhead. Compression is accepted only when measured retrieval loss is small. The 4-bit choices remain explicitly experimental.

An expensive reranker is not included in this preview. The initial focus is responsive CPU search; a reranker needs separate quality and latency evidence before becoming a default.

## Privacy and network use

- No telemetry or analytics. Notes and queries are not uploaded.
- Network access occurs only when you click model setup/repair and an asset is missing or corrupt. The plugin requests pinned public files from `huggingface.co`; Hugging Face may redirect the file transfer to its CDN. The provider receives normal download request information, including your IP address.
- The plugin accesses the shared model folder outside the vault for reuse across vaults. It does not write note content there.
- Once setup completes, model loading, indexing, and search work offline.
- The plugin is desktop-only because it uses a shared filesystem cache. English is the evaluated language; other languages are not claimed to have equivalent retrieval quality.

Pause/resume indexing and exclude folders in settings. Disabling semantic search in one vault leaves the shared model available to other vaults. To remove all generated data, disable the plugin and remove that vault's plugin index directory; remove the shared model folder separately when no vault needs it. Vault sync tools may synchronize the vault-local index; configure exclusions if desired.

## Development

Requires Node.js 22+:

```sh
npm ci
npm test
npm run package
```

`npm run package` writes the installable ZIP and the three Community Plugins release assets under `dist/release/`. `npm run dev` watches the main plugin source. Changes to the embedding worker require restarting the build.

Benchmark data preparation additionally uses Python 3.11+ and Requests. The optional native-runtime diagnostic uses NumPy, tokenizers, and ONNX Runtime. Final quality measurements use the same JavaScript/WASM engine shipped in the plugin. Those are development tools, not user requirements.

## Credits

The plugin is MIT licensed. It uses ONNX Runtime (MIT), Hugging Face Tokenizers.js (Apache-2.0), the unified CommonMark parser (MIT), and porter-stemmer (MIT). BGE was developed by the Beijing Academy of Artificial Intelligence; the pinned ONNX conversion is published by Xenova. See [third-party notices](THIRD_PARTY_NOTICES.md) for licenses and sources.
