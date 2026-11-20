# Laptop test machine

The Iris Xe laptop (i7-11390H, Windows) measures what the development desktop (RTX 3070) cannot: integrated graphics.
Code changes happen on the desktop. Work is handed over through this branch, not by copying files.

## Rules for the laptop session

- Pull before starting; push when done.
- Only add files under `benchmarks/results/` (and the "Results log" below). Don't change code, scripts, or other docs.
  If something is broken, describe it in the commit message or the results log, with the error text, and stop.
- Commit as the GitHub no-reply address, never a personal one (GitHub rejects the push otherwise). Check with
  `git config user.email`; the address to use is `git log -1 --format=%ae origin/main`.
- Keep Obsidian focused and the laptop on AC power during timed runs. Quit Obsidian completely between runs that use
  the native engine (a reload in the same process gives "OrtEnv already destroyed").
- Test vaults live under `.test-vault/` (ignored by git).

## One-time setup

Needs Node 22, Python 3 with `requests`, and the current Obsidian installer (Electron 43 or later; Settings > About).

```sh
git clone https://github.com/toolanzyhhh1234/heading-search.git   # or, in an existing clone: git fetch
cd heading-search
git checkout directml-gpu-graph
git config user.email "$(git log -1 --format=%ae origin/main)"
npm ci
python scripts/prepare-benchmarks.py                                       # SciFact + NFCorpus
node scripts/pin-model.mjs src/model-spec.json                             # shipped model + GPU graph
node scripts/pin-model.mjs benchmarks/models/granite-small-r2.json         # fp16 / q4 / q4f16 candidates
npm install --prefix .cache/runtime-dev --ignore-scripts --no-save onnxruntime-node@1.22.0
npm install --prefix .cache/ortweb-dev --ignore-scripts --no-save onnxruntime-web@1.31.1
```

About 1.5 GB of downloads, once. Afterwards `git pull` is enough unless a task says otherwise.

## Running the engine probe

```sh
node scripts/engine-probe/build.mjs --vault .test-vault/probe --suite lowbit   # or --suite engines
```

Open `.test-vault/probe` as a vault in Obsidian (turn on community plugins if asked) and keep it focused until the
notice says the probe finished. Results: `.test-vault/probe/.obsidian/plugins/heading-search-engine-probe/results.json`
(`"done": true` at the end). Copy it to `benchmarks/results/engine-probe-irisxe-<suite>[-runN].json`, commit, push.

## Current task

Time the plugin's new built-in GPU engine (WebGPU, fp16 weights) on this laptop, on the same notes as the desktop.
The person at the laptop does the Obsidian clicks; you prepare, watch files, and record.

1. `git pull`, then `npm ci` (new dependency: onnxruntime-web), `npm run build`,
   `node scripts/make-benchmark-vault.mjs --dataset scifact` (5,183 notes in `.test-vault/scifact`), and
   `npm run deploy:test-vault`.
2. Record the current `~/.cache/heading-search/runtime/device.json` if it exists (an earlier native setup), and list the
   files under `~/.cache/heading-search/models/`. Setup may replace both.
3. Laptop on AC power. Ask the person to open `.test-vault/scifact` as a vault in Obsidian (trust the plugin), then
   Settings > Heading Search > **Enable semantic search** (or **Use shared model** / **Check / repair**, whichever shows),
   and keep the window focused until the index status reads `8499 sections ready · 8499 embedded (… new in … s)`.
   Expect about 4–6 minutes for indexing, after a 97 MB download if the fp16 weights are not there yet.
4. Ask for the setup message (it names the engine, adapter, and the 12-section time) and the final status line.
   Copy the new `device.json` to `benchmarks/results/plugin-irisxe-device.json`.
5. Ask for two searches in Hybrid mode, for example "vitamin D and bone fractures" and "how do statins affect muscle",
   and whether the top results look relevant.
6. Add a Results log entry: setup message, the embedded count and seconds, ms per section, the old and new device
   choice, what was downloaded or removed, and anything unusual (focus losses, errors). Commit and push.

Desktop reference (RTX 3070): WebGPU chosen, 12 probe sections in 81 ms, 8,499 sections in about 65 s.

## Results log

Short notes from the laptop, newest first: date, task, commit, anything unusual.

- 2026-11-20, add the Iris Xe engine probe results, commit "Add the Iris Xe engine probe results". One-time setup done
  in the laptop's existing clone (Obsidian 1.15.1, Electron 43.8.9). The three files are byte-identical copies.
  `engine-probe-irisxe-lowbit.json` has focus losses on three native rows (CPU fp32 4, DirectML fp32 1, DirectML
  q4f16 1), which inflated them by 5-13%; `-run2` has none and its WebGPU rows match run 1 within 2%. Native
  DirectML still varies by about 10% between runs without focus losses (fp16: 108.7 vs 96.0 ms/document).
