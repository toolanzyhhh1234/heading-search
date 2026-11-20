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

Commit the Iris Xe probe results from before this branch was shared, if they are still in the old working directory
(outside the repo): `engine-probe-irisxe-electron43.json`, `engine-probe-irisxe-lowbit.json`, and
`engine-probe-irisxe-lowbit-run2.json`. Copy them unchanged into `benchmarks/results/`, commit
("Add the Iris Xe engine probe results"), and push. Nothing needs to run.

## Results log

Short notes from the laptop, newest first: date, task, commit, anything unusual.

- 2026-11-20, add the Iris Xe engine probe results, commit "Add the Iris Xe engine probe results". One-time setup done
  in the laptop's existing clone (Obsidian 1.15.1, Electron 43.8.9). The three files are byte-identical copies.
  `engine-probe-irisxe-lowbit.json` has focus losses on three native rows (CPU fp32 4, DirectML fp32 1, DirectML
  q4f16 1), which inflated them by 5-13%; `-run2` has none and its WebGPU rows match run 1 within 2%. Native
  DirectML still varies by about 10% between runs without focus losses (fp16: 108.7 vs 96.0 ms/document).
