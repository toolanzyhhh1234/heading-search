import { build, context } from 'esbuild';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { brotliCompressSync, constants } from 'node:zlib';
import { createHash } from 'node:crypto';
import path from 'node:path';

export async function buildWorker() {
  const result = await build({
    entryPoints: ['src/embedding/worker.ts'], bundle: true, write: false, format: 'iife',
    platform: 'browser', target: 'chrome120', minify: true, legalComments: 'inline',
  });
  return result.outputFiles[0].text;
}

/** The built-in WebGPU engine's worker: an ES module, because the onnxruntime-web bundle is one. */
export async function buildWebWorker() {
  const result = await build({
    entryPoints: ['src/embedding/web-worker.ts'], bundle: true, write: false, format: 'esm',
    platform: 'browser', target: 'chrome120', minify: true, legalComments: 'inline', supported: {'top-level-await': true},
  });
  return result.outputFiles[0].text;
}

/**
 * onnxruntime-web's JSPI WebAssembly binary, Brotli-compressed (17.9 MB to 3.8 MB) and base64-encoded for main.js.
 * Compression takes a while at the highest quality, so the result is cached in dist/ by the binary's hash.
 */
export async function webRuntimeWasm() {
  const wasm = await readFile('node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jspi.wasm');
  const cache = `dist/ort-wasm-simd-threaded.jspi.${createHash('sha256').update(wasm).digest('hex').slice(0, 16)}.br.b64`;
  try { return await readFile(cache, 'utf8'); } catch {}
  const encoded = brotliCompressSync(wasm, {params: {[constants.BROTLI_PARAM_QUALITY]: 11, [constants.BROTLI_PARAM_LGWIN]: 24,
    [constants.BROTLI_PARAM_SIZE_HINT]: wasm.length}}).toString('base64');
  await mkdir('dist', {recursive: true});
  await writeFile(cache, encoded);
  return encoded;
}

if (process.argv[1]?.replaceAll('\\', '/').endsWith('/scripts/build.mjs')) {
  const source = await buildWorker();
  const webSource = await buildWebWorker();
  const webWasm = await webRuntimeWasm();
  const license = await readFile('LICENSE', 'utf8');
  const notices = await readFile('THIRD_PARTY_NOTICES.md', 'utf8');
  await mkdir('dist', {recursive: true});
  await writeFile('dist/embedding-worker.js', source);
  const options = {
    entryPoints: ['src/main.ts'], outfile: 'main.js', bundle: true, format: 'cjs',
    platform: 'node', target: 'es2022', external: ['obsidian', 'electron'],
    define: {EMBEDDING_WORKER_SOURCE: JSON.stringify(source), WEB_WORKER_SOURCE: JSON.stringify(webSource), WEB_WASM_BROTLI: JSON.stringify(webWasm)},
    loader: {'.onnx': 'binary'}, sourcemap: false, minify: false, legalComments: 'inline',
    banner: {js: `/*!\n${(license+'\n'+notices).replaceAll('*/','* /')}\n*/`},
  };
  if (process.argv.includes('--watch')) { const ctx = await context(options); await ctx.watch(); }
  else { await build(options); console.log('Built main.js with the embedding worker.'); }
}
