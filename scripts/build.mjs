import { build, context } from 'esbuild';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

export const wasmPlugin = {
  name: 'inline-wasm',
  setup(builder) {
    builder.onResolve({filter: /\?base64$/}, args => ({
      path: path.resolve('node_modules', args.path.replace(/\?base64$/, '')), namespace: 'base64',
    }));
    builder.onLoad({filter: /.*/, namespace: 'base64'}, async args => ({
      contents: `export default ${JSON.stringify((await readFile(args.path)).toString('base64'))};`, loader: 'js',
    }));
  },
};

export async function buildWorker() {
  const result = await build({
    entryPoints: ['src/embedding/worker.ts'], bundle: true, write: false, format: 'iife',
    platform: 'browser', target: 'chrome120', minify: true, legalComments: 'inline',
    plugins: [wasmPlugin],
  });
  return result.outputFiles[0].text;
}

if (process.argv[1]?.replaceAll('\\', '/').endsWith('/scripts/build.mjs')) {
  const source = await buildWorker();
  const license = await readFile('LICENSE', 'utf8');
  const notices = await readFile('THIRD_PARTY_NOTICES.md', 'utf8');
  await mkdir('dist', {recursive: true});
  await writeFile('dist/embedding-worker.js', source);
  const options = {
    entryPoints: ['src/main.ts'], outfile: 'main.js', bundle: true, format: 'cjs',
    platform: 'node', target: 'es2022', external: ['obsidian', 'electron'],
    define: {EMBEDDING_WORKER_SOURCE: JSON.stringify(source)},
    sourcemap: false, minify: false, legalComments: 'inline',
    banner: {js: `/*!\n${(license+'\n'+notices).replaceAll('*/','* /')}\n*/`},
  };
  if (process.argv.includes('--watch')) { const ctx = await context(options); await ctx.watch(); }
  else { await build(options); console.log('Built main.js with bundled CPU runtime and worker.'); }
}
