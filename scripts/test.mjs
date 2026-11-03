import { build } from 'esbuild';
import { spawn } from 'node:child_process';
import { readdir, mkdir } from 'node:fs/promises';
await mkdir('dist/tests',{recursive:true});
const tests=(await readdir('tests')).filter(file=>file.endsWith('.test.ts')).map(file=>`tests/${file}`);
await build({entryPoints:tests,outdir:'dist/tests',outExtension:{'.js':'.cjs'},bundle:true,platform:'node',format:'cjs',external:['obsidian']});
const child=spawn(process.execPath,['--test',...(await readdir('dist/tests')).filter(file=>file.endsWith('.test.cjs')).map(file=>`dist/tests/${file}`)],{stdio:'inherit'});
child.on('exit',code=>process.exitCode=code??1);
