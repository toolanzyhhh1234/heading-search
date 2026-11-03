// Run the exact browser worker in Node for repeatable CPU-only benchmarks.
const { parentPort, workerData } = require('node:worker_threads');
const { readFileSync } = require('node:fs');
const source = readFileSync(workerData.source, 'utf8');
globalThis.self = globalThis;
globalThis.postMessage = (message, options) => parentPort.postMessage(message, options?.transfer);
// The browser build has no Node runtime dependencies. Use the browser code path.
globalThis.process = undefined;
parentPort.on('message', data => globalThis.onmessage({data}));
eval(source);
