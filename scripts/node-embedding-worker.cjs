// Run the exact browser worker in Node for repeatable benchmarks.
const { parentPort, workerData } = require('node:worker_threads');
const { readFileSync } = require('node:fs');
const source = readFileSync(workerData.source, 'utf8');
globalThis.self = globalThis;
globalThis.postMessage = (message, options) => parentPort.postMessage(message, options?.transfer);
// Obsidian workers expose Node's require, which the native engine uses to load ONNX Runtime.
globalThis.require = require;
parentPort.on('message', data => globalThis.onmessage({data}));
eval(source);
