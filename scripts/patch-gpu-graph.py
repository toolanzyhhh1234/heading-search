"""Builds the DirectML graph for a model's fp32 weights: the pinned upstream graph with each ONNX Gelu rewritten.

DirectML (onnxruntime 1.22 to at least 1.31) has no kernel for the opset-20 Gelu op, so every layer's activation ran
on the CPU with two copies across the bus; on an RTX 3070 that made the GPU barely faster than the CPU. The exact
form 0.5 * x * (1 + erf(x * (1/sqrt 2))) runs on the GPU and gives the same vectors (cosine 1.000000). It must stay a
multiplication: ORT's GeluFusion turns the Div(x, sqrt 2) spelling back into Gelu, for DirectML too.

Only the small graph file changes. It references the same external weights file as the upstream graph, so the plugin
bundles this graph and still downloads the unchanged weights from Hugging Face. The CPU keeps the upstream graph,
whose fused Gelu kernel is faster there.

Usage: python scripts/patch-gpu-graph.py [spec.json] [--write]
  Reads the upstream graph from .cache/models (see scripts/pin-model.mjs) and checks the result against the spec's
  pinned "gpu" entry. --write stores the graph and its hash instead, after a deliberate change to this script.
Requires onnx==1.18.0 (benchmarks/requirements.txt); other versions may serialize different bytes.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
write = '--write' in sys.argv
spec_path = ROOT / (args[0] if args else 'src/model-spec.json')
spec = json.loads(spec_path.read_text('utf8'))
variant = spec['weights']['fp32']
directory = ROOT / '.cache' / 'models' / spec['id'] / spec['revision']
upstream = (directory / variant['path']).read_bytes()
if hashlib.sha256(upstream).hexdigest() != variant['sha256']:
    raise SystemExit(f'{variant["path"]} does not match its pinned hash; run node scripts/pin-model.mjs {spec_path.relative_to(ROOT).as_posix()}')

model = onnx.ModelProto()
model.ParseFromString(upstream)  # weights stay external: only references to model.onnx_data are kept
graph = model.graph
for name, value in {'gpu_gelu_inv_sqrt2': 1 / math.sqrt(2), 'gpu_gelu_one': 1.0, 'gpu_gelu_half': 0.5}.items():
    graph.initializer.append(numpy_helper.from_array(np.array(value, np.float32), name))
nodes, replaced = [], 0
for node in graph.node:
    if node.op_type != 'Gelu' or node.domain not in ('', 'ai.onnx'):
        nodes.append(node)
        continue
    approximate = next((a.s.decode() for a in node.attribute if a.name == 'approximate'), 'none')
    if approximate != 'none':
        raise SystemExit(f'{node.name}: Gelu approximate={approximate} is not handled')
    x, y, prefix = node.input[0], node.output[0], node.name or f'gelu_{replaced}'
    nodes += [
        helper.make_node('Mul', [x, 'gpu_gelu_inv_sqrt2'], [f'{prefix}/scaled'], f'{prefix}/Scale'),
        helper.make_node('Erf', [f'{prefix}/scaled'], [f'{prefix}/erf'], f'{prefix}/Erf'),
        helper.make_node('Add', [f'{prefix}/erf', 'gpu_gelu_one'], [f'{prefix}/shifted'], f'{prefix}/Shift'),
        helper.make_node('Mul', [x, f'{prefix}/shifted'], [f'{prefix}/product'], f'{prefix}/Product'),
        helper.make_node('Mul', [f'{prefix}/product', 'gpu_gelu_half'], [y], f'{prefix}/Half'),
    ]
    replaced += 1
if not replaced:
    raise SystemExit('No Gelu nodes found; this model does not need a separate GPU graph.')
del graph.node[:]
graph.node.extend(nodes)
# Apache-2.0 section 4(b): modified files carry a notice of the change.
model.metadata_props.append(onnx.StringStringEntryProto(key='heading-search.modification',
    value=f'Modified by Heading Search: {replaced} Gelu nodes rewritten as Erf for DirectML (scripts/patch-gpu-graph.py). '
          f'Source: {spec["id"]}@{spec["revision"]}/{variant["path"]}'))
data = model.SerializeToString(deterministic=True)
digest = hashlib.sha256(data).hexdigest()
asset = ROOT / 'src' / 'assets' / f'{spec["key"]}.gpu.onnx'
entry = {'path': str(Path(variant['path']).with_name('model.gpu.onnx').as_posix()),
         'bundled': asset.relative_to(ROOT).as_posix(), 'sha256': digest, 'bytes': len(data)}
# Beside the weights in the development cache, where the benchmarks load it and the checker can resolve external data.
installed = directory / entry['path']
installed.write_bytes(data)
onnx.checker.check_model(str(installed))

if write:
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(data)
    variant['gpu'] = entry
    spec_path.write_text(json.dumps(spec, indent=2) + '\n', 'utf8', newline='\n')
    print(f'Wrote {asset.relative_to(ROOT).as_posix()}: {replaced} Gelu nodes rewritten, {len(data):,} bytes, sha256 {digest}')
else:
    pinned = variant.get('gpu')
    if not pinned or pinned['sha256'] != digest or pinned['bytes'] != len(data):
        raise SystemExit(f'The GPU graph ({digest}) differs from the pinned one; rerun with --write after reviewing the change.')
    if not asset.exists() or hashlib.sha256(asset.read_bytes()).hexdigest() != digest:
        raise SystemExit(f'{asset.relative_to(ROOT).as_posix()} is missing or differs; rerun with --write.')
    print(f'GPU graph matches the pinned hash ({replaced} Gelu nodes rewritten, {len(data):,} bytes).')
