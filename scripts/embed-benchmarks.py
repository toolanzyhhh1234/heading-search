"""Full-corpus CPU encoding with the exact pinned ONNX graph, tokenizer, and pooling.

The JavaScript benchmark separately records the shipping WASM runtime's latency
and numerical agreement, rather than attributing native timings to the plugin.
"""
import hashlib
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['scifact', 'nfcorpus'])
parser.add_argument('--workers', type=int, default=1, choices=[1,2,3])
args = parser.parse_args()
spec = json.loads((ROOT / 'src/model-spec.json').read_text())
directory = ROOT / '.cache/models' / spec['id'] / spec['revision']
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(directory / 'onnx/model_quantized.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
tokenizer.enable_truncation(max_length=spec['maxTokens'])

def encode_one(item):
    text, query = item
    before = time.perf_counter()
    encoded = tokenizer.encode((spec['queryPrefix'] if query else '') + text)
    ids = np.array([encoded.ids], dtype=np.int64)
    feeds = {'input_ids': ids, 'attention_mask': np.ones_like(ids), 'token_type_ids': np.zeros_like(ids)}
    output = session.run(None, feeds)[0]
    vector = output[0, 0, :].astype(np.float32)
    vector /= max(np.linalg.norm(vector), 1e-12)
    return vector, 1000*(time.perf_counter()-before)

for dataset in ([args.dataset] if args.dataset else ['scifact', 'nfcorpus']):
    base = ROOT / 'benchmarks/data' / dataset
    corpus = [json.loads(line) for line in (base / 'corpus.jsonl').read_text().splitlines()]
    qrels = {line.split('\t')[0] for line in (base / 'qrels/test.tsv').read_text().splitlines()[1:]}
    queries = [json.loads(line) for line in (base / 'queries.jsonl').read_text().splitlines()]
    queries = [q for q in queries if q['_id'] in qrels]
    for kind, texts, query in [('corpus', [(d['title']+'\n'+d['text']).strip() for d in corpus], False), ('queries', [q['text'] for q in queries], True)]:
        serialized = json.dumps(texts, ensure_ascii=False, separators=(',', ':'))
        digest = hashlib.sha256((spec['revision'] + serialized + str(query).lower()).encode()).hexdigest()[:16]
        destination = ROOT / 'benchmarks/cache' / f'{dataset}-{kind}-{digest}.f32'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == len(texts)*384*4:
            print(f'{dataset}-{kind}: cached', flush=True)
            continue
        vectors, latencies = [], []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (vector, latency) in enumerate(pool.map(encode_one, ((text, query) for text in texts))):
                vectors.append(vector)
                latencies.append(latency)
                if (i+1) % 500 == 0:
                    print(f'{dataset}-{kind}: {i+1}/{len(texts)}, {time.perf_counter()-started:.1f}s', flush=True)
        np.array(vectors, dtype='<f4').tofile(destination)
        timing = {'totalMs': 1000*(time.perf_counter()-started), 'p50Ms': float(np.percentile(latencies,50)), 'p95Ms': float(np.percentile(latencies,95)), 'backend': f'onnxruntime {ort.__version__} native CPU, one thread per request, {args.workers} concurrent requests; not plugin latency'}
        Path(str(destination)+'.json').write_text(json.dumps(timing))
        print(f'{dataset}-{kind}: finished {len(texts)} in {time.perf_counter()-started:.1f}s', flush=True)
