"""Fetch public BEIR test collections. No private vault data.

Model files are pinned and fetched separately by scripts/pin-model.mjs, from the spec they belong to. (This script
used to pin bge-small itself and, after the switch to Granite, would have overwritten src/model-spec.json.)
"""
from pathlib import Path
import hashlib
import json
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.partial')
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = {}
for dataset in ['scifact', 'nfcorpus']:
    url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip'
    target = ROOT / 'benchmarks' / 'data' / f'{dataset}.zip'
    digest = fetch(url, target)
    with zipfile.ZipFile(target) as archive:
        for name in ['corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv']:
            destination = ROOT / 'benchmarks' / 'data' / dataset / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(f'{dataset}/{name}'))
    sources[dataset] = {'url': url, 'sha256': digest}
    print(f'{dataset}: downloaded and extracted', flush=True)

(ROOT / 'benchmarks' / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n', newline='\n')
print('Benchmark data ready. Fetch the pinned model with: node scripts/pin-model.mjs src/model-spec.json', flush=True)
