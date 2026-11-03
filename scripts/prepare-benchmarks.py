"""Fetch public BEIR test collections and pin model assets. No private vault data."""
from pathlib import Path
import hashlib
import json
import zipfile
import requests
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--update-model', action='store_true', help='Explicitly repin the model to the current Hub revision.')
args = parser.parse_args()

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

model = 'Xenova/bge-small-en-v1.5'
existing_path = ROOT / 'src' / 'model-spec.json'
existing = json.loads(existing_path.read_text()) if existing_path.exists() and not args.update_model else None
if existing:
    revision = existing['revision']
else:
    response = requests.get(f'https://huggingface.co/api/models/{model}', timeout=60)
    response.raise_for_status()
    revision = response.json()['sha']
files = []
for name in ['tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx']:
    target = ROOT / '.cache' / 'models' / model / revision / name
    url = f'https://huggingface.co/{model}/resolve/{revision}/{name}'
    digest = fetch(url, target)
    if existing:
        expected = next(file for file in existing['files'] if file['path'] == name)
        if digest != expected['sha256']:
            raise ValueError(f'Pinned model asset checksum mismatch: {name}')
    files.append({'path': name, 'sha256': digest, 'bytes': target.stat().st_size})
    print(f'{name}: {target.stat().st_size:,} bytes', flush=True)
spec = {
    'id': model, 'revision': revision, 'dimensions': 384, 'maxTokens': 512,
    'pooling': 'cls', 'queryPrefix': 'Represent this sentence for searching relevant passages: ',
    'license': 'MIT (BAAI upstream model)', 'files': files
}
(ROOT / 'src' / 'model-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
(ROOT / 'benchmarks' / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n')
print('Pinned model specification written.', flush=True)
