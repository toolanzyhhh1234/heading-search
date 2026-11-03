from pathlib import Path
import requests

destination = Path(__file__).resolve().parents[1] / 'third-party'
destination.mkdir(exist_ok=True)
sources = {
    'onnxruntime-LICENSE.txt': 'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.17.3/LICENSE',
    'onnxruntime-ThirdPartyNotices.txt': 'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.17.3/ThirdPartyNotices.txt',
    'bge-LICENSE.txt': 'https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/LICENSE',
}
for filename, url in sources.items():
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    (destination / filename).write_text(response.text, encoding='utf-8')
    print(filename, len(response.content), flush=True)
