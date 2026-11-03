import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
root=Path(__file__).resolve().parents[1]
spec=json.loads((root/'src/model-spec.json').read_text())
directory=root/'.cache/models'/spec['id']/spec['revision']
tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
tokenizer.enable_truncation(max_length=512)
texts=json.loads((root/'benchmarks/cache/parity-input.json').read_text())
reference=np.fromfile(root/'benchmarks/cache/parity-wasm.f32',dtype='<f4').reshape(-1,384)
for name,level in [('disabled',ort.GraphOptimizationLevel.ORT_DISABLE_ALL),('basic',ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),('extended',ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),('all',ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;options.graph_optimization_level=level
    session=ort.InferenceSession(str(directory/'onnx/model_quantized.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    cosine=[]
    for text,other in zip(texts,reference):
        ids=np.array([tokenizer.encode(text).ids],dtype=np.int64)
        vector=session.run(None,{'input_ids':ids,'attention_mask':np.ones_like(ids),'token_type_ids':np.zeros_like(ids)})[0][0,0,:]
        vector=vector/np.linalg.norm(vector)
        cosine.append(float(vector@other))
    print(name, 'min cosine',min(cosine),'mean',np.mean(cosine),flush=True)
