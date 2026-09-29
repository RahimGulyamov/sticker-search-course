"""Download pinned model revisions into HF_HOME for later offline transfer.

Example: HF_HOME=/data/hf_cache python scripts/cache_models.py --preset retrieval
Transfer that HF_HOME directory unchanged (including symlinks) to the GPU node.
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

p=argparse.ArgumentParser()
p.add_argument('--preset',choices=['retrieval','annotation','generation','all'],default='retrieval')
args=p.parse_args()
if not os.environ.get('HF_HOME'):
    p.error('Set HF_HOME to the directory that you will transfer')
revisions=json.loads((Path(__file__).resolve().parents[1]/'configs/model_revisions.json').read_text())
presets={
 'retrieval':['sentence-transformers/clip-ViT-B-32','sentence-transformers/clip-ViT-B-32-multilingual-v1'],
 'annotation':['Qwen/Qwen2.5-VL-7B-Instruct'],
 'generation':['Qwen/Qwen2.5-7B-Instruct','black-forest-labs/FLUX.1-schnell'],
}
models=list(revisions) if args.preset=='all' else presets[args.preset]
for model in models:
    path=snapshot_download(model,revision=revisions[model],
        ignore_patterns=['onnx/*','openvino/*','*.h5','*.ot','*.msgpack','*.md','*.png','*.jpg'])
    print(model,path,flush=True)
if args.preset in {'generation','all'}:
    from rembg import new_session
    new_session('u2net',providers=['CPUExecutionProvider'])
    print('Also transfer U2NET_HOME (or ~/.u2net) for offline background removal.')
