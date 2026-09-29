"""Download and bundle pinned model files on a CPU laptop, without importing torch."""
import argparse
import json
from pathlib import Path

from .common import ROOT, sha256, write_json
from .file_bundle import pack_files

PRESETS = {
    'siglip2': ['google/siglip2-so400m-patch14-384'],
    'retrieval': ['sentence-transformers/clip-ViT-B-32',
                  'sentence-transformers/clip-ViT-B-32-multilingual-v1'],
    'annotation': ['Qwen/Qwen2.5-VL-7B-Instruct'],
    'generation': ['Qwen/Qwen2.5-7B-Instruct', 'black-forest-labs/FLUX.1-schnell'],
}
FLUX_FILES = ['model_index.json', 'scheduler/*', 'text_encoder/*', 'text_encoder_2/*',
              'tokenizer/*', 'tokenizer_2/*', 'transformer/*', 'vae/*']
IGNORE = ['*.bin', '*.h5', '*.ot', '*.msgpack', 'onnx/*', 'openvino/*',
          '*.png', '*.jpg', '*.jpeg', '.gitattributes']


def snapshot(root, model, revision):
    return root / 'hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['download', 'pack'])
    p.add_argument('--preset', choices=[*PRESETS, 'all'], default='retrieval')
    p.add_argument('--root', default='data/offline_models')
    p.add_argument('--output', help='Required for pack')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    root = Path(args.root).resolve()
    revisions = json.loads((ROOT / 'configs/model_revisions.json').read_text())
    models = list(revisions) if args.preset == 'all' else PRESETS[args.preset]
    manifest = root / f'assets-{args.preset}.json'
    if args.action == 'download':
        from huggingface_hub import snapshot_download
        files = []
        for model in models:
            folder = snapshot(root, model, revisions[model])
            # Real files in cache-compatible snapshot paths; no symlink portability issue.
            snapshot_download(model, revision=revisions[model], local_dir=folder,
                allow_patterns=FLUX_FILES if model.startswith('black-forest-labs/') else None,
                ignore_patterns=IGNORE, max_workers=args.workers)
            paths = sorted(x for x in folder.rglob('*') if x.is_file() and '.cache' not in x.relative_to(folder).parts)
            if not paths or not any(x.suffix == '.safetensors' for x in paths):
                raise ValueError(f'Incomplete snapshot: {model}')
            # Ensure every referenced weight shard is present.
            for index in folder.rglob('*.safetensors.index.json'):
                for name in set(json.loads(index.read_text())['weight_map'].values()):
                    if not (index.parent / name).is_file():
                        raise ValueError(f'Missing weight shard: {name}')
            for path in paths:
                files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
            print(f'Ready: {model}', flush=True)
        if args.preset in {'generation', 'all'}:
            import pooch
            path = Path(pooch.retrieve(
                'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx',
                known_hash='md5:60024c5c889badc19c04ad937298a77b',
                path=root/'u2net', fname='u2net.onnx', progressbar=True))
            files.append(dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)))
        write_json(manifest, dict(preset=args.preset, models={m:revisions[m] for m in models}, files=files))
        print(f'Downloaded {sum(x["bytes"] for x in files)/1e9:.2f} GB; ready for pack', flush=True)
    else:
        if not args.output:
            p.error('--output is required for pack')
        info = json.loads(manifest.read_text())
        if info['models'] != {m:revisions[m] for m in models}:
            raise ValueError('Model revisions differ from the downloaded manifest')
        for row in info['files']:
            if sha256(root/row['path']) != row['sha256']:
                raise ValueError(f'File changed: {row["path"]}')
        pack_files(root, [r['path'] for r in info['files']] + [manifest.name], args.output)


if __name__ == '__main__':
    main()
