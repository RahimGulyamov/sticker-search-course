"""Restore an exact manifest snapshot, including original image bytes and hashes.

For NYU this uses the pinned source archive; no different random sample is drawn.
OpenMoji derivatives are included under evidence/openmoji_pilot for exact restoration.
"""
import argparse
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import download,read_jsonl,sha256
from .download_nyu import download_source


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',default='evidence/full_catalog/manifest.jsonl')
    p.add_argument('--output',default='data/catalog')
    p.add_argument('--archive',default='downloads/nyuuzyou-valid.zip')
    p.add_argument('--pilot',default='evidence/openmoji_pilot')
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    snapshot=Path(args.snapshot);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(snapshot)
    nyu=[r for r in rows if r['source']=='nyuuzyou/stickers']
    def valid(row):
        path=out/row['image_path']
        return path.exists() and sha256(path)==row['file_sha256']
    if nyu:
        source=download_source(args.archive)
        with zipfile.ZipFile(source) as z:
            for i,row in enumerate(nyu,1):
                if not valid(row):
                    path=out/row['image_path'];path.parent.mkdir(parents=True,exist_ok=True)
                    data=z.read(row['source_member']);path.write_bytes(data)
                    if not valid(row):raise ValueError('Source image checksum mismatch')
                if i%5000==0:print(f'Restored NYU: {i}/{len(nyu)}',flush=True)
    def restore_other(row):
        if valid(row):return
        target=out/row['image_path'];target.parent.mkdir(parents=True,exist_ok=True)
        if row['source']=='Synthetic-OpenMoji':
            source=Path(args.pilot)/'images'/target.name
            if not source.exists():raise ValueError('OpenMoji derivative missing from pilot bundle')
            shutil.copyfile(source,target)
            if not valid(row):raise ValueError('OpenMoji checksum mismatch')
        else:
            download(row['source_url'],target,row['file_sha256'])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(restore_other,[r for r in rows if r['source']!='nyuuzyou/stickers']))
    # Metadata is copied only after the images are fully restored.
    for path in snapshot.parent.rglob('*'):
        if path.is_file():
            destination=out/path.relative_to(snapshot.parent)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination)
    print(f'Exact catalog restored: {len(rows)} images')


if __name__=='__main__':main()
