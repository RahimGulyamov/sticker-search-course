"""Download a pinned source and build a deduplicated 10k sticker catalog.

Python 3.10+, Pillow. No GPU, Hugging Face account or Telegram token is needed.
"""
from pathlib import Path
from collections import defaultdict
from contextlib import closing
import argparse, hashlib, io, json, random, struct, tarfile, time, urllib.request, zipfile
from PIL import Image

REVISION = '3f58989c03c3898ee31ffc54699bb0726f77e4ee'
SOURCE_URL = f'https://huggingface.co/datasets/nyuuzyou/stickers/resolve/{REVISION}/valid.zip'
SOURCE_SIZE = 6598157862
SOURCE_SHA256 = 'c5ec48956446b9c9aea65fcdc81ce1ce58ffbb86a54496c433bbc050851e47b0'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def download_source(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.stat().st_size==SOURCE_SIZE and sha256(path)==SOURCE_SHA256:return path
        raise RuntimeError(f'Existing archive has wrong size/hash: {path}; preserve it and choose another --archive path')
    partial=path.with_suffix('.zip.part')
    for attempt in range(4):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>SOURCE_SIZE:raise RuntimeError('Partial download is larger than the source')
        if offset==SOURCE_SIZE:break
        headers={'User-Agent':'sticker-course-project/1.0'}
        if offset:headers['Range']=f'bytes={offset}-'
        try:
            req=urllib.request.Request(SOURCE_URL,headers=headers)
            with urllib.request.urlopen(req,timeout=60) as response:
                if offset:
                    cr=response.headers.get('Content-Range','')
                    if response.status!=206 or not cr.startswith(f'bytes {offset}-'):
                        raise RuntimeError('Server did not honor resume Range; partial archive kept')
                elif response.status!=200:
                    raise RuntimeError(f'Unexpected initial response status: {response.status}')
                with partial.open('ab' if offset else 'wb') as out:
                    last=offset
                    while True:
                        block=response.read(8*1024*1024)
                        if not block:break
                        out.write(block);offset+=len(block)
                        if offset>SOURCE_SIZE:raise RuntimeError('Source length changed')
                        if offset-last>=100*1024*1024:
                            print(f'Downloaded {offset/1e9:.2f}/{SOURCE_SIZE/1e9:.2f} GB',flush=True);last=offset
            if partial.stat().st_size==SOURCE_SIZE:break
        except Exception as error:
            print(f'Download attempt {attempt+1}: {error}',flush=True)
            if attempt==3:raise
            time.sleep(2)
    if not partial.exists() or partial.stat().st_size!=SOURCE_SIZE:raise RuntimeError('Incomplete download; rerun to resume')
    if sha256(partial)!=SOURCE_SHA256:raise RuntimeError('SHA-256 mismatch; archive is not used')
    partial.replace(path)
    return path

def candidate_order(files,seed):
    """Cover available emoji categories, then fill proportionally to remaining files."""
    groups=defaultdict(list)
    for item in sorted(files,key=lambda x:x.filename):groups[item.filename.split('/')[-2]].append(item)
    rng=random.Random(seed);first=[];rest=[]
    for key in sorted(groups):
        group=groups[key];rng.shuffle(group);first.append(group[0]);rest.extend(group[1:])
    rng.shuffle(first);rng.shuffle(rest)
    return first+rest
