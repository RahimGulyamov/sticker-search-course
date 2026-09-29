"""Stream large regular files through checksum-verified YT-compatible tar bundles."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from .common import sha256, write_json


def safe_path(root, name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or not pure.parts or '\\' in name:
        raise ValueError(f'Unsafe relative path: {name}')
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes bundle root')
    return path


def pack_files(root, names, output, shard_bytes=2 * 1024**3):
    root, output = Path(root), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Bundle output must be outside its input')
    names = sorted(set(names))
    if not names:
        raise ValueError('No files selected')
    output.mkdir(parents=True, exist_ok=True)
    chunks, chunk, size = [], [], 0
    for name in names:
        path = safe_path(root, name)
        if not path.is_file():
            raise ValueError(f'Missing file: {name}')
        length = path.stat().st_size
        if chunk and size + length > shard_bytes:
            chunks.append(chunk); chunk, size = [], 0
        chunk.append(name); size += length
    if chunk:
        chunks.append(chunk)
    shards, members = [], []
    for i, names_chunk in enumerate(chunks):
        target = output / f'files-{i:05d}.tar'
        temp = target.with_suffix('.part')
        with tarfile.open(temp, 'w') as tar:
            for name in names_chunk:
                path = safe_path(root, name)
                digest = sha256(path)
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                with path.open('rb') as src:
                    tar.addfile(info, src)
                members.append(dict(path=name, bytes=info.size, sha256=digest))
        temp.replace(target)
        shards.append(dict(file=target.name, bytes=target.stat().st_size, sha256=sha256(target)))
        print(f'Packed {i+1}/{len(chunks)}: {target.name}', flush=True)
    write_json(output / 'bundle.json', dict(version=1, kind='regular_files', files=shards, members=members))


def unpack_files(bundle, output):
    bundle, output = Path(bundle), Path(output)
    info = json.loads((bundle / 'bundle.json').read_text())
    if info.get('kind') != 'regular_files':
        raise ValueError('Wrong bundle kind')
    expected = {r['path']: r for r in info['members']}
    if len(expected) != len(info['members']):
        raise ValueError('Duplicate manifest paths')
    seen = set()
    output.mkdir(parents=True, exist_ok=True)
    for shard in info['files']:
        if PurePosixPath(shard['file']).name != shard['file']:
            raise ValueError('Unsafe shard name')
        source = safe_path(bundle, shard['file'])
        if sha256(source) != shard['sha256']:
            raise ValueError('Corrupt shard')
        with tarfile.open(source, 'r') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError('Unexpected or repeated tar member')
                target = safe_path(output, member.name)
                record = expected[member.name]
                if member.size != record['bytes']:
                    raise ValueError('Member size mismatch')
                seen.add(member.name)
                if target.exists():
                    if sha256(target) != record['sha256']:
                        raise ValueError(f'Refusing to replace a different file: {target}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + '.part')
                with tar.extractfile(member) as src, temp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=8*1024*1024)
                if sha256(temp) != record['sha256']:
                    raise ValueError('Extracted checksum mismatch')
                temp.replace(target)
        print(f'Restored {shard["file"]}', flush=True)
    if seen != set(expected):
        raise ValueError('Incomplete bundle')
    print(f'Restored and verified {len(seen)} files', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['pack', 'unpack'])
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.action == 'unpack':
        unpack_files(args.input, args.output)
    else:
        root = Path(args.input)
        pack_files(root, [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()], args.output)


if __name__ == '__main__':
    main()
