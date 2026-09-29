"""Portable checksum-verified catalog shards, preserving original relative paths."""
import argparse
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from .common import read_jsonl, sha256, write_json


def add(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = len(data), 0, 0o644
    tar.addfile(info, io.BytesIO(data))


def pack(root, output, shard_size=500):
    root, output = Path(root), Path(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Bundle output must be outside the catalog directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(root / "manifest.jsonl")
    inventory = []
    for start in range(0,len(rows),shard_size):
        target = output / f"images-{start//shard_size:05}.tar"
        temporary = target.with_suffix(".tmp")
        with tarfile.open(temporary,"w") as tar:
            for row in rows[start:start+shard_size]:
                name = row["image_path"]
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe image path")
                if sha256(root/name) != row["file_sha256"]:
                    raise ValueError(f"Changed image: {name}")
                add(tar,name,(root/name).read_bytes())
        temporary.replace(target)
        inventory.append(dict(file=target.name,bytes=target.stat().st_size,sha256=sha256(target)))
        print(f"Packed {min(start+shard_size,len(rows))}/{len(rows)}",flush=True)
    metadata = output / "metadata.tar"
    with tarfile.open(metadata,"w") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.relative_to(root).parts[0] == "images":
                add(tar,path.relative_to(root).as_posix(),path.read_bytes())
    inventory.append(dict(file=metadata.name,bytes=metadata.stat().st_size,sha256=sha256(metadata)))
    write_json(output/"bundle.json",dict(version=1,num_images=len(rows),files=inventory))


def unpack(bundle, output):
    bundle, output = Path(bundle), Path(output)
    inventory = json.loads((bundle/"bundle.json").read_text())
    output.mkdir(parents=True,exist_ok=True)
    for entry in inventory["files"]:
        if PurePosixPath(entry["file"]).name != entry["file"]:
            raise ValueError("Unsafe shard filename")
        path = bundle/entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Corrupt shard: {path.name}")
        with tarfile.open(path,"r") as tar:
            for member in tar:
                pure = PurePosixPath(member.name)
                if not member.isfile() or pure.is_absolute() or ".." in pure.parts:
                    raise ValueError("Unsafe tar member")
                destination = output/member.name
                if not destination.resolve().is_relative_to(output.resolve()):
                    raise ValueError("Destination escapes output directory")
                destination.parent.mkdir(parents=True,exist_ok=True)
                contents = tar.extractfile(member).read()
                if destination.exists() and destination.read_bytes() != contents:
                    raise ValueError(f"Refusing to overwrite changed file: {destination}")
                destination.write_bytes(contents)
    rows = read_jsonl(output/"manifest.jsonl")
    if len(rows) != inventory["num_images"]:
        raise ValueError("Restored image count mismatch")
    for row in rows:
        if sha256(output/row["image_path"]) != row["file_sha256"]:
            raise ValueError("Restored image checksum mismatch")
    print(f"Restored and verified {len(rows)} images")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["pack","unpack"])
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--shard-size",type=int,default=500)
    args = p.parse_args()
    if args.shard_size < 1:
        p.error("shard-size must be positive")
    if args.action == "pack":
        pack(args.input,args.output,args.shard_size)
    else:
        unpack(args.input,args.output)


if __name__ == "__main__":
    main()
