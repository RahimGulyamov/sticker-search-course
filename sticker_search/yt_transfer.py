"""Transfer a local bundle to/from YT files; credentials come from YT_TOKEN.

Upload is immutable: existing files are reused only if their recorded checksum
matches. A failed write is never marked as complete. bundle.json is written last.
"""
import argparse
import json
import os
import shutil
from contextlib import closing
from pathlib import Path, PurePosixPath

from .common import sha256


def safe_name(name):
    if PurePosixPath(name).name != name or name in {".","..",""}:
        raise ValueError("Bundle entries must be simple file names")
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["upload","download"])
    p.add_argument("--local",required=True)
    p.add_argument("--remote",required=True,help="Versioned YT directory, e.g. //home/.../sticker_project/v1")
    p.add_argument("--proxy",default=os.environ.get("YT_PROXY"))
    args = p.parse_args()
    if not args.proxy or not args.remote.startswith("//") or not args.remote.strip("/"):
        p.error("Provide --proxy/YT_PROXY and a concrete --remote directory")
    import yt.wrapper as yt
    client = yt.YtClient(proxy=args.proxy,token=os.environ.get("YT_TOKEN"))
    remote = args.remote.rstrip("/")
    local = Path(args.local)
    local.mkdir(parents=True,exist_ok=True)
    if args.action == "upload":
        bundle_path = local/"bundle.json"
        bundle = json.loads(bundle_path.read_text())
        files = bundle["files"] + [dict(file="bundle.json",sha256=sha256(bundle_path))]
        client.create("map_node",remote,recursive=True,ignore_existing=True)
        for entry in files:
            name = safe_name(entry["file"])
            path = local/name
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Local checksum mismatch: {name}")
            destination = remote+"/"+name
            if client.exists(destination):
                attribute = destination+"/@project_sha256"
                existing = client.get(attribute) if client.exists(attribute) else None
                if existing != entry["sha256"]:
                    raise ValueError(f"Existing remote file differs or is incomplete: {name}; choose a new versioned directory")
            else:
                # Transaction makes file bytes and completion checksum visible together.
                with client.Transaction():
                    client.create("file",destination)
                    with path.open("rb") as stream:
                        client.write_file(destination,stream)
                    client.set(destination+"/@project_sha256",entry["sha256"])
            print(f"Verified/uploaded {name}",flush=True)
    else:
        with closing(client.read_file(remote+"/bundle.json")) as stream:
            bundle_bytes = stream.read()
        bundle = json.loads(bundle_bytes)
        for entry in bundle["files"]:
            name = safe_name(entry["file"])
            path = local/name
            if path.exists() and sha256(path)==entry["sha256"]:
                continue
            temporary = path.with_suffix(path.suffix+".part")
            with closing(client.read_file(remote+"/"+name)) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src,dst,length=8*1024*1024)
            if sha256(temporary)!=entry["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {name}")
            temporary.replace(path)
            print(f"Downloaded/verified {name}",flush=True)
        (local/"bundle.json").write_bytes(bundle_bytes)


if __name__ == "__main__":
    main()
