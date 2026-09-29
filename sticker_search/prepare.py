"""Download the static catalog. Labels are NEVER stored in the searchable manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import download, image_properties, write_json, write_jsonl

SQ_REV = "c28a358226990ef183703bb3797c2b4c5b59e73c"
SQ_URL = f"https://huggingface.co/datasets/metchee/sticker-queries/resolve/{SQ_REV}"
CARD_REV = "94525a359f9edcd25cd062a346595bc039d3c67f"
CARD_URL = f"https://huggingface.co/datasets/gauravs101/synthetic-greeting-cards/resolve/{CARD_REV}"
NYU_REV = "3f58989c03c3898ee31ffc54699bb0726f77e4ee"


def group_and_split(rows, seed=42, image_root=None):
    """Connected components of known families and conservative dHash neighbours.

    The heuristic does not recover complete Telegram packs. Audit the largest
    groups and review examples before calling this a pack-independent benchmark.
    """
    n = len(rows)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)

    families, blocks = {}, defaultdict(list)
    candidate_edges = []
    thumbnails = {}

    def thumbnail(i):
        import numpy as np
        from PIL import Image, ImageChops, ImageOps
        from .common import rgb_image
        if i not in thumbnails:
            image = rgb_image(Path(image_root)/rows[i]["image_path"])
            mask = ImageChops.difference(image, Image.new("RGB",image.size,"white")).convert("L").point(lambda p:255 if p>20 else 0)
            bbox = mask.getbbox()
            if bbox:
                image = image.crop(bbox)
            image = ImageOps.contain(image,(64,64))
            canvas = Image.new("RGB",(64,64),"white")
            canvas.paste(image,((64-image.width)//2,(64-image.height)//2))
            arr = np.asarray(canvas,dtype=np.float32)/255
            thumbnails[i] = (arr, np.min(arr,axis=-1)<.85)
        return thumbnails[i]

    def visual_match(i,j):
        import numpy as np
        a,ma = thumbnail(i);b,mb = thumbnail(j)
        union_mask = ma | mb
        if not union_mask.any():
            return False
        iou = (ma & mb).sum()/union_mask.sum()
        return iou >= .85 and float(np.abs(a-b).mean()) <= .03
    # At Hamming distance <=3 one of four disjoint 16-bit blocks must match.
    for i, row in enumerate(rows):
        family = row.get("source_family")
        if family:
            if family in families:
                union(i, families[family])
            else:
                families[family] = i
        h = int(row["dhash"], 16)
        candidates = set()
        for b in range(4):
            candidates.update(blocks[(b, (h >> (16 * b)) & 65535)])
        for j in candidates:
            other = rows[j]
            if row["kind"] != other["kind"]:
                continue
            # Shared postcard layouts dominate dHash. Known different artwork
            # families must not be merged merely because their frames match.
            if family and other.get("source_family") and family != other["source_family"]:
                continue
            distance = (h ^ int(other["dhash"], 16)).bit_count()
            ratio = (row["width"] / row["height"]) / (other["width"] / other["height"])
            color = max(abs(a - b) for a, b in zip(row["mean_rgb"], other["mean_rgb"]))
            if distance <= 3 and 0.9 <= ratio <= 1.1 and color <= 25:
                if image_root is not None and not visual_match(i,j):
                    continue
                union(i, j)
                candidate_edges.append({"a": row["item_id"], "b": other["item_id"], "dhash_distance": distance})
        for b in range(4):
            blocks[(b, (h >> (16 * b)) & 65535)].append(i)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    for members in groups.values():
        group_id = min(row["item_id"] for row in members)
        fraction = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 2**32
        split = "train" if fraction < 0.7 else "dev" if fraction < 0.85 else "test"
        for row in members:
            row.update(group_id=group_id, split=split)
    return candidate_edges


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    build_config = vars(args).copy()
    config_path = out / "build_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != build_config:
        raise ValueError("Output belongs to a different selection; use another --output")
    write_json(config_path, build_config)
    images = out / "images"
    images.mkdir(exist_ok=True)
    raw = out / "raw"
    rows, source_queries, failures = [], [], []
    en_path = download(f"{SQ_URL}/sticker_queries_en_release.csv", raw / "sq_en.csv")
    download(f"{SQ_URL}/README.md", raw / "stickerqueries_README.md")
    with en_path.open(encoding="utf-8-sig") as stream:
        annotations = list(csv.DictReader(stream))
    eligible = sorted([r for r in annotations if r["sticker_id"].endswith(".png")], key=lambda r: r["sticker_id"])
    selected = random.Random(args.seed).sample(eligible, min(args.sq_limit, len(eligible)))

    def fetch_sq(label):
        name = label["sticker_id"]
        item_id = "sq_" + Path(name).stem
        path = images / (item_id + ".png")
        if args.pilot and not path.exists():
            cached = Path(args.pilot) / "data/images/stickers" / name
            if cached.exists():
                shutil.copyfile(cached, path)
        url = f"{SQ_URL}/stickers/{name}"
        download(url, path)
        row = dict(item_id=item_id, image_path=f"images/{path.name}", kind="sticker",
                   source="StickerQueries", source_revision=SQ_REV, source_url=url,
                   source_family=None, synthetic=False, license_as_declared="MIT (dataset card)")
        row.update(image_properties(path))
        queries = list(dict.fromkeys(q.strip().lower() for q in label["labeled_queries"].split(",") if q.strip()))
        return row, queries

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(label, pool.submit(fetch_sq, label)) for label in selected]
        for i, (label, job) in enumerate(jobs, 1):
            try:
                row, queries = job.result()
                rows.append(row)
                for j, query in enumerate(queries):
                    source_queries.append(dict(query_id=f"{row['item_id']}_en_{j}", text=query,
                                               language="en", item_id=row["item_id"], label_source="human_source"))
            except Exception as e:
                failures.append({"source": "StickerQueries", "asset": label["sticker_id"], "error": str(e)})
            if i % 100 == 0:
                print(f"StickerQueries: {i}/{len(selected)}", flush=True)

    if args.pilot:
        pilot = Path(args.pilot)
        for line in (pilot / "data/manifest.jsonl").read_text().splitlines():
            old = json.loads(line)
            if old["kind"] != "postcard":
                continue
            path = images / (old["item_id"] + ".png")
            shutil.copyfile(pilot / old["image_path"], path)
            row = {k: old[k] for k in ["item_id", "kind", "source", "source_revision", "source_url", "source_family", "synthetic"]}
            row.update(image_path=f"images/{path.name}", license_as_declared="CC-BY-SA-4.0", attribution=old["attribution"])
            row.update(image_properties(path))
            rows.append(row)
        # Preserve attribution without copying evaluation labels to the index.
        shutil.copytree(pilot / "licenses", out / "licenses", dirs_exist_ok=True)

    if args.synth_cards:
        import urllib.request
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/gauravs101/synthetic-greeting-cards/revision/{CARD_REV}", timeout=30) as response:
            siblings = json.load(response)["siblings"]
        names = sorted(f["rfilename"] for f in siblings if f["rfilename"].lower().endswith((".png", ".jpg", ".jpeg")))
        download(f"{CARD_URL}/README.md", raw / "synth_gcd_README.md")
        def fetch_card(name):
            item_id = "gcd_" + hashlib.sha256(name.encode()).hexdigest()[:20]
            path = images / (item_id + Path(name).suffix)
            url = f"{CARD_URL}/{name}"
            download(url, path)
            row = dict(item_id=item_id, image_path=f"images/{path.name}", kind="postcard",
                       source="Synth-GCD", source_revision=CARD_REV, source_url=url,
                       source_family=None, synthetic=True, license_as_declared="Apache-2.0 (dataset card)")
            row.update(image_properties(path))
            return row
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [(name, pool.submit(fetch_card, name)) for name in names]
            for name, job in jobs:
                try:
                    rows.append(job.result())
                except Exception as e:
                    failures.append({"source": "Synth-GCD", "asset": name, "error": str(e)})

    if args.nyu_count:
        from .download_nyu import download_source, candidate_order
        archive = download_source(args.nyu_archive)
        added = 0
        seen_nyu = set()
        with zipfile.ZipFile(archive) as z:
            entries = [x for x in z.infolist() if not x.is_dir() and x.filename.lower().endswith(".png")]
            for entry in candidate_order(entries, args.seed):
                if added >= args.nyu_count:
                    break
                if not 0 < entry.file_size <= 20 * 1024 * 1024:
                    failures.append({"source": "nyuuzyou", "asset": entry.filename, "error": "size_guard"})
                    continue
                item_id = "nyu_" + hashlib.sha256(entry.filename.encode()).hexdigest()[:24]
                path = images / (item_id + ".png")
                if not path.exists():
                    path.write_bytes(z.read(entry))  # zipfile verifies CRC.
                try:
                    props = image_properties(path)
                except Exception as e:
                    failures.append({"source": "nyuuzyou", "asset": entry.filename, "error": str(e)})
                    continue
                if props["pixel_sha256"] in seen_nyu:
                    continue
                seen_nyu.add(props["pixel_sha256"])
                rows.append(dict(item_id=item_id, image_path=f"images/{path.name}", kind="sticker",
                                 source="nyuuzyou/stickers", source_revision=NYU_REV,
                                 source_archive="valid.zip", source_member=entry.filename,
                                 source_family=None, synthetic=False,
                                 emoji_label=entry.filename.split("/")[-2],
                                 license_as_declared="WTFPL (dataset card)", **props))
                added += 1
                if added % 1000 == 0:
                    print(f"Large catalog: {added}/{args.nyu_count}", flush=True)
        if added < args.nyu_count:
            failures.append({"source": "nyuuzyou", "error": "insufficient_unique_items", "wanted": args.nyu_count, "obtained": added})

    # Exact duplicates across sources share one item and all their labels.
    unique, aliases, exact = {}, {}, []
    for row in sorted(rows, key=lambda r: r["item_id"]):
        h = row["pixel_sha256"]
        if h in unique:
            aliases[row["item_id"]] = unique[h]["item_id"]
            exact.append({"duplicate": row["item_id"], "canonical": unique[h]["item_id"]})
        else:
            unique[h] = row
    rows = sorted(unique.values(), key=lambda r: r["item_id"])
    near = group_and_split(rows, args.seed, image_root=out)
    by_id = {r["item_id"]: r for r in rows}
    labeled = []
    for q in source_queries:
        item_id = aliases.get(q["item_id"], q["item_id"])
        item = by_id[item_id]
        labeled.append({k: v for k, v in q.items() if k != "item_id"} | dict(
            positive_item_ids=[item_id], group_id=item["group_id"], split=item["split"]))
    dedup_labels = {}
    for q in labeled:
        dedup_labels.setdefault((q["text"].strip().lower(), q["language"], tuple(q["positive_item_ids"])), q)
    labeled = list(dedup_labels.values())
    write_jsonl(out / "manifest.jsonl", rows)
    write_jsonl(out / "labels/queries_en.jsonl", labeled)
    write_jsonl(out / "audit/failed_downloads.jsonl", failures)
    write_jsonl(out / "audit/exact_duplicates.jsonl", exact)
    write_jsonl(out / "audit/near_duplicates.jsonl", near)
    summary = dict(num_images=len(rows), sources=dict(Counter(r["source"] for r in rows)),
                   kinds=dict(Counter(r["kind"] for r in rows)), splits=dict(Counter(r["split"] for r in rows)),
                   source_query_count=len(labeled), source_png_eligible=len(eligible),
                   source_non_png_excluded=len(annotations)-len(eligible),
                   failed_count=len(failures), exact_duplicate_count=len(exact), near_edges=len(near),
                   group_count=len({r["group_id"] for r in rows}),
                   largest_groups=Counter(r["group_id"] for r in rows).most_common(10),
                   animated_png_first_frame=sum(r["n_frames"] > 1 for r in rows), seed=args.seed,
                   rights_note="Dataset-card declarations do not independently establish third-party artwork rights.",
                   benchmark_note="Own grouped split, NOT original source benchmark. Near-duplicate grouping is heuristic; pack IDs unavailable.")
    write_json(out / "audit/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit("Partial catalog written; inspect audit/failed_downloads.jsonl and rerun to retry.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/catalog")
    p.add_argument("--nyu-count", type=int, default=30000)
    p.add_argument("--nyu-archive", default="downloads/nyuuzyou-valid.zip")
    p.add_argument("--sq-limit", type=int, default=578)
    p.add_argument("--synth-cards", action="store_true")
    p.add_argument("--pilot", help="Optional earlier stage-1 project path (240 OpenMoji cards).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.nyu_count < 0 or args.sq_limit < 0 or args.workers < 1:
        p.error("Counts must be nonnegative and workers positive")
    prepare(args)


if __name__ == "__main__":
    main()
