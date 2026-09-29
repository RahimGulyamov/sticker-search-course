"""Frozen CLIP image + aligned multilingual text representations.

The catalog contains only image-derived captions/OCR, never source queries.
All vectors are 512-dimensional, L2-normalized; absent text maps to zero.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import model_revision, read_jsonl, rgb_image, sha256, write_json, write_jsonl

IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def text_encoder(device=None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL, revision=model_revision(TEXT_MODEL), device=device)


def encode_nonempty(model, texts, batch_size):
    result = np.zeros((len(texts), 512), dtype=np.float32)
    indices = [i for i, text in enumerate(texts) if text.strip()]
    if indices:
        result[indices] = model.encode([texts[i] for i in indices], batch_size=batch_size,
                                       normalize_embeddings=True, show_progress_bar=False)
    return result


def build(args):
    import torch
    from sentence_transformers import SentenceTransformer

    if args.threads:
        torch.set_num_threads(args.threads)
    root, out = Path(args.catalog), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(root / "manifest.jsonl")
    if args.limit:
        # Independent of retrieval labels; deterministic mixed-source pilot.
        from .common import fingerprint
        rows = sorted(rows, key=lambda r: fingerprint([42, r["item_id"]]))[:args.limit]
    rows = sorted(rows, key=lambda r: r["item_id"])
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate item IDs")
    annotations = {r["item_id"]: r for r in read_jsonl(args.annotations)} if args.annotations else {}
    search_rows = []
    for row in rows:
        annotation = annotations.get(row["item_id"], {})
        caption = ". ".join(annotation.get(k, "").strip() for k in ["caption_ru", "emotion_ru", "style_ru"] if annotation.get(k, "").strip())
        search_rows.append(row | dict(caption_text=caption, ocr_text=annotation.get("ocr_text", "")))
    metadata = dict(manifest_sha256=sha256(root / "manifest.jsonl"),
                    annotation_sha256=sha256(args.annotations) if args.annotations else None,
                    item_ids=ids, image_model=IMAGE_MODEL, text_model=TEXT_MODEL,
                    image_revision=model_revision(IMAGE_MODEL), text_revision=model_revision(TEXT_MODEL),
                    background="white alpha composite", dimension=512)
    meta_path = out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Different index inputs; choose another output directory")
    write_json(meta_path, metadata)
    start = time.perf_counter()
    image_cache = out / "images.npy"
    if image_cache.exists():
        image_vectors = np.load(image_cache, allow_pickle=False)
        if image_vectors.shape != (len(rows), 512):
            raise ValueError("Image cache shape mismatch")
    else:
        model = SentenceTransformer(IMAGE_MODEL, revision=model_revision(IMAGE_MODEL), device=args.device)
        image_vectors = np.empty((len(rows), 512), dtype=np.float32)
        for offset in range(0, len(rows), args.batch_size):
            images = [rgb_image(root / r["image_path"]) for r in rows[offset:offset+args.batch_size]]
            image_vectors[offset:offset+len(images)] = model.encode(images, batch_size=args.batch_size,
                                                                   normalize_embeddings=True, show_progress_bar=False)
            if offset == 0 or (offset // args.batch_size) % 50 == 0:
                print(f"Image embeddings: {min(offset+args.batch_size,len(rows))}/{len(rows)}", flush=True)
        temporary = out / "images.tmp.npy"
        np.save(temporary, image_vectors)
        temporary.replace(image_cache)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model = text_encoder(args.device)
    captions = encode_nonempty(model, [r["caption_text"] for r in search_rows], args.batch_size)
    ocr = encode_nonempty(model, [r["ocr_text"] for r in search_rows], args.batch_size)
    np.savez_compressed(out / "vectors.npz", image=image_vectors, caption=captions, ocr=ocr)
    write_jsonl(out / "items.jsonl", search_rows)
    write_json(out / "timing.json", dict(seconds=time.perf_counter()-start, num_items=len(rows), device=str(model.device)))
    print(f"Index built: {len(rows)} images -> {out}", flush=True)


def encode_queries(args):
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    source = read_jsonl(args.queries)
    if len({q["query_id"] for q in source}) != len(source):
        raise ValueError("Duplicate query IDs")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = text_encoder(args.device)
    vectors = encode_nonempty(model, [q["text"] for q in source], args.batch_size)
    np.save(out / "queries.npy", vectors)
    write_jsonl(out / "queries.jsonl", source)
    write_json(out / "metadata.json", dict(source_sha256=sha256(args.queries), model=TEXT_MODEL, revision=model_revision(TEXT_MODEL)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["index", "queries"])
    p.add_argument("--catalog", default="data/catalog")
    p.add_argument("--annotations")
    p.add_argument("--queries")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.action == "queries" and not args.queries:
        p.error("--queries is required")
    (build if args.action == "index" else encode_queries)(args)


if __name__ == "__main__":
    main()
