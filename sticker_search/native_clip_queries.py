"""Encode original English queries with the already cached native CLIP encoder.

This is an EN-only ablation, not a Russian-query method and not a replacement
for the translated Russian evaluation. Image vectors remain unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import model_revision, read_jsonl, sha256, write_json, write_jsonl
from .features import IMAGE_MODEL, TEXT_MODEL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="runs/full_v2/queries_en")
    p.add_argument("--index", default="runs/full_v2/index")
    p.add_argument("--output", default="runs/retrieval_v3/native_en_queries")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise ValueError("Native query output exists; choose another --output")
    index_meta = json.loads((Path(args.index) / "metadata.json").read_text())
    if index_meta["image_model"] != IMAGE_MODEL or index_meta["image_revision"] != model_revision(IMAGE_MODEL):
        raise ValueError("Native text encoder must match the indexed CLIP image model/revision")
    queries = read_jsonl(source / "queries.jsonl")
    if any(q["language"] != "en" for q in queries):
        raise ValueError("Native CLIP experiment requires the EN-only query directory")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    model = SentenceTransformer(IMAGE_MODEL, revision=model_revision(IMAGE_MODEL), device=args.device,
                                local_files_only=True)
    texts = [q["text"] for q in queries]
    # CLIP has a short context. Explicit truncation prevents errors on long
    # source strings without modifying the original query records.
    tokenizer = model[0].processor.tokenizer
    max_length = model[0].model.config.text_config.max_position_embeddings
    shortened = []
    truncated = []
    for row, text in zip(queries, texts):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) + 2 > max_length:
            truncated.append(row["query_id"])
            tokens = tokens[:max_length - 2]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            # Guard against tokenization changes caused by decode/re-encode.
            while len(tokenizer(text, add_special_tokens=False)["input_ids"]) + 2 > max_length:
                tokens = tokens[:-1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
        shortened.append(text)
    vectors = model.encode(shortened, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    if vectors.shape != (len(queries), 512) or not np.isfinite(vectors).all():
        raise ValueError("Invalid native CLIP query matrix")
    old_vectors = np.load(source / "queries.npy", allow_pickle=False)
    if old_vectors.shape != vectors.shape:
        raise ValueError("Native/multilingual query alignment mismatch")
    dev = [i for i, q in enumerate(queries) if q["split"] == "dev"]
    agreement = np.sum(old_vectors[dev] * vectors[dev], axis=1)
    output.mkdir(parents=True)
    np.save(output / "queries.npy", vectors)
    write_jsonl(output / "queries.jsonl", queries)
    write_json(output / "metadata.json", dict(source_sha256=sha256(source / "queries.jsonl"),
        model=IMAGE_MODEL, revision=model_revision(IMAGE_MODEL), experiment="native_CLIP_EN_only",
        truncated_query_ids=truncated, max_text_tokens=max_length))
    write_json(output / "encoder_comparison.json", dict(native_model=IMAGE_MODEL, multilingual_model=TEXT_MODEL,
        native_architecture=str(model), dev_query_count=len(dev),
        paired_cosine_mean=float(agreement.mean()), paired_cosine_p05=float(np.quantile(agreement,.05)),
        paired_cosine_p95=float(np.quantile(agreement,.95)),
        note="Native versus saved multilingual embeddings for the SAME original English strings. No Russian strings evaluated."))
    print(f"Native CLIP EN queries saved: {output}; {len(truncated)} truncated", flush=True)


if __name__ == "__main__":
    main()
