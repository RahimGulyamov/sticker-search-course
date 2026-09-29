"""Fit a reproducible candidate-score offset on source TRAIN queries only."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .common import read_jsonl, sha256, write_json
from .fusion import fit_query_mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", default="runs/full_v2/index")
    p.add_argument("--queries", default="runs/full_v2/queries")
    p.add_argument("--output", default="runs/retrieval_v3/calibration")
    args = p.parse_args()
    root, index, out = map(Path, (args.queries, args.index, args.output))
    if out.exists():
        raise ValueError("Calibration exists; choose a new --output")
    queries = read_jsonl(root / "queries.jsonl")
    vectors = np.load(root / "queries.npy", allow_pickle=False)
    index_meta = json.loads((index / "metadata.json").read_text())
    query_meta = json.loads((root / "metadata.json").read_text())
    if query_meta["source_sha256"] != sha256(root / "queries.jsonl"):
        raise ValueError("Query source hash mismatch")
    if query_meta["model"] != index_meta["text_model"] or query_meta["revision"] != index_meta["text_revision"]:
        raise ValueError("Query/catalog encoder mismatch")
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs")
    mean, positions = fit_query_mean(queries, vectors)
    if mean.shape != (512,):
        raise ValueError("Expected 512-dimensional features")
    meta = dict(index_sha256=sha256(index / "vectors.npz"),
        queries_sha256=sha256(root / "queries.jsonl"), vectors_sha256=sha256(root / "queries.npy"),
        text_model=query_meta["model"], text_revision=query_meta["revision"],
        fit_split="train", query_count=len(positions),
        language_counts=dict(Counter(queries[i]["language"] for i in positions)),
        query_ids=[queries[i]["query_id"] for i in positions],
        score_formula="(query - mean_train_query) dot item; no renormalization",
        note="Unsupervised background-score correction, not a fine-tuned encoder. Dev/test do not fit the mean.")
    out.mkdir(parents=True)
    np.save(out / "query_mean.npy", mean)
    write_json(out / "metadata.json", meta)
    print(f"Calibration fitted on {len(positions)} TRAIN queries: {out}", flush=True)


if __name__ == "__main__":
    main()
