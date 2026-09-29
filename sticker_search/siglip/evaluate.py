"""Full-catalog evaluation; checkpoint selection uses only macro EN/RU DEV NDCG@5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..common import read_jsonl, sha256, write_json, write_jsonl
from ..metrics import aggregate, query_metrics
from .data import validate_catalog
from .io import encode_images, encode_texts, load_local, model_identity


def all_images(model, processor, items, catalog, device, config):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    positions = list(range(rank, len(items), world))
    local = encode_images(model, processor, [items[i] for i in positions], catalog,
                          device, config["eval_batch_size"], config["precision"])
    if world > 1:
        gathered = [None] * world if rank == 0 else None
        dist.gather_object((positions,local), gathered, dst=0)
        if rank != 0:
            return None
    else:
        gathered = [(positions,local)]
    result = np.empty((len(items),local.shape[1]), dtype=np.float32)
    for positions, vectors in gathered:
        result[positions] = vectors
    return result


def evaluate_vectors(model, processor, vectors, items, queries, device, config, split, output, method="image"):
    output = Path(output)
    selected = [q for q in queries if q["split"] == split and q["label_source"] != "vlm_weak"]
    if not selected:
        raise ValueError(f"No {split} queries")
    query_vectors = encode_texts(model, processor, [q["text"] for q in selected], device,
        config["eval_batch_size"], config["precision"], config["max_text_length"])
    candidates = torch.from_numpy(vectors).to(device)
    fusion = None
    if method == "image_ocr_rrf":
        from .search import OcrFusion
        fusion = OcrFusion(items)
    records = []
    for offset in range(0,len(selected),64):
        scores = (torch.from_numpy(query_vectors[offset:offset+64]).to(device) @ candidates.T).cpu().numpy()
        for q, score in zip(selected[offset:offset+64], scores):
            if fusion is not None:
                score = fusion.scores(q["text"],score)
            # Stable sort makes ties deterministic across runs.
            order = np.argsort(-score,kind="stable")[:10]
            ranked = [items[int(j)]["item_id"] for j in order]
            qrels = q.get("qrels") or {pid:1 for pid in q["positive_item_ids"]}
            records.append(dict(query_id=q["query_id"], text=q["text"], group_id=q["group_id"],
                language=q["language"], positive_item_ids=q["positive_item_ids"], ranked_ids=ranked,
                scores=[float(score[j]) for j in order], metrics=query_metrics(ranked,qrels)))
    metrics = {lang:aggregate([r for r in records if r["language"] == lang])
               for lang in sorted({r["language"] for r in records})}
    macro = float(np.mean([m["NDCG@5"]["mean"] for m in metrics.values()]))
    if split == "dev" and set(metrics) != {"en","ru"}:
        raise ValueError("Checkpoint selection requires both EN and RU dev queries")
    summary = dict(split=split, method=method, candidate_count=len(items), query_count=len(records),
                   macro_en_ru_image_ndcg5=macro, metrics=metrics,
                   note="Known-positive retrieval; RU queries are translations, not independent human judgments.")
    write_json(output / "metrics.json", summary)
    write_jsonl(output / "cases.jsonl", records)
    return summary


def export_index(path, vectors, items, model_path, config, manifest_hash):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / "images.tmp.npy"
    np.save(tmp, vectors)
    tmp.replace(path / "images.npy")
    write_jsonl(path / "items.jsonl", items)
    write_json(path / "metadata.json", dict(model_path=str(Path(model_path).resolve()),
        model_sha256=model_identity(model_path), manifest_sha256=manifest_hash,
        dimension=vectors.shape[1], max_text_length=config["max_text_length"], precision=config["precision"],
        method="siglip_image", background="white alpha composite", item_ids=[i["item_id"] for i in items]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--catalog", default="data/catalog")
    p.add_argument("--which", choices=["baseline", "selected"], default="selected")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--method",choices=["image","image_ocr_rrf"],default="image")
    p.add_argument("--allow-test", action="store_true", help="Only after fixing all choices on dev")
    a = p.parse_args()
    if a.split == "test" and not a.allow_test:
        p.error("Test is deferred; fix the experiment and use --allow-test explicitly")
    run = Path(a.run)
    if a.which == "selected" and not (run / "complete.json").is_file():
        raise ValueError("Training has not completed; do not evaluate an interim selection on test")
    config = json.loads((run / "config.json").read_text())["config"]
    index_path = run / "baseline" if a.which == "baseline" else Path(json.loads((run/"selection.json").read_text())["index_path"])
    meta = json.loads((index_path / "metadata.json").read_text())
    if sha256(Path(a.catalog) / "manifest.jsonl") != meta["manifest_sha256"]:
        raise ValueError("Catalog differs from trained experiment")
    if model_identity(meta["model_path"]) != meta["model_sha256"]:
        raise ValueError("Indexed model weights changed")
    items = read_jsonl(index_path / "items.jsonl")
    queries = read_jsonl(run / "queries.jsonl")
    validate_catalog(items,queries)
    model, processor = load_local(meta["model_path"],a.device)
    vectors = np.load(index_path / "images.npy",allow_pickle=False)
    output = run / f"{a.which}_{a.split}" if a.method == "image" else run / f"{a.which}_{a.split}_{a.method}"
    result = evaluate_vectors(model,processor,vectors,items,queries,a.device,config,a.split,output,a.method)
    write_json(output / "provenance.json", dict(index_metadata_sha256=sha256(index_path / "metadata.json"),
        queries_sha256=sha256(run / "queries.jsonl")))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
