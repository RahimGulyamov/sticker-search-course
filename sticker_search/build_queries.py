"""Combine human-source queries, their translations and TRAIN-only VLM weak labels."""
import argparse
from pathlib import Path

from .common import read_jsonl, write_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--catalog",default="data/catalog")
    p.add_argument("--translated")
    p.add_argument("--annotations")
    p.add_argument("--output",default="data/queries_all.jsonl")
    args=p.parse_args()
    root=Path(args.catalog)
    items={r["item_id"]:r for r in read_jsonl(root/"manifest.jsonl")}
    queries=read_jsonl(root/"labels/queries_en.jsonl")
    if args.translated:
        queries+=read_jsonl(args.translated)
    if args.annotations:
        for a in read_jsonl(args.annotations):
            item=items[a["item_id"]]
            if item["split"] != "train":
                continue
            for i,text in enumerate(a.get("usage_queries_ru",[])):
                queries.append(dict(query_id=f"{item['item_id']}_weak_ru_{i}",text=text,language="ru",
                    positive_item_ids=[item["item_id"]],group_id=item["group_id"],split="train",label_source="vlm_weak"))
    if len({q["query_id"] for q in queries})!=len(queries):
        raise ValueError("Duplicate query IDs")
    for q in queries:
        if any(items[p]["split"]!=q["split"] for p in q["positive_item_ids"]):
            raise ValueError("Cross-split positive")
    write_jsonl(args.output,queries)
    print(f"Saved {len(queries)} queries")


if __name__=="__main__":
    main()
