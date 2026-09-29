"""Compare retrieval methods over the SAME catalog and held-out query set."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import read_jsonl, sha256, write_json, write_jsonl
from .metrics import aggregate, query_metrics
from .retrieval import SearchIndex
from .train import select_queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--calibration", help="Optional TRAIN-fitted query mean directory")
    p.add_argument("--split", choices=["dev","test"], default="test")
    p.add_argument("--methods", nargs="+", default=["image","hybrid","lexical"])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    index = SearchIndex(args.index,args.checkpoint,calibration=args.calibration)
    queries = read_jsonl(Path(args.queries)/"queries.jsonl")
    vectors = np.load(Path(args.queries)/"queries.npy",allow_pickle=False)
    selected = [i for i in select_queries(queries,index,args.split) if queries[i]["label_source"] != "vlm_weak"]
    if not selected:
        raise ValueError("No independent held-out queries with known positives inside the index")
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    summaries, all_cases = {}, []
    for method in args.methods:
        records, times = [], []
        for i in selected:
            q = queries[i]
            start = time.perf_counter()
            scores = index.scores(vectors[i:i+1],method,[q["text"]])[0]
            top = index.topk(scores,10)
            times.append((time.perf_counter()-start)*1000)
            qrels = q.get("qrels",{key:1 for key in q["positive_item_ids"]})
            m = query_metrics([row["item_id"] for row,_ in top],qrels)
            case = dict(method=method,query_id=q["query_id"],query=q["text"],language=q["language"],
                        group_id=q["group_id"],metrics=m,positive_item_ids=q["positive_item_ids"],
                        top10=[dict(item_id=row["item_id"],score=score) for row,score in top])
            records.append(case)
        summaries[method] = dict(metrics=aggregate(records),
            search_latency_ms=dict(p50=float(np.median(times)),p95=float(np.quantile(times,.95)),
                                   includes="scoring and top-k; excludes query encoder and image loading"),
            query_count=len(records),group_count=len({r["group_id"] for r in records}))
        all_cases.extend(records)
    write_json(out/"metrics.json",dict(split=args.split,catalog_size=len(index.items),
        index_sha256=sha256(Path(args.index)/"vectors.npz"), methods=summaries,
        protocol="Known-positive retrieval unless qrels supplied. Unjudged alternatives treated as 0; metrics are not complete human relevance estimates.",
        language_counts={lang:sum(queries[i]["language"]==lang for i in selected) for lang in {queries[i]["language"] for i in selected}}))
    write_jsonl(out/"cases.jsonl",all_cases)
    print(json.dumps({method:{k:round(v['mean'],4) for k,v in s['metrics'].items()} for method,s in summaries.items()},indent=2))


if __name__ == "__main__":
    main()
