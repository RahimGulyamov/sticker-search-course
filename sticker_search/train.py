"""Train residual retrieval adapters on frozen embeddings; select only on dev."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import read_jsonl, set_seed, sha256, write_json, write_jsonl
from .metrics import query_metrics
from .model import RetrievalModel, multi_positive_loss
from .retrieval import SearchIndex


def select_queries(queries, index, split):
    selected = []
    for i, q in enumerate(queries):
        if q["split"] != split:
            continue
        positives = q["positive_item_ids"]
        if "qrels" in q and {key for key, grade in q["qrels"].items() if grade > 0} != set(positives):
            raise ValueError("qrels and positive_item_ids disagree")
        if not positives or not all(p in index.id_to_pos for p in positives):
            continue
        if any(index.items[index.id_to_pos[p]]["split"] != split for p in positives):
            raise ValueError("A query's positive item is in another split")
        selected.append(i)
    return selected


def validation(model, vectors, features, queries, selected, ids, batch_size=64):
    results = []
    model.eval()
    with torch.inference_mode():
        candidates = model.encode_items(*features)
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start+batch_size]
            scores = (model.encode_query(vectors[batch]) @ candidates.T).cpu().numpy()
            for qi, score in zip(batch, scores):
                ranked = [ids[j] for j in np.argsort(-score, kind="stable")[:10]]
                results.append(query_metrics(ranked, {p:1 for p in queries[qi]["positive_item_ids"]})["NDCG@5"])
    return float(np.mean(results))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--queries", required=True, help="Directory created by features queries")
    p.add_argument("--output", default="runs/adapter")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.temperature <= 0:
        p.error("epochs>=1, batch-size>=2 and temperature>0 are required")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    index = SearchIndex(args.index)
    queries = read_jsonl(Path(args.queries) / "queries.jsonl")
    q_np = np.load(Path(args.queries) / "queries.npy", allow_pickle=False)
    if q_np.shape != (len(queries),512):
        raise ValueError("Query vectors/order mismatch")
    train_ids = select_queries(queries,index,"train")
    dev_ids = [i for i in select_queries(queries,index,"dev") if queries[i]["label_source"] != "vlm_weak"]
    if len(train_ids) < 2 or not dev_ids:
        raise ValueError("Need training queries and independent dev source queries")
    device = args.device
    features = [torch.from_numpy(x).to(device) for x in [index.image,index.caption,index.ocr]]
    qv = torch.from_numpy(q_np).to(device)
    model = RetrievalModel(rank=args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    positive_sets = [{index.id_to_pos[p] for p in q["positive_item_ids"] if p in index.id_to_pos} for q in queries]
    # Identical query strings in a batch must not turn their other positives into negatives.
    positives_by_text = {}
    for i in train_ids:
        positives_by_text.setdefault(queries[i]["text"].strip().lower(),set()).update(positive_sets[i])
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out / "best.pt").exists():
        raise ValueError("Experiment exists; choose a new --output")
    rng = np.random.default_rng(args.seed)
    initial = validation(model,qv,features,queries,dev_ids,index.ids)
    best = initial
    architecture = dict(dim=512,rank=args.rank)
    config = dict(architecture=architecture,index_sha256=sha256(Path(args.index)/"vectors.npz"),
                  queries_sha256=sha256(Path(args.queries)/"queries.jsonl"), arguments=vars(args),
                  training_queries=len(train_ids), dev_queries=len(dev_ids),
                  backbone_frozen=True, objective="multi-positive InfoNCE; within-batch negatives; sibling-group masking")
    write_json(out / "config.json",config)

    def save(epoch, score, filename="best.pt"):
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        architecture=architecture,index_sha256=config["index_sha256"],
                        epoch=epoch,dev_ndcg5=score),out/filename)

    save(0,initial)  # Training is allowed to lose to the untouched baseline.
    history = [dict(epoch=0,dev_ndcg5=initial,loss=None)]
    for epoch in range(1,args.epochs+1):
        model.train()
        order = rng.permutation(train_ids).tolist()
        losses = []
        for start in range(0,len(order),args.batch_size):
            batch = order[start:start+args.batch_size]
            if len(batch) < 2:
                continue
            candidate_ids = sorted(set().union(*(positives_by_text[queries[i]["text"].strip().lower()] for i in batch)))
            if len(candidate_ids) < 2:
                continue
            positive = np.array([[j in positives_by_text[queries[i]["text"].strip().lower()] for j in candidate_ids] for i in batch])
            valid = np.array([[index.items[j]["group_id"] != queries[i]["group_id"] for j in candidate_ids] for i in batch]) | positive
            logits = model(qv[batch],*[x[candidate_ids] for x in features])
            loss = multi_positive_loss(logits,torch.as_tensor(positive,device=device),torch.as_tensor(valid,device=device),args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        score = validation(model,qv,features,queries,dev_ids,index.ids)
        record = dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,dev_ndcg5=score)
        history.append(record)
        write_jsonl(out/"history.jsonl",history)
        print(json.dumps(record),flush=True)
        if score > best:
            best = score
            save(epoch,score)
    save(args.epochs,score,"last.pt")
    write_json(out/"summary.json",dict(initial_dev_ndcg5=initial,best_dev_ndcg5=best,
               note="Test split was not consulted. If best epoch is 0, trained updates did not improve dev."))


if __name__ == "__main__":
    main()
